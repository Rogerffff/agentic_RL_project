#!/usr/bin/env python3
"""
Estimate Jina Reader token usage for CaRR DeepSearch on local macOS/Linux.

This script replays a realistic subset of the tool chain:
    query -> Serper/SerpAPI search -> Jina Reader open

It measures two different token views:
1. raw_reader_tokens:
   Tokens in the full Jina Reader response. This is the best proxy for billing,
   because Reader usage is counted on output response tokens.
2. truncated_reader_tokens:
   Tokens after CaRR tool server truncates browser.open output to 10,000 chars.
   This is the best proxy for what the agent actually sees in-context.

Then it projects total Jina token usage for formal RL runs using configurable
"open calls per trajectory" scenarios.

Recommended usage:
    export JINA_API_KEY=...
    export SERPER_API_KEY=...
    python examples/carr_deepsearch/scripts/estimate_jina_usage.py \
        --dataset-file examples/carr_deepsearch/data/rl_train.parquet \
        --sample-size 100 \
        --open-ranks 0,1 \
        --open-per-traj-scenarios 0.5,1.0,1.5 \
        --num-runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_SERVER_DIR = PROJECT_ROOT / "CaRR" / "tool_server"
if str(TOOL_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_SERVER_DIR))

from web_search import parse_url, search, search_serper  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Jina Reader token usage for CaRR DeepSearch.")
    parser.add_argument(
        "--dataset-file",
        default="examples/carr_deepsearch/data/rl_train.parquet",
        help="Parquet file used to sample representative questions.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of sampled questions used for search->open replay.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--search-backend",
        choices=["serper", "serpapi"],
        default="serper",
        help="Search backend used to obtain URLs before Jina opens them.",
    )
    parser.add_argument(
        "--search-num",
        type=int,
        default=5,
        help="How many search results to request per query.",
    )
    parser.add_argument(
        "--open-ranks",
        default="0,1",
        help="Comma-separated search result ranks to open per sampled query, e.g. '0' or '0,1,2'.",
    )
    parser.add_argument(
        "--max-open-per-query",
        type=int,
        default=2,
        help="Hard cap on number of URLs to open per sampled query after applying open-ranks.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Concurrent Jina open requests for local estimation. Keep modest to avoid rate limits.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=["segmenter", "chars4"],
        default="segmenter",
        help="Token counting method. 'segmenter' uses Jina Segmenter (preferred, free). 'chars4' uses len(text)/4.",
    )
    parser.add_argument(
        "--open-per-traj-scenarios",
        default="0.5,1.0,1.5",
        help="Comma-separated overall open calls per trajectory scenarios for projection.",
    )
    parser.add_argument("--train-size", type=int, default=2123, help="RL train dataset size.")
    parser.add_argument("--epochs", type=int, default=3, help="RL epochs per formal run.")
    parser.add_argument("--train-batch-size", type=int, default=128, help="RL train_batch_size.")
    parser.add_argument("--rollout-n", type=int, default=16, help="RL rollout.n.")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of formal RL runs to budget for. Use 2 for GRPO + C-GRPO.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write detailed JSON results.",
    )
    return parser.parse_args()


def extract_query(row: pd.Series) -> str | None:
    extra_info = row.get("extra_info", {})
    if isinstance(extra_info, dict):
        search_forbidden = extra_info.get("search_forbidden_strs", [])
        if search_forbidden:
            return str(search_forbidden[0])

    prompt = row.get("prompt", None)
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    return None


def sample_queries(dataset_file: str, sample_size: int, seed: int) -> list[str]:
    df = pd.read_parquet(dataset_file)
    queries = []
    seen = set()
    for _, row in df.iterrows():
        query = extract_query(row)
        if query and query not in seen:
            seen.add(query)
            queries.append(query)

    rng = random.Random(seed)
    rng.shuffle(queries)
    return queries[: min(sample_size, len(queries))]


def parse_open_ranks(raw: str) -> list[int]:
    ranks = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ranks.append(int(part))
    return ranks


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def chars4_token_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    rank = (len(values) - 1) * p
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (rank - lo))


def is_tool_error(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("failed to") or lowered.startswith("an unexpected error occurred")


async def count_tokens_segmenter(
    session: aiohttp.ClientSession,
    text: str,
    jina_api_key: str | None,
) -> int:
    headers = {"Content-Type": "application/json"}
    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"

    async with session.post(
        "https://segment.jina.ai/",
        headers=headers,
        json={"content": text},
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        body = await resp.text()

    if "application/json" in content_type:
        data = json.loads(body)
        # Be liberal in what we accept; Segmenter response shape may evolve.
        for key in ("num_tokens", "token_count", "count", "tokens"):
            val = data.get(key)
            if isinstance(val, int):
                return val
            if isinstance(val, list):
                return len(val)
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            for key in ("total_tokens", "num_tokens", "token_count"):
                val = usage.get(key)
                if isinstance(val, int):
                    return val
        raise ValueError(f"Unrecognized Segmenter JSON response keys: {list(data.keys())}")

    match = re.search(r"(\d+)\s+token", body.lower())
    if match:
        return int(match.group(1))

    match = re.search(r"^\s*(\d+)\s*$", body)
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not parse Segmenter response: {body[:200]!r}")


async def count_tokens(
    session: aiohttp.ClientSession,
    text: str,
    tokenizer_mode: str,
    jina_api_key: str | None,
) -> int:
    if tokenizer_mode == "chars4":
        return chars4_token_estimate(text)
    try:
        return await count_tokens_segmenter(session, text, jina_api_key)
    except Exception:
        return chars4_token_estimate(text)


async def fetch_search_results(
    query: str,
    search_backend: str,
    search_num: int,
    serper_api_key: str | None,
    serpapi_api_key: str | None,
) -> dict[int, str]:
    if search_backend == "serper":
        _, idx2url = await search_serper(
            query=query,
            num=search_num,
            forbidden_strs=[query],
            serper_api_key=serper_api_key,
        )
    else:
        _, idx2url = await search(
            query=query,
            num=search_num,
            forbidden_strs=[query],
            serp_api_key=serpapi_api_key,
        )
    return idx2url


async def process_single_open(
    query: str,
    rank: int,
    url: str,
    semaphore: asyncio.Semaphore,
    token_session: aiohttp.ClientSession,
    jina_api_key: str,
    tokenizer_mode: str,
) -> dict:
    async with semaphore:
        started = time.perf_counter()
        raw_text = await parse_url(url=url, forbidden_strs=[query], jina_api_key=jina_api_key)
        latency_s = time.perf_counter() - started

        if not raw_text or is_tool_error(raw_text):
            return {
                "query": query,
                "rank": rank,
                "url": url,
                "domain": urlparse(url).netloc,
                "ok": False,
                "latency_s": latency_s,
                "error": raw_text[:200] if raw_text else "empty response",
            }

        truncated_text = raw_text[:10000]
        raw_tokens = await count_tokens(token_session, raw_text, tokenizer_mode, jina_api_key)
        truncated_tokens = await count_tokens(token_session, truncated_text, tokenizer_mode, jina_api_key)

        return {
            "query": query,
            "rank": rank,
            "url": url,
            "domain": urlparse(url).netloc,
            "ok": True,
            "latency_s": latency_s,
            "raw_chars": len(raw_text),
            "truncated_chars": len(truncated_text),
            "raw_reader_tokens": raw_tokens,
            "truncated_reader_tokens": truncated_tokens,
        }


def summarize_open_records(open_records: list[dict]) -> dict:
    success = [r for r in open_records if r.get("ok")]
    failed = [r for r in open_records if not r.get("ok")]
    raw_tokens = [r["raw_reader_tokens"] for r in success]
    trunc_tokens = [r["truncated_reader_tokens"] for r in success]
    raw_chars = [r["raw_chars"] for r in success]
    domains: dict[str, int] = {}
    for r in success:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1

    top_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "queries_replayed": len({r["query"] for r in open_records}),
        "open_attempts": len(open_records),
        "open_success": len(success),
        "open_failures": len(failed),
        "success_rate": (len(success) / len(open_records)) if open_records else 0.0,
        "latency_mean_s": statistics.mean([r["latency_s"] for r in success]) if success else 0.0,
        "raw_chars_mean": statistics.mean(raw_chars) if raw_chars else 0.0,
        "raw_tokens_mean": statistics.mean(raw_tokens) if raw_tokens else 0.0,
        "raw_tokens_median": statistics.median(raw_tokens) if raw_tokens else 0.0,
        "raw_tokens_p90": percentile(raw_tokens, 0.9),
        "raw_tokens_p95": percentile(raw_tokens, 0.95),
        "truncated_tokens_mean": statistics.mean(trunc_tokens) if trunc_tokens else 0.0,
        "truncated_tokens_median": statistics.median(trunc_tokens) if trunc_tokens else 0.0,
        "truncated_tokens_p90": percentile(trunc_tokens, 0.9),
        "truncated_tokens_p95": percentile(trunc_tokens, 0.95),
        "top_domains": top_domains,
    }


def project_total_trajectories(train_size: int, epochs: int, train_batch_size: int, rollout_n: int, num_runs: int) -> int:
    steps_per_run = math.ceil(train_size * epochs / train_batch_size)
    return steps_per_run * train_batch_size * rollout_n * num_runs


def project_usage(summary: dict, total_trajectories: int, open_per_traj_scenarios: list[float]) -> list[dict]:
    rows = []
    raw_mean = summary["raw_tokens_mean"]
    raw_p90 = summary["raw_tokens_p90"]
    trunc_mean = summary["truncated_tokens_mean"]
    for open_per_traj in open_per_traj_scenarios:
        total_opens = total_trajectories * open_per_traj
        rows.append({
            "open_per_trajectory": open_per_traj,
            "projected_open_calls": int(round(total_opens)),
            "reader_tokens_mean_budget": int(round(total_opens * raw_mean)),
            "reader_tokens_p90_budget": int(round(total_opens * raw_p90)),
            "agent_visible_tokens_mean_budget": int(round(total_opens * trunc_mean)),
        })
    return rows


async def main_async(args: argparse.Namespace) -> dict:
    jina_api_key = os.environ.get("JINA_API_KEY")
    serper_api_key = os.environ.get("SERPER_API_KEY")
    serpapi_api_key = os.environ.get("SERPAPI_API_KEY")

    if not jina_api_key:
        raise SystemExit("JINA_API_KEY is required.")
    if args.search_backend == "serper" and not serper_api_key:
        raise SystemExit("SERPER_API_KEY is required for --search-backend=serper.")
    if args.search_backend == "serpapi" and not serpapi_api_key:
        raise SystemExit("SERPAPI_API_KEY is required for --search-backend=serpapi.")

    queries = sample_queries(args.dataset_file, args.sample_size, args.seed)
    open_ranks = parse_open_ranks(args.open_ranks)
    open_ranks = open_ranks[: args.max_open_per_query]
    open_per_traj_scenarios = parse_float_list(args.open_per_traj_scenarios)

    print(f"Loaded {len(queries)} sampled queries from {args.dataset_file}")
    print(f"Search backend: {args.search_backend}")
    print(f"Open ranks per query: {open_ranks}")
    print(f"Tokenizer mode: {args.tokenizer_mode}")

    semaphore = asyncio.Semaphore(args.concurrency)
    open_records: list[dict] = []
    token_timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=token_timeout) as token_session:
        for idx, query in enumerate(queries, start=1):
            print(f"[{idx}/{len(queries)}] search: {query[:100]!r}")
            idx2url = await fetch_search_results(
                query=query,
                search_backend=args.search_backend,
                search_num=args.search_num,
                serper_api_key=serper_api_key,
                serpapi_api_key=serpapi_api_key,
            )
            selected = [(rank, idx2url[rank]) for rank in open_ranks if rank in idx2url]
            if not selected:
                open_records.append({
                    "query": query,
                    "rank": None,
                    "url": None,
                    "domain": None,
                    "ok": False,
                    "latency_s": 0.0,
                    "error": "no_search_results_for_selected_ranks",
                })
                continue

            tasks = [
                process_single_open(
                    query=query,
                    rank=rank,
                    url=url,
                    semaphore=semaphore,
                    token_session=token_session,
                    jina_api_key=jina_api_key,
                    tokenizer_mode=args.tokenizer_mode,
                )
                for rank, url in selected
            ]
            open_records.extend(await asyncio.gather(*tasks))

    summary = summarize_open_records(open_records)
    total_trajectories = project_total_trajectories(
        train_size=args.train_size,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        rollout_n=args.rollout_n,
        num_runs=args.num_runs,
    )
    projections = project_usage(summary, total_trajectories, open_per_traj_scenarios)

    result = {
        "config": {
            "dataset_file": args.dataset_file,
            "sample_size": args.sample_size,
            "search_backend": args.search_backend,
            "search_num": args.search_num,
            "open_ranks": open_ranks,
            "tokenizer_mode": args.tokenizer_mode,
            "concurrency": args.concurrency,
            "train_size": args.train_size,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "rollout_n": args.rollout_n,
            "num_runs": args.num_runs,
            "open_per_traj_scenarios": open_per_traj_scenarios,
        },
        "summary": summary,
        "total_trajectories_projected": total_trajectories,
        "projections": projections,
        "sample_open_records": open_records[:20],
    }
    return result


def print_projection_table(result: dict) -> None:
    summary = result["summary"]
    print("\n=== Jina Reader Open Summary ===")
    print(f"queries_replayed:       {summary['queries_replayed']}")
    print(f"open_attempts:          {summary['open_attempts']}")
    print(f"open_success:           {summary['open_success']}")
    print(f"open_failures:          {summary['open_failures']}")
    print(f"success_rate:           {summary['success_rate']:.2%}")
    print(f"latency_mean_s:         {summary['latency_mean_s']:.2f}")
    print(f"raw_tokens_mean:        {summary['raw_tokens_mean']:.1f}")
    print(f"raw_tokens_median:      {summary['raw_tokens_median']:.1f}")
    print(f"raw_tokens_p90:         {summary['raw_tokens_p90']:.1f}")
    print(f"truncated_tokens_mean:  {summary['truncated_tokens_mean']:.1f}")
    print(f"truncated_tokens_p90:   {summary['truncated_tokens_p90']:.1f}")

    print("\n=== Projection (formal RL) ===")
    print(f"total_trajectories_projected: {result['total_trajectories_projected']}")
    print(
        "open/traj | projected_opens | reader_tokens(mean) | reader_tokens(p90) | agent_visible_tokens(mean)"
    )
    for row in result["projections"]:
        print(
            f"{row['open_per_trajectory']:>8.2f} | "
            f"{row['projected_open_calls']:>15,d} | "
            f"{row['reader_tokens_mean_budget']:>19,d} | "
            f"{row['reader_tokens_p90_budget']:>18,d} | "
            f"{row['agent_visible_tokens_mean_budget']:>26,d}"
        )

    if summary["top_domains"]:
        print("\nTop domains:")
        for domain, count in summary["top_domains"]:
            print(f"  {domain}: {count}")


def main() -> None:
    args = parse_args()
    result = asyncio.run(main_async(args))
    print_projection_table(result)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nSaved detailed report to {output_path}")


if __name__ == "__main__":
    main()
