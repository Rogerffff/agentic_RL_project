#!/usr/bin/env python3
"""Summarize sampled eval JSONL dumps and choose an async start checkpoint.

Each candidate is passed as:
  --candidate "<label>::<model_path>::<eval_dir>"

The selector reads the latest JSONL file in each eval_dir and computes:
  - outcome_reward mean
  - task_unfinished mean
  - termination_rollout_timeout mean

Selection rule:
  1. Keep only candidates with task_unfinished <= 0.70 and
     termination_rollout_timeout <= 0.45.
  2. Rank by outcome_reward mean descending.
  3. If the top candidates are within 0.03 outcome_reward, prefer lower
     task_unfinished, then lower termination_rollout_timeout.
  4. If nobody passes the gate, fall back to label == "sft" when available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_candidate(spec: str) -> dict[str, str]:
    parts = spec.split("::", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"Invalid --candidate {spec!r}. Expected '<label>::<model_path>::<eval_dir>'."
        )
    label, model_path, eval_dir = parts
    return {"label": label, "model_path": model_path, "eval_dir": eval_dir}


def _jsonl_sort_key(path: Path) -> tuple[int, float]:
    try:
        step = int(path.stem)
    except ValueError:
        step = -1
    return (step, path.stat().st_mtime)


def _latest_jsonl(eval_dir: Path) -> Path:
    files = sorted(eval_dir.glob("*.jsonl"), key=_jsonl_sort_key)
    if not files:
        raise FileNotFoundError(f"No JSONL files found under {eval_dir}")
    return files[-1]


def _mean(entries: list[dict[str, Any]], key: str) -> float:
    if not entries:
        return 0.0
    total = 0.0
    for entry in entries:
        total += float(entry.get(key, 0.0) or 0.0)
    return total / len(entries)


def _load_summary(candidate: dict[str, str]) -> dict[str, Any]:
    eval_dir = Path(candidate["eval_dir"]).expanduser().resolve()
    jsonl_path = _latest_jsonl(eval_dir)
    entries: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        raise ValueError(f"Latest JSONL file is empty: {jsonl_path}")

    outcome_reward_mean = _mean(entries, "outcome_reward")
    task_unfinished_mean = _mean(entries, "task_unfinished")
    termination_rollout_timeout_mean = _mean(entries, "termination_rollout_timeout")

    return {
        "label": candidate["label"],
        "model_path": str(Path(candidate["model_path"]).expanduser().resolve()),
        "eval_dir": str(eval_dir),
        "jsonl_path": str(jsonl_path),
        "sample_count": len(entries),
        "outcome_reward_mean": outcome_reward_mean,
        "task_unfinished_mean": task_unfinished_mean,
        "termination_rollout_timeout_mean": termination_rollout_timeout_mean,
        "eligible": task_unfinished_mean <= 0.70 and termination_rollout_timeout_mean <= 0.45,
    }


def _choose_candidate(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    eligible = [summary for summary in summaries if summary["eligible"]]
    if not eligible:
        sft = next((summary for summary in summaries if summary["label"] == "sft"), None)
        if sft is not None:
            return sft, "no candidate passed unfinished/timeout gate; fallback to sft"
        selected = max(
            summaries,
            key=lambda item: (
                item["outcome_reward_mean"],
                -item["task_unfinished_mean"],
                -item["termination_rollout_timeout_mean"],
            ),
        )
        return selected, "no candidate passed gate; fallback to highest outcome_reward"

    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["outcome_reward_mean"],
            item["task_unfinished_mean"],
            item["termination_rollout_timeout_mean"],
            item["label"],
        ),
    )
    best = ranked[0]
    near_best = [
        item
        for item in ranked
        if best["outcome_reward_mean"] - item["outcome_reward_mean"] <= 0.03 + 1e-9
    ]
    if len(near_best) == 1:
        return best, "highest eligible outcome_reward"

    selected = min(
        near_best,
        key=lambda item: (
            item["task_unfinished_mean"],
            item["termination_rollout_timeout_mean"],
            -item["outcome_reward_mean"],
            item["label"],
        ),
    )
    return selected, "top eligible outcome_reward candidates were within 0.03; tie-broken by unfinished then timeout"


def _write_text(path: str | None, content: str) -> None:
    if not path:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=_parse_candidate,
        help="Candidate spec: '<label>::<model_path>::<eval_dir>'",
    )
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--selected-path-out", type=str, default=None)
    parser.add_argument("--export-sh", type=str, default=None)
    args = parser.parse_args()

    summaries = [_load_summary(candidate) for candidate in args.candidate]
    selected, reason = _choose_candidate(summaries)

    result = {
        "selected_label": selected["label"],
        "selected_model_path": selected["model_path"],
        "selection_reason": reason,
        "candidates": summaries,
    }

    if args.output_json:
        _write_text(args.output_json, json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.selected_path_out:
        _write_text(args.selected_path_out, selected["model_path"] + "\n")
    if args.export_sh:
        export_text = (
            f'export ASYNC_START_MODEL_PATH="{selected["model_path"]}"\n'
            f'export ASYNC_START_MODEL_LABEL="{selected["label"]}"\n'
        )
        _write_text(args.export_sh, export_text)

    print("Async start candidate summaries:")
    for summary in summaries:
        print(
            "  - {label}: outcome={outcome:.4f}, unfinished={unfinished:.4f}, timeout={timeout:.4f}, "
            "samples={samples}, eligible={eligible}, jsonl={jsonl}".format(
                label=summary["label"],
                outcome=summary["outcome_reward_mean"],
                unfinished=summary["task_unfinished_mean"],
                timeout=summary["termination_rollout_timeout_mean"],
                samples=summary["sample_count"],
                eligible=summary["eligible"],
                jsonl=summary["jsonl_path"],
            )
        )

    print(f"Selected async start: {selected['label']} -> {selected['model_path']}")
    print(f"Selection reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
