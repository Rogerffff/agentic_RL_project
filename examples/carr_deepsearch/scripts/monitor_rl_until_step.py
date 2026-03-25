#!/usr/bin/env python3
"""Watch a verl RL run until a target step and emit health alerts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"step:(\d+) - (.*)")
ERROR_RE = re.compile(
    r"(Traceback|AssertionError|RuntimeError|CUDA out of memory|OutOfMemory|"
    r"NCCL|Segmentation fault|ConnectionError|ReadTimeout|MaxRetryError|"
    r"ERROR[: ]|Exception:|failed\b|FATAL)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_line(path: Path, level: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {level} {message}\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def tail_text(path: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    return data.decode("utf-8", errors="ignore")


def parse_latest_step(log_path: Path) -> tuple[int | None, dict[str, float], str | None]:
    text = tail_text(log_path)
    matches = list(STEP_RE.finditer(text))
    if not matches:
        return None, {}, None
    step_match = matches[-1]
    step = int(step_match.group(1))
    metrics: dict[str, float] = {}
    for part in step_match.group(2).split(" - "):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return step, metrics, step_match.group(0)


def get_matching_processes(patterns: list[str]) -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    matches: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, cmd = line.split(maxsplit=1)
            pid = int(pid_str)
        except ValueError:
            continue
        if all(pattern in cmd for pattern in patterns):
            matches.append((pid, cmd))
    return matches


def pid_exists(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def http_get_json(url: str, timeout: float = 8.0) -> tuple[bool, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - watchdog path
        return False, str(exc)


def http_post_json(url: str, payload: dict[str, Any], timeout: float = 12.0) -> tuple[bool, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - watchdog path
        body = exc.read().decode("utf-8", errors="ignore")
        return False, f"{exc.code} {body}"
    except Exception as exc:  # pragma: no cover - watchdog path
        return False, str(exc)


def collect_error_lines(log_path: Path, seen: set[str]) -> list[str]:
    text = tail_text(log_path, max_bytes=1024 * 1024)
    lines = []
    for line in text.splitlines():
        if not ERROR_RE.search(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def seed_seen_errors(log_path: Path) -> set[str]:
    seen: set[str] = set()
    text = tail_text(log_path, max_bytes=1024 * 1024)
    for line in text.splitlines():
        if ERROR_RE.search(line):
            seen.add(line)
    return seen


def build_status(
    *,
    target_step: int,
    latest_step: int | None,
    latest_metrics: dict[str, float],
    trainer_pid: int | None,
    tool_pids: list[int],
    reward_pids: list[int],
    tool_ok: bool,
    tool_stats: Any,
    reward_ok: bool,
    reward_health: Any,
    last_progress_at: float,
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "target_step": target_step,
        "latest_step": latest_step,
        "remaining_steps": None if latest_step is None else max(target_step - latest_step, 0),
        "latest_metrics": latest_metrics,
        "trainer_pid": trainer_pid,
        "trainer_alive": pid_exists(trainer_pid),
        "tool_pids": tool_pids,
        "reward_pids": reward_pids,
        "tool_ok": tool_ok,
        "tool_stats": tool_stats,
        "reward_ok": reward_ok,
        "reward_health": reward_health,
        "last_progress_at_utc": datetime.fromtimestamp(last_progress_at, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--trainer-pid", type=int, default=None)
    parser.add_argument("--tool-port", type=int, default=7230)
    parser.add_argument("--reward-port", type=int, default=8888)
    parser.add_argument("--poll-seconds", type=int, default=45)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--event-log", required=True)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    status_file = Path(args.status_file)
    event_log = Path(args.event_log)

    if not log_path.exists():
        append_line(event_log, "ERROR", f"log file missing: {log_path}")
        return 1

    tool_stats_url = f"http://127.0.0.1:{args.tool_port}/stats"
    reward_url = f"http://127.0.0.1:{args.reward_port}/evaluate"
    reward_payload = {
        "history": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "label": "a",
        "task_unfinished": True,
        "remote_env_info": {
            "search_forbidden_strs": ["q"],
            "rubrics": [],
            "rubric_reward_ratio": 0.3,
        },
    }

    latest_step, latest_metrics, latest_line = parse_latest_step(log_path)
    seen_errors = seed_seen_errors(log_path)
    last_progress_step = latest_step
    last_progress_at = time.time()
    missing_trainer_polls = 0

    append_line(
        event_log,
        "INFO",
        f"watchdog start target_step={args.target_step} initial_step={latest_step} poll={args.poll_seconds}s stall={args.stall_seconds}s",
    )
    if latest_line:
        append_line(event_log, "INFO", f"initial latest step line: {latest_line}")

    while True:
        trainer_pid = args.trainer_pid
        if trainer_pid is None:
            matches = get_matching_processes(["verl.trainer.main_ppo", "config-name=carr_grpo"])
            trainer_pid = matches[0][0] if matches else None

        tool_matches = get_matching_processes(["CaRR/tool_server/launch_server.py", f"--port {args.tool_port}"])
        reward_matches = get_matching_processes(["launch_server.py", f"--port {args.reward_port}"])

        if pid_exists(trainer_pid):
            missing_trainer_polls = 0
        else:
            missing_trainer_polls += 1
            append_line(event_log, "ERROR", f"trainer pid missing: {trainer_pid}")
            if missing_trainer_polls >= 3:
                append_line(event_log, "FATAL", "trainer missing for 3 consecutive polls, stopping watchdog")
                return 2

        step, metrics, step_line = parse_latest_step(log_path)
        if step is not None and step != last_progress_step:
            last_progress_step = step
            last_progress_at = time.time()
            append_line(event_log, "INFO", f"step advanced to {step}: outcome={metrics.get('outcome_reward/mean')} rubric={metrics.get('rubric_reward/mean')} unfinished={metrics.get('task_unfinished/ratio')} step_s={metrics.get('timing_s/step')}")

            if metrics.get("task_unfinished/ratio", 0.0) > 0.65:
                append_line(event_log, "WARN", f"high unfinished ratio at step {step}: {metrics['task_unfinished/ratio']:.3f}")
            if metrics.get("parse_error_count/mean", 0.0) > 0.20:
                append_line(event_log, "WARN", f"high parse error count at step {step}: {metrics['parse_error_count/mean']:.3f}")
            if metrics.get("termination_response_limit/ratio", 0.0) > 0.45:
                append_line(event_log, "WARN", f"response limit dominates at step {step}: {metrics['termination_response_limit/ratio']:.3f}")
            if metrics.get("termination_search_budget/ratio", 0.0) > 0.20:
                append_line(event_log, "WARN", f"search budget hit ratio high at step {step}: {metrics['termination_search_budget/ratio']:.3f}")
            if metrics.get("timing_s/step", 0.0) > 650:
                append_line(event_log, "WARN", f"slow step at step {step}: {metrics['timing_s/step']:.1f}s")

        if time.time() - last_progress_at > args.stall_seconds:
            append_line(
                event_log,
                "WARN",
                f"no step progress for {int(time.time() - last_progress_at)}s; latest_step={last_progress_step}",
            )
            last_progress_at = time.time()

        tool_ok, tool_stats = http_get_json(tool_stats_url)
        if not tool_ok:
            append_line(event_log, "ERROR", f"tool server stats failed: {tool_stats}")

        reward_ok, reward_health = http_post_json(reward_url, reward_payload)
        if not reward_ok:
            append_line(event_log, "ERROR", f"reward health failed: {reward_health}")

        for line in collect_error_lines(log_path, seen_errors):
            append_line(event_log, "ERROR", f"log-match: {line}")

        status = build_status(
            target_step=args.target_step,
            latest_step=step,
            latest_metrics=metrics,
            trainer_pid=trainer_pid,
            tool_pids=[pid for pid, _ in tool_matches],
            reward_pids=[pid for pid, _ in reward_matches],
            tool_ok=tool_ok and bool(tool_matches),
            tool_stats=tool_stats,
            reward_ok=reward_ok and bool(reward_matches),
            reward_health=reward_health,
            last_progress_at=last_progress_at,
        )
        write_json(status_file, status)

        if step is not None and step >= args.target_step:
            append_line(event_log, "INFO", f"target step reached: {step} >= {args.target_step}")
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
