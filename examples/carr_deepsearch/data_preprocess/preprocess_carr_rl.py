# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess CaRR DeepDive RL data (deepdive-rl-2k-rubrics.jsonl) to verl parquet format.

Input:  CaRR/data/deepdive-rl-2k-rubrics.jsonl
Output: rl_train.parquet, rl_val.parquet
"""

import argparse
import json
import os

import pandas as pd


from typing import Optional


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def convert_record(record: dict, idx: int, split: str) -> Optional[dict]:
    """Convert a single CaRR RL record to verl parquet format."""
    input_messages = record.get("input_messages", [])
    label = record.get("label", "")
    metadata = record.get("metadata", {})
    remote_env_info = metadata.get("remote_env_info", {})

    rubrics = remote_env_info.get("rubrics", [])
    search_forbidden_strs = remote_env_info.get("search_forbidden_strs", [])

    # search_forbidden_strs must be non-empty (CaRR reward server uses [0] as question)
    if not search_forbidden_strs:
        print(f"WARNING: record {idx} has empty search_forbidden_strs, skipping")
        return None

    # Build tools_kwargs for all 3 browser tools
    tools_kwargs = {
        "browser.search": {
            "create_kwargs": {
                "search_forbidden_strs": search_forbidden_strs,
            },
        },
        "browser.open": {
            "create_kwargs": {
                "search_forbidden_strs": search_forbidden_strs,
            },
        },
    }

    return {
        "data_source": "carr_deepsearch",
        "agent_name": "carr_tool_agent",
        "prompt": input_messages,
        "ability": "deepsearch",
        "reward_model": {"style": "rule", "ground_truth": label},
        "extra_info": {
            "split": split,
            "index": idx,
            "rubrics": rubrics,
            "search_forbidden_strs": search_forbidden_strs,
            "rubric_reward_ratio": 0.3,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess CaRR RL data to verl parquet format")
    parser.add_argument("--input_file", required=True, help="Path to deepdive-rl-2k-rubrics.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.input_file}...")
    raw_records = load_jsonl(args.input_file)
    print(f"Loaded {len(raw_records)} records")

    # Shuffle with fixed seed and split
    import random

    rng = random.Random(args.seed)
    indices = list(range(len(raw_records)))
    rng.shuffle(indices)

    val_size = int(len(raw_records) * args.val_ratio)
    val_indices = set(indices[:val_size])

    train_records = []
    val_records = []
    skipped = 0

    for i, record in enumerate(raw_records):
        split = "val" if i in val_indices else "train"
        converted = convert_record(record, idx=i, split=split)
        if converted is None:
            skipped += 1
            continue
        if i in val_indices:
            val_records.append(converted)
        else:
            train_records.append(converted)

    print(f"Train: {len(train_records)}, Val: {len(val_records)}, Skipped: {skipped}")

    # Save to parquet
    train_path = os.path.join(args.output_dir, "rl_train.parquet")
    val_path = os.path.join(args.output_dir, "rl_val.parquet")

    pd.DataFrame(train_records).to_parquet(train_path)
    pd.DataFrame(val_records).to_parquet(val_path)

    print(f"Saved: {train_path} ({len(train_records)} rows)")
    print(f"Saved: {val_path} ({len(val_records)} rows)")

    # Sanity check
    df = pd.read_parquet(train_path)
    print(f"\nSanity check (train):")
    print(f"  Columns: {list(df.columns)}")
    print(f"  agent_name unique: {df['agent_name'].unique()}")
    sample_extra = df.iloc[0]["extra_info"]
    print(f"  extra_info keys: {list(sample_extra.keys())}")
    print(f"  rubrics count (first): {len(sample_extra.get('rubrics', []))}")
    print(f"  search_forbidden_strs non-empty: {bool(sample_extra.get('search_forbidden_strs'))}")
    print(f"  tools_kwargs keys: {list(sample_extra.get('tools_kwargs', {}).keys())}")


if __name__ == "__main__":
    main()
