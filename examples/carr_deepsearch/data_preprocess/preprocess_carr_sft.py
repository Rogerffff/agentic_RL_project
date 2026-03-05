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
Preprocess CaRR DeepDive SFT data (deepdive-sft-glm46-trajectory-1k.jsonl) to verl parquet format.

Converts CaRR message format to Qwen3 chat template format:
- assistant.tool_calls: {tool_call_id, name, arguments: dict} -> {type: "function", id, function: {name, arguments: json_string}}
- tool.content: [{output: text}] -> "text" (plain string)
- assistant.content: null -> ""
- reasoning_content: preserved
- tools: loaded from canonical YAML config (carr_browser_tools.yaml) to ensure
  SFT training and RL rollout see identical tool schemas in the system prompt.

Input:  CaRR/data/deepdive-sft-glm46-trajectory-1k.jsonl
Output: sft_train.parquet, sft_val.parquet
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def convert_message(msg: dict) -> dict:
    """Convert a single CaRR message to Qwen3 chat template format."""
    converted = {"role": msg["role"]}

    if msg["role"] == "assistant":
        converted["content"] = msg.get("content") if msg.get("content") is not None else ""
        if msg.get("reasoning_content"):
            converted["reasoning_content"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            converted["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.get("tool_call_id", "call_%d" % i),
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                        if isinstance(tc.get("arguments"), dict)
                        else str(tc.get("arguments", "{}")),
                    },
                }
                for i, tc in enumerate(msg["tool_calls"])
            ]
    elif msg["role"] == "tool":
        content = msg.get("content", "")
        if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
            converted["content"] = content[0].get("output", "")
        else:
            converted["content"] = str(content)
    else:
        converted["content"] = msg.get("content", "")

    return converted


def load_canonical_tools(tool_config_path: str) -> list:
    """Load canonical tool schemas from the YAML config used by RL rollout.

    This ensures SFT training and RL rollout see identical tool schemas
    in the system prompt rendered by apply_chat_template.
    """
    with open(tool_config_path) as f:
        cfg = yaml.safe_load(f)
    return [t["tool_schema"] for t in cfg["tools"]]


def convert_record(record: dict, canonical_tools: list, tokenizer=None) -> Optional[dict]:
    """Convert a single CaRR SFT record to verl parquet format.

    Args:
        record: Raw CaRR SFT record with messages and tools.
        canonical_tools: Tool schemas loaded from carr_browser_tools.yaml.
        tokenizer: Optional tokenizer for validation.

    Returns None if the record fails validation.
    """
    messages = record.get("messages", [])

    if not messages:
        return None

    # Convert messages
    converted_messages = [convert_message(m) for m in messages]

    # Validate with tokenizer if available (enable_thinking=True to match real training)
    if tokenizer is not None:
        try:
            tokenizer.apply_chat_template(
                converted_messages,
                tools=canonical_tools if canonical_tools else None,
                enable_thinking=True,
                tokenize=False,
            )
        except Exception as e:
            logger.warning("apply_chat_template failed: %s", e)
            return None

    result = {
        "messages": converted_messages,
        "tools": canonical_tools,
        "enable_thinking": True,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess CaRR SFT data to verl parquet format")
    parser.add_argument("--input_file", required=True, help="Path to deepdive-sft-glm46-trajectory-1k.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory for parquet files")
    parser.add_argument(
        "--tool_config",
        default=None,
        help="Path to carr_browser_tools.yaml (canonical tool schemas). "
        "Auto-detected relative to this script if not provided.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Tokenizer model name for validation (e.g. Qwen/Qwen3-4B). Skip validation if not provided.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load canonical tool schemas
    if args.tool_config is None:
        args.tool_config = str(
            Path(__file__).resolve().parent.parent / "config" / "tool_config" / "carr_browser_tools.yaml"
        )
    canonical_tools = load_canonical_tools(args.tool_config)
    print(f"Loaded {len(canonical_tools)} canonical tools from {args.tool_config}")
    for t in canonical_tools:
        print(f"  - {t['function']['name']}")

    # Load tokenizer if specified
    tokenizer = None
    if args.model_name:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer from {args.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load data
    print(f"Loading data from {args.input_file}...")
    raw_records = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))
    print(f"Loaded {len(raw_records)} records")

    # Convert all records
    converted = []
    skipped = 0
    for i, record in enumerate(raw_records):
        result = convert_record(record, canonical_tools=canonical_tools, tokenizer=tokenizer)
        if result is None:
            skipped += 1
            logger.warning("Skipped record %d", i)
        else:
            converted.append(result)

    print(f"Converted: {len(converted)}, Skipped: {skipped}")

    # Shuffle and split
    rng = random.Random(args.seed)
    indices = list(range(len(converted)))
    rng.shuffle(indices)

    val_size = int(len(converted) * args.val_ratio)
    val_indices = set(indices[:val_size])

    train_records = [converted[i] for i in range(len(converted)) if i not in val_indices]
    val_records = [converted[i] for i in range(len(converted)) if i in val_indices]

    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    # Save to parquet
    train_path = os.path.join(args.output_dir, "sft_train.parquet")
    val_path = os.path.join(args.output_dir, "sft_val.parquet")

    pd.DataFrame(train_records).to_parquet(train_path)
    pd.DataFrame(val_records).to_parquet(val_path)

    print(f"Saved: {train_path} ({len(train_records)} rows)")
    print(f"Saved: {val_path} ({len(val_records)} rows)")

    # Sanity check
    df = pd.read_parquet(train_path)
    print(f"\nSanity check (train):")
    print(f"  Columns: {list(df.columns)}")
    sample = df.iloc[0]
    msgs = sample["messages"]
    print(f"  messages count (first sample): {len(msgs)}")
    print(f"  first msg role: {msgs[0]['role']}")
    print(f"  last msg role: {msgs[-1]['role']}")
    # Check a tool_calls message exists
    tc_msgs = [m for m in msgs if m.get("tool_calls")]
    if tc_msgs:
        tc = tc_msgs[0]["tool_calls"][0]
        print(f"  tool_call format: type={tc.get('type')}, has id={bool(tc.get('id'))}, "
              f"func name={tc['function']['name']}, args type={type(tc['function']['arguments']).__name__}")
    tools_val = sample.get("tools")
    if tools_val is not None and len(tools_val) > 0:
        t = tools_val[0]
        print(f"  tools format: type={t.get('type')}, func name={t['function']['name']}")
    print(f"  enable_thinking: {sample.get('enable_thinking')}")


if __name__ == "__main__":
    main()
