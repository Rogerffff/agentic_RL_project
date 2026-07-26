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
"""Verify SFT and RL tool schema consistency through their REAL execution paths.

SFT path: parquet -> convert_nested_value_to_list_recursive -> normalize_tool_schema -> apply_chat_template
RL path:  YAML -> OpenAIFunctionToolSchema.model_validate -> model_dump(exclude_none) -> normalize_tool_schema -> apply_chat_template

Usage:
    python examples/carr_deepsearch/scripts/verify_tool_consistency.py \
        --sft_parquet examples/carr_deepsearch/data/sft_train.parquet \
        --tool_config examples/carr_deepsearch/config/tool_config/carr_browser_tools.yaml \
        --model_name Qwen/Qwen3-4B
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd


def convert_nested_value_to_list_recursive(data_item):
    """Exact copy of multiturn_sft_dataset.py logic."""
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        return data_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_parquet", required=True)
    parser.add_argument("--tool_config", required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from verl.tools.schemas import OpenAIFunctionToolSchema, normalize_tool_schema

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # --- SFT path (real): parquet -> convert -> normalize ---
    df = pd.read_parquet(args.sft_parquet)
    raw_tools = df.iloc[0]["tools"]
    sft_tools_raw = convert_nested_value_to_list_recursive(raw_tools)
    sft_tools = [normalize_tool_schema(t) for t in sft_tools_raw]

    # --- RL path (real): YAML -> OmegaConf -> model_validate -> model_dump -> normalize ---
    tools_config = OmegaConf.load(args.tool_config)
    rl_tools = []
    for tool_config in tools_config.tools:
        tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        rl_tools.append(normalize_tool_schema(tool_schema.model_dump(exclude_unset=True, exclude_none=True)))

    # --- Render both ---
    test_msg = [{"role": "user", "content": "test"}]
    sft_text = tokenizer.apply_chat_template(
        test_msg, tools=sft_tools, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    rl_text = tokenizer.apply_chat_template(
        test_msg, tools=rl_tools, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    def extract_system(text):
        match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", text, re.DOTALL)
        return match.group(1).strip() if match else "NOT FOUND"

    sft_sys = extract_system(sft_text)
    rl_sys = extract_system(rl_text)

    # --- Compare ---
    print("=" * 70)
    print("SFT tools (normalized, first tool):")
    print(json.dumps(sft_tools[0], indent=2))
    print()
    print("RL tools (normalized, first tool):")
    print(json.dumps(rl_tools[0], indent=2))
    print()

    identical = sft_sys == rl_sys
    print("=" * 70)
    print(f"IDENTICAL {identical}")
    print("=" * 70)

    if not identical:
        sft_lines = sft_sys.split("\n")
        rl_lines = rl_sys.split("\n")
        print(f"SFT: {len(sft_lines)} lines, RL: {len(rl_lines)} lines")
        for i in range(max(len(sft_lines), len(rl_lines))):
            s = sft_lines[i] if i < len(sft_lines) else "<MISSING>"
            r = rl_lines[i] if i < len(rl_lines) else "<MISSING>"
            if s != r:
                print(f"\n  Line {i} differs:")
                print(f"    SFT: {s[:200]}")
                print(f"     RL: {r[:200]}")
        sys.exit(1)
    else:
        print("SFT and RL system prompts are IDENTICAL through real execution paths!")


if __name__ == "__main__":
    main()
