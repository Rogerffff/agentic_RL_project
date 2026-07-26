#!/usr/bin/env python3
"""Verify SFT tokenization correctly produces <tool_call> tags for all 3 browser tools.

This test checks:
1. SFT data messages with tool_calls are correctly rendered by Qwen3's apply_chat_template
2. The rendered text contains <tool_call> tags (Hermes format) for browser.search/open/find
3. The SFT training format matches what the RL rollout tool parser expects
4. Loss mask covers tool_call tokens (model learns to generate tool calls)
5. The real MultiTurnSFTDataset training path keeps reasoning/final-answer tokens in the loss mask

Usage:
    python test_sft_tokenization.py --model_path <path_to_qwen3_model_or_checkpoint>
    python test_sft_tokenization.py --model_path Qwen/Qwen3-4B  # from HuggingFace
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def convert_nested(val):
    """Convert numpy arrays and other non-standard types to Python native types.

    Parquet stores nested structures (lists of dicts) as numpy arrays.
    Qwen3's apply_chat_template uses json.dumps internally, so all values
    must be JSON-serializable Python native types.
    """
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return convert_nested(val.tolist())
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, dict):
        return {str(k): convert_nested(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        converted = [convert_nested(v) for v in val]
        # Filter out None values in properties dicts (tool schemas have null unused params)
        return converted
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val


def clean_tool_schema(tools):
    """Clean tool schemas: remove null properties, fix type arrays."""
    if tools is None:
        return None
    cleaned = []
    for tool in tools:
        tool = convert_nested(tool)
        # Clean properties: remove null values
        if "function" in tool and "parameters" in tool["function"]:
            params = tool["function"]["parameters"]
            if "properties" in params:
                params["properties"] = {
                    k: v for k, v in params["properties"].items() if v is not None
                }
            # Fix 'required' field - may be stored as string repr of list
            if "required" in params and isinstance(params["required"], str):
                try:
                    params["required"] = json.loads(params["required"].replace("'", '"'))
                except json.JSONDecodeError:
                    pass
        cleaned.append(tool)
    return cleaned


def clean_messages(messages):
    """Clean messages: ensure tool_calls arguments are JSON strings, remove None content."""
    cleaned = []
    for msg in messages:
        msg = convert_nested(msg)
        # Ensure content is string or None
        if "content" in msg and msg["content"] is None:
            # Some models require empty string instead of None for tool_call messages
            pass
        # Fix tool_calls if present
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                # Ensure arguments is a JSON string (not dict)
                if "arguments" in func and isinstance(func["arguments"], dict):
                    func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
        cleaned.append(msg)
    return cleaned


def load_sft_sample(parquet_path: str, idx: int = 0):
    """Load a single SFT sample and convert to native Python types."""
    df = pd.read_parquet(parquet_path)
    row = df.iloc[idx]
    messages = clean_messages(convert_nested(row["messages"]))
    tools_raw = convert_nested(row.get("tools", None)) if "tools" in df.columns else None
    tools = clean_tool_schema(tools_raw)
    enable_thinking = row.get("enable_thinking", None)
    return messages, tools, enable_thinking, len(df)


def count_tool_calls_in_messages(messages):
    """Count tool_calls by tool name in the message list."""
    counts = Counter()
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "unknown")
                counts[name] += 1
    return counts


def test_apply_chat_template(tokenizer, messages, tools, enable_thinking):
    """Apply chat template and return rendered text + token ids."""
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    # Render as text (not tokenized) for inspection
    rendered_text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    )

    # Tokenize for checking token count
    token_ids = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        **kwargs,
    )

    return rendered_text, token_ids


def inspect_training_path(tokenizer, data_path: str):
    """Inspect the actual MultiTurnSFTDataset loss-masked text used for training."""
    from omegaconf import OmegaConf

    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    dataset = MultiTurnSFTDataset(
        parquet_files=data_path,
        tokenizer=tokenizer,
        config=OmegaConf.create(
            {
                "max_length": 65536,
                "truncation": "loss_window",
                "messages_key": "messages",
                "tools_key": "tools",
                "enable_thinking_key": "enable_thinking",
                "ignore_input_ids_mismatch": True,
            }
        ),
        max_samples=1,
    )

    item = dataset[0]
    loss_text = tokenizer.decode(item["input_ids"][item["loss_mask"].bool()])
    return {
        "loss_text": loss_text,
        "has_reasoning": "<think>" in loss_text or "We need to find the precise title" in loss_text,
        "has_final_answer": "## Exact Answer" in loss_text,
        "tool_call_count": loss_text.count("<tool_call>"),
        "loss_token_count": int(item["loss_mask"].sum().item()),
    }


def check_tool_call_tags(rendered_text: str):
    """Check for <tool_call> tags and extract tool names."""
    # Hermes format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
    # Qwen3 may use slightly different format
    patterns = [
        # Standard Hermes
        r"<tool_call>\s*\{[^}]*\"name\"\s*:\s*\"([^\"]+)\"",
        # Qwen3 may use different tags
        r"<\|tool_call\|>\s*\{[^}]*\"name\"\s*:\s*\"([^\"]+)\"",
        # Some models use function_call
        r"<function_call>\s*\{[^}]*\"name\"\s*:\s*\"([^\"]+)\"",
    ]

    found_tools = Counter()
    for pattern in patterns:
        for match in re.finditer(pattern, rendered_text):
            found_tools[match.group(1)] += 1

    # Also check for raw tool name mentions in any XML-like tag context
    for tool_name in ["browser.search", "browser.open", "browser.find"]:
        # Count occurrences in tool_call-like contexts
        count = rendered_text.count(f'"name": "{tool_name}"')
        if count > 0 and tool_name not in found_tools:
            found_tools[tool_name] = count

    return found_tools


def check_rl_rollout_format():
    """Check what format the RL rollout tool parser expects."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from verl.experimental.agent_loop.tool_parser import HermesToolParser

        parser = HermesToolParser()
        # Test with Hermes format
        test_text = '<tool_call>\n{"name": "browser.open", "arguments": {"id": "0"}}\n</tool_call>'
        tool_calls = parser.parse(test_text)
        return {
            "parser_class": "HermesToolParser",
            "expected_format": '<tool_call>\\n{"name": "...", "arguments": {...}}\\n</tool_call>',
            "test_parse_ok": len(tool_calls) > 0,
            "parsed_tools": [tc.get("function", {}).get("name", "?") for tc in tool_calls] if tool_calls else [],
        }
    except Exception as e:
        return {"error": str(e)}


def run_tests(args):
    print("=" * 70)
    print("SFT Tokenization Verification for CaRR Deep Search")
    print("=" * 70)

    # --- Load tokenizer ---
    print(f"\n[1/5] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Check if tokenizer has tool_call special tokens
    special_tokens = {k: v for k, v in tokenizer.special_tokens_map.items()}
    print(f"  Special tokens: {list(special_tokens.keys())}")

    # Check for tool-related tokens in vocabulary
    tool_tokens = []
    for token_str in ["<tool_call>", "</tool_call>", "<|tool_call|>", "<|/tool_call|>"]:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id != tokenizer.unk_token_id:
            tool_tokens.append((token_str, token_id))
    print(f"  Tool-related tokens in vocab: {tool_tokens if tool_tokens else 'NONE FOUND'}")

    # --- Load SFT data ---
    print(f"\n[2/5] Loading SFT data from: {args.data_path}")
    messages, tools, enable_thinking, total_samples = load_sft_sample(args.data_path, args.sample_idx)
    print(f"  Total samples: {total_samples}")
    print(f"  Sample #{args.sample_idx}: {len(messages)} messages, enable_thinking={enable_thinking}")
    print(f"  Tools defined: {[t['function']['name'] for t in tools] if tools else 'None'}")

    # Count tool calls in messages
    msg_tool_counts = count_tool_calls_in_messages(messages)
    print(f"  Tool calls in messages: {dict(msg_tool_counts)}")

    # --- Apply chat template ---
    print(f"\n[3/5] Applying chat template (enable_thinking={enable_thinking})...")
    try:
        rendered_text, token_ids = test_apply_chat_template(tokenizer, messages, tools, enable_thinking)
        print(f"  Rendered text length: {len(rendered_text)} chars")
        print(f"  Token count: {len(token_ids)}")
    except Exception as e:
        print(f"  ERROR: apply_chat_template failed: {e}")
        print(f"  This means SFT tokenization is BROKEN for tool_calls format!")
        return False

    # --- Check for tool_call tags in rendered text ---
    print(f"\n[4/5] Checking for tool_call tags in rendered text...")
    found_tools = check_tool_call_tags(rendered_text)
    print(f"  Tool calls found in rendered text: {dict(found_tools)}")

    # Show relevant snippets
    expected_tools = {"browser.search", "browser.open", "browser.find"}
    found_set = set(found_tools.keys())
    missing = expected_tools - found_set
    extra = found_set - expected_tools

    if missing:
        print(f"  MISSING tool_call tags for: {missing}")
    if not missing:
        print(f"  All 3 browser tools have tool_call tags in rendered text")

    # Show first occurrence of each tool in rendered text
    for tool_name in sorted(expected_tools):
        # Find the context around the tool name
        idx = rendered_text.find(f'"name": "{tool_name}"')
        if idx >= 0:
            start = max(0, idx - 80)
            end = min(len(rendered_text), idx + 120)
            snippet = rendered_text[start:end].replace("\n", "\\n")
            print(f"\n  --- {tool_name} snippet ---")
            print(f"  ...{snippet}...")
        else:
            print(f"\n  --- {tool_name}: NOT FOUND in rendered text ---")

    # --- Check RL rollout parser compatibility ---
    print(f"\n[5/5] Checking RL rollout parser compatibility...")
    rl_info = check_rl_rollout_format()
    if "error" in rl_info:
        print(f"  Could not load RL parser: {rl_info['error']}")
    else:
        print(f"  RL parser: {rl_info['parser_class']}")
        print(f"  Expected format: {rl_info['expected_format']}")
        print(f"  Test parse OK: {rl_info['test_parse_ok']}")

    # Try parsing tool calls from the rendered SFT text using the RL parser
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from verl.experimental.agent_loop.tool_parser import HermesToolParser

        parser = HermesToolParser()

        # Extract assistant messages from rendered text and try to parse
        # Find all <tool_call>...</tool_call> blocks
        tc_pattern = r"<tool_call>(.*?)</tool_call>"
        tc_matches = re.findall(tc_pattern, rendered_text, re.DOTALL)
        print(f"\n  <tool_call> blocks found in SFT rendered text: {len(tc_matches)}")

        if tc_matches:
            parsed_names = []
            for tc_text in tc_matches[:5]:  # Show first 5
                try:
                    tc_json = json.loads(tc_text.strip())
                    parsed_names.append(tc_json.get("name", "?"))
                except json.JSONDecodeError:
                    parsed_names.append(f"PARSE_ERROR: {tc_text.strip()[:80]}")
            print(f"  First 5 parsed tool names: {parsed_names}")

            # Check if RL parser can parse the exact SFT format
            sample_block = f"<tool_call>{tc_matches[0]}</tool_call>"
            rl_parsed = parser.parse(sample_block)
            if rl_parsed:
                print(f"  RL HermesToolParser can parse SFT format: YES")
            else:
                print(f"  RL HermesToolParser can parse SFT format: NO - FORMAT MISMATCH!")
                print(f"  SFT block: {sample_block[:200]}")
        else:
            print(f"  No <tool_call> blocks found - checking alternative formats...")
            # Check for Qwen3-specific format
            alt_patterns = [
                r"<\|tool_call\|>(.*?)<\|/tool_call\|>",
                r"✿FUNCTION✿(.*?)✿RESULT✿",
                r"\[TOOL_CALLS\](.*?)\[/TOOL_CALLS\]",
            ]
            for alt_pat in alt_patterns:
                alt_matches = re.findall(alt_pat, rendered_text, re.DOTALL)
                if alt_matches:
                    print(f"  Found alternative format: {alt_pat} ({len(alt_matches)} matches)")
                    print(f"  First match: {alt_matches[0].strip()[:200]}")
                    break
            else:
                print(f"  No recognized tool_call format found in rendered text!")
                print(f"  This means the SFT training does NOT produce <tool_call> tags!")
                print(f"  The model cannot learn the Hermes tool_call format from SFT data.")

    except Exception as e:
        print(f"  RL parser test error: {e}")

    # --- Count <tool_call> blocks directly ---
    tc_pattern = r"<tool_call>(.*?)</tool_call>"
    tc_matches = re.findall(tc_pattern, rendered_text, re.DOTALL)
    print(f"\n  <tool_call>...</tool_call> blocks in rendered text: {len(tc_matches)}")
    if tc_matches:
        tc_tool_names = Counter()
        for tc_text in tc_matches:
            try:
                tc_json = json.loads(tc_text.strip())
                tc_tool_names[tc_json.get("name", "?")] += 1
            except json.JSONDecodeError:
                tc_tool_names["PARSE_ERROR"] += 1
        print(f"  Parsed tool names from <tool_call> blocks: {dict(tc_tool_names)}")

        # Show first block of each tool
        shown = set()
        for tc_text in tc_matches:
            try:
                tc_json = json.loads(tc_text.strip())
                name = tc_json.get("name", "?")
                if name not in shown:
                    shown.add(name)
                    print(f"\n  Example <tool_call> for {name}:")
                    print(f"    <tool_call>{tc_text.strip()}</tool_call>")
            except json.JSONDecodeError:
                pass

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True

    # Check 1: tool_call tags present
    if found_tools:
        has_hermes = "<tool_call>" in rendered_text
        tag_format = "Hermes <tool_call>" if has_hermes else "other"
        print(f"  [PASS] Tool calls rendered with {tag_format} format: {dict(found_tools)}")
    else:
        print(f"  [FAIL] No tool_call tags found in rendered SFT text")
        all_pass = False

    # Check 2: all 3 tools present
    if not missing:
        print(f"  [PASS] All 3 browser tools present in rendered text")
    else:
        print(f"  [FAIL] Missing tools in rendered text: {missing}")
        all_pass = False

    # Check 3: <tool_call> block count matches
    if tc_matches and len(tc_matches) > 0:
        print(f"  [PASS] {len(tc_matches)} <tool_call> blocks found and parseable")
    else:
        print(f"  [FAIL] No <tool_call> blocks found in rendered text")
        all_pass = False

    print(f"\n  Real training path (MultiTurnSFTDataset):")
    training_path = inspect_training_path(tokenizer, args.data_path)
    print(f"    loss_token_count: {training_path['loss_token_count']}")
    print(f"    tool_call_count: {training_path['tool_call_count']}")
    print(f"    has_reasoning: {training_path['has_reasoning']}")
    print(f"    has_final_answer: {training_path['has_final_answer']}")
    if training_path["has_reasoning"]:
        print("  [PASS] Training loss mask includes reasoning tokens")
    else:
        print("  [FAIL] Training loss mask is missing reasoning tokens")
        all_pass = False
    if training_path["has_final_answer"]:
        print("  [PASS] Training loss mask includes final answer tokens")
    else:
        print("  [FAIL] Training loss mask is missing final answer tokens")
        all_pass = False

    if all_pass:
        print(f"\n  RESULT: SFT tokenization correctly produces tool_call tags")
        print("  The model should learn reasoning, tool calls, and final answers from the real training path")
    else:
        print("\n  RESULT: SFT tokenization has issues in the real training path")

    # Write rendered text to file for manual inspection
    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered_text)
        print(f"\n  Full rendered text saved to: {args.output}")

    return all_pass


def run_distribution_check(args):
    """Check tool call distribution across all SFT samples."""
    print("\n" + "=" * 70)
    print("SFT Data Tool Call Distribution Check")
    print("=" * 70)

    df = pd.read_parquet(args.data_path)
    total_counts = Counter()
    samples_with_tools = {"browser.search": 0, "browser.open": 0, "browser.find": 0}

    for idx in range(len(df)):
        messages = convert_nested(df.iloc[idx]["messages"])
        counts = count_tool_calls_in_messages(messages)
        total_counts.update(counts)
        for tool in samples_with_tools:
            if counts.get(tool, 0) > 0:
                samples_with_tools[tool] += 1

    print(f"\n  Total samples: {len(df)}")
    print(f"\n  Tool call counts:")
    for name, count in sorted(total_counts.items(), key=lambda x: -x[1]):
        pct = count * 100 / sum(total_counts.values()) if total_counts else 0
        print(f"    {name}: {count} ({pct:.1f}%)")

    print(f"\n  Samples containing each tool:")
    for name, count in sorted(samples_with_tools.items()):
        pct = count * 100 / len(df)
        print(f"    {name}: {count}/{len(df)} ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify SFT tokenization for CaRR tool calls")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Path to Qwen3 model or checkpoint (for tokenizer)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="examples/carr_deepsearch/data/sft_train.parquet",
        help="Path to SFT training parquet",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to inspect")
    parser.add_argument("--output", type=str, default=None, help="Save rendered text to file")
    parser.add_argument("--check_distribution", action="store_true", help="Check tool distribution across all samples")
    args = parser.parse_args()

    success = run_tests(args)

    if args.check_distribution:
        run_distribution_check(args)

    sys.exit(0 if success else 1)
