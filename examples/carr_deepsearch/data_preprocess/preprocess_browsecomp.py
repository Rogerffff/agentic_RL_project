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
Preprocess BrowseComp dataset (smolagents/browse_comp) to verl parquet format for evaluation.

The dataset only has a 'test' split with fields: problem, answer, problem_topic, canary.

Key distinction:
- prompt.content = problem + FORMAT_SUFFIX (full prompt for agent)
- search_forbidden_strs[0] = raw problem (without FORMAT_SUFFIX, used by CaRR reward server as question)

rubric_reward_ratio is set to 0.0 so reward server only computes outcome reward (LLM judge accuracy).

Input:  HuggingFace smolagents/browse_comp (test split)
Output: browsecomp_eval.parquet
"""

import argparse
import os

import datasets
import pandas as pd

FORMAT_SUFFIX = """

Your response should be in the following format:

## Explanation with Citations
{your explanation for your final answer with inline citations}

## Exact Answer
{your succinct, final answer}

## References
{listed cited sources in numerical order}"""


def main():
    parser = argparse.ArgumentParser(description="Preprocess BrowseComp dataset to verl parquet format")
    parser.add_argument("--output_dir", required=True, help="Output directory for parquet files")
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Local path to dataset. If not provided, downloads from HuggingFace.",
    )
    parser.add_argument("--subset_size", type=int, default=256, help="Subset size for fixed eval (0 to skip)")
    parser.add_argument("--subset_seed", type=int, default=42, help="Random seed for subset sampling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset (test split only)
    print("Loading BrowseComp dataset...")
    if args.dataset_path:
        dataset = datasets.load_dataset(args.dataset_path, split="test")
    else:
        dataset = datasets.load_dataset("smolagents/browse_comp", split="test")

    print(f"Loaded {len(dataset)} records")

    records = []
    for idx, example in enumerate(dataset):
        problem = example["problem"]
        answer = example["answer"]

        # Build tools_kwargs (same structure as RL data)
        tools_kwargs = {
            "browser.search": {
                "create_kwargs": {
                    "search_forbidden_strs": [problem],
                },
            },
            "browser.open": {
                "create_kwargs": {
                    "search_forbidden_strs": [problem],
                },
            },
        }

        records.append({
            "data_source": "browsecomp",
            "agent_name": "carr_tool_agent",
            "prompt": [{"role": "user", "content": problem + FORMAT_SUFFIX}],
            "ability": "deepsearch",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "rubrics": [],
                "search_forbidden_strs": [problem],
                "rubric_reward_ratio": 0.0,
                "need_tools_kwargs": True,
                "tools_kwargs": tools_kwargs,
                "split": "eval",
                "index": idx,
            },
        })

    # Save to parquet
    output_path = os.path.join(args.output_dir, "browsecomp_eval.parquet")
    pd.DataFrame(records).to_parquet(output_path)
    print(f"Saved: {output_path} ({len(records)} rows)")

    # Sanity check
    df = pd.read_parquet(output_path)
    print(f"\nSanity check:")
    print(f"  Columns: {list(df.columns)}")
    print(f"  data_source unique: {df['data_source'].unique()}")
    print(f"  agent_name unique: {df['agent_name'].unique()}")
    sample_extra = df.iloc[0]["extra_info"]
    print(f"  extra_info keys: {list(sample_extra.keys())}")
    print(f"  rubric_reward_ratio: {sample_extra.get('rubric_reward_ratio')}")
    print(f"  rubrics: {sample_extra.get('rubrics')}")
    print(f"  search_forbidden_strs non-empty: {bool(sample_extra.get('search_forbidden_strs'))}")
    print(f"  tools_kwargs keys: {list(sample_extra.get('tools_kwargs', {}).keys())}")

    # Verify prompt != search_forbidden_strs (prompt has FORMAT_SUFFIX, forbidden doesn't)
    sample_prompt = df.iloc[0]["prompt"][0]["content"]
    sample_forbidden = sample_extra["search_forbidden_strs"][0]
    assert sample_prompt.endswith(FORMAT_SUFFIX.strip()), "prompt should end with FORMAT_SUFFIX"
    assert not sample_forbidden.endswith(FORMAT_SUFFIX.strip()), "search_forbidden_strs should NOT have FORMAT_SUFFIX"
    print(f"  prompt vs forbidden: prompt is longer by {len(sample_prompt) - len(sample_forbidden)} chars (FORMAT_SUFFIX)")

    # Generate fixed subset for reproducible evaluation
    if args.subset_size > 0 and args.subset_size < len(df):
        subset_df = df.sample(n=args.subset_size, random_state=args.subset_seed)
        subset_filename = f"browsecomp_eval_subset_{args.subset_size}_seed{args.subset_seed}.parquet"
        subset_path = os.path.join(args.output_dir, subset_filename)
        subset_df.to_parquet(subset_path)
        print(f"\nSaved subset: {subset_path} ({len(subset_df)} rows)")


if __name__ == "__main__":
    main()
