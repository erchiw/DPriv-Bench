# Copyright 2024 The DPriv-Bench Authors.
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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from utils.dpriv_data import DEFAULT_HF_REPO_ID, load_category_2_from_hf, load_category_2_from_local
from utils.eval_utils import (
    IF_Batch_Response,
    MODEL_TO_CHECKPOINT,
    SUPPORTED_MODELS,
    get_predicted_labels_category_2,
    get_responses,
    load_json_or_jsonl,
    prompt_construction,
)



def load_category_2_data(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path, Path]:
    """Load all Category 2 questions and identify which ones still need answers.

    For local data, reads ``cate_2.json`` from ``args.question_root`` via
    pandas.  For Hub data, fetches from the ``cate_2`` config.  Compares
    loaded questions against any existing responses on disk to determine the
    remaining work.

    Args:
        args: Parsed command-line arguments. Uses ``question_root``,
            ``data_source``, ``task``, ``model``, ``qa_prompt_mode``,
            and ``seed``.

    Returns:
        A 4-tuple of:
        - ``remaining_items``: questions not yet answered,
        - ``all_items``: all questions with metadata,
        - ``response_path``: path where responses will be saved,
        - ``predictions_path``: path where predictions will be saved.
    """
    response_path = Path(
        f"response_category_2/{args.task}_{args.model}_{args.qa_prompt_mode}_{args.seed}_responses.json"
    )
    predictions_path = Path(
        f"response_category_2/{args.task}_{args.model}_{args.qa_prompt_mode}_{args.seed}_predictions.json"
    )

    existing = load_json_or_jsonl(str(response_path)) if response_path.exists() else []
    answered_ids = {int(item["question_id"]) for item in existing if "question_id" in item}

    all_items: List[Dict[str, Any]] = []
    remaining_items: List[Dict[str, Any]] = []

    if args.data_source == "huggingface":
        all_items = load_category_2_from_hf(args.hf_repo, split=args.hf_split)
    else:
        all_items = load_category_2_from_local(args.question_root)
    remaining_items = [it for it in all_items if int(it["question_id"]) not in answered_ids]

    return remaining_items, all_items, response_path, predictions_path


def limit_category_2_to_first_n(
    max_questions: Optional[int],
    remaining_items: List[Dict[str, Any]],
    all_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep only the first ``N`` items in ``all_items`` order (smoke / partial runs).

    Pending items are intersected with that prefix.

    Args:
        max_questions: If ``None``, return inputs unchanged. Otherwise must be >= 1.
        remaining_items: Questions still needing a model response.
        all_items: Full benchmark list in canonical order.

    Returns:
        ``(limited_remaining, limited_all)``.

    Raises:
        ValueError: If ``max_questions`` is set but not positive.
    """
    if max_questions is None:
        return remaining_items, all_items
    if max_questions < 1:
        raise ValueError("--max_questions must be >= 1 when set")
    limited_all = all_items[:max_questions]
    allowed_qids = {int(item["question_id"]) for item in limited_all}
    limited_remaining = [
        it for it in remaining_items if int(it["question_id"]) in allowed_qids
    ]
    return limited_remaining, limited_all


def generate_and_persist_responses(
    args: argparse.Namespace,
    dataset: List[Dict[str, Any]],
    response_path: Path,
) -> List[Dict[str, Any]]:
    """Query the model on pending questions and append responses to disk.

    For API models, each response is appended and flushed after it arrives.
    For vLLM models, all responses are flushed in a single write after the
    full batch completes.

    Args:
        args: Parsed command-line arguments. Uses ``model``, ``qa_prompt_mode``,
            ``seed``, and ``download_path``.
        dataset: Questions that still need a response.
        response_path: File path where responses are appended (existing records
            are preserved).

    Returns:
        Combined list of pre-existing responses and newly generated ones.
    """
    response_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_json_or_jsonl(str(response_path)) if response_path.exists() else []

    if not dataset:
        return existing

    prompt_list = [prompt_construction(item["question"], args.qa_prompt_mode) for item in dataset]
    new_entries: List[Dict[str, Any]] = []

    with tqdm(total=len(prompt_list), desc="Responses", unit="resp") as pbar:

        def _on_response(index: int, response_text: str) -> None:
            """Append a single API response and flush to disk."""
            row = {"question_id": dataset[index]["question_id"], "response": response_text}
            new_entries.append(row)
            all_records = existing + new_entries
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
            pbar.update(1)
        def _on_response_batch(responses: List[str]) -> None:
            """Append all vLLM responses at once and flush to disk."""
            question_ids = [item['question_id'] for item in dataset]
            if len(question_ids) != len(responses):
                raise ValueError("Length of question IDs and responses do not match")
            
            for idx, response_text in enumerate(responses):
                row = {"question_id": question_ids[idx], "response": response_text}
                new_entries.append(row)
                all_records = existing + new_entries
            with open(response_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
            pbar.update(len(new_entries))
        
        if IF_Batch_Response[args.model]:
            get_responses(prompt_list=prompt_list, model=args.model, on_response=_on_response_batch, seed=args.seed, download_dir=args.download_path)
        else:
            get_responses(prompt_list=prompt_list, model=args.model, on_response=_on_response, seed=args.seed, download_dir=args.download_path)

    return existing + new_entries


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the Category 2 evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate category-2 benchmark data.")
    parser.add_argument("--qa_prompt_mode", type=str, default="cot")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--task", type=str, default="dp-judge", choices=["dp-judge"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--question_root",
        type=str,
        default="data/category_2",
        help="Local only: directory containing cate_2.json.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="local",
        choices=["local", "huggingface"],
        help="Load Category-2 items from local cate_2.json or from the Hub (--hf_repo, cate_2).",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help="Hugging Face dataset repo id when --data_source huggingface.",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="test",
        help="Split name passed to datasets.load_dataset when --data_source huggingface.",
    )
    parser.add_argument(
        "--download_path",
        type=str,
        default="/data/user/.cache",
        help="Path to where open-source model is downloaded.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        metavar="N",
        help="If set, only evaluate the first N questions in dataset order (smoke test).",
    )
    return parser


def main() -> None:
    """Entry point: parse arguments, run evaluation, and write predictions to disk."""
    args = build_parser().parse_args()
    dataset, total_dataset, response_path, predictions_path = load_category_2_data(args)
    dataset, total_dataset = limit_category_2_to_first_n(
        args.max_questions, dataset, total_dataset
    )
    print(f"Model: {MODEL_TO_CHECKPOINT[args.model]}")
    print(f"Task: {args.task}_Category_2")
    print(f"Question IDs to answer: {[item['question_id'] for item in dataset]}")

    responses = generate_and_persist_responses(args, dataset, response_path)

    predictions = get_predicted_labels_category_2(responses, total_dataset)
    
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
