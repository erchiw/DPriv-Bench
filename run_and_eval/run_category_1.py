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

from utils.dpriv_data import DEFAULT_HF_REPO_ID, load_category_1_from_hf, load_category_1_from_local
from utils.eval_utils import (
    IF_Batch_Response,
    MODEL_TO_CHECKPOINT,
    SUPPORTED_MODELS,
    get_predicted_labels_category_1,
    get_responses,
    load_json_or_jsonl,
    prompt_construction_category_1,
)


CATEGORY_1_TOPICS = [
    "selection_mechanism_expoMech_pureDP",
    "selection_mechanism_PF_pureDP",
    "selection_mechanism_LaplaceRNM_pureDP",
    "laplace_mechanism",
    "gaussian_mechanism_zCDP",
    "gaussian_mechanism_GDP",
]
MODEL_CHOICES = sorted(SUPPORTED_MODELS)

# Local JSON and response filenames use this suffix (matches ``data/category_1/*_hard.json``).
CATEGORY_1_FILE_SUFFIX = "hard"


def _normalize_response_records(
    raw_responses: List[Any],
) -> List[Dict[str, Any]]:
    """Coerce raw response entries to canonical ``{question_id, response}`` dicts.

    Filters out any items that are not dicts or are missing ``question_id``,
    and casts ``question_id`` to ``int`` and ``response`` to ``str`` for
    consistent downstream comparisons.

    Args:
        raw_responses: Arbitrary list loaded from a responses JSON file.

    Returns:
        Filtered, normalised list of response records.
    """
    return [
        {"question_id": int(item["question_id"]), "response": str(item.get("response", ""))}
        for item in raw_responses
        if isinstance(item, dict) and "question_id" in item
    ]


def load_benchmark_data(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path, Path]:
    """Load the benchmark dataset and determine which questions still need answers.

    Reads the benchmark JSON for the given task/topic (local: ``*_hard.json``)
    and compares it against any existing responses on disk to compute the
    remaining work. Also creates the output directory if it does not yet exist.

    Args:
        args: Parsed command-line arguments. Uses ``task``, ``topic``,
            ``model``, ``qa_prompt_mode``, and ``seed``.

    Returns:
        A 4-tuple of:
        - ``remaining_dataset``: questions not yet answered,
        - ``full_dataset``: all questions (used for ordered output),
        - ``response_path``: path where responses will be saved,
        - ``predictions_path``: path where predictions will be saved.
    """
    if args.data_source == "huggingface":
        full_dataset = load_category_1_from_hf(args.hf_repo, args.topic, split=args.hf_split)
    else:
        full_dataset = load_category_1_from_local("data/category_1", args.topic)

    response_path = Path(
        f"response_category_1/{args.task}_{args.topic}_{CATEGORY_1_FILE_SUFFIX}_"
        f"{args.model}_{args.qa_prompt_mode}_{args.seed}_responses.json"
    )
    predictions_path = Path(
        f"response_category_1/{args.task}_{args.topic}_{CATEGORY_1_FILE_SUFFIX}_"
        f"{args.model}_{args.qa_prompt_mode}_{args.seed}_predictions.json"
    )
    response_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    raw_responses: List[Any] = load_json_or_jsonl(str(response_path)) if response_path.exists() else []
    response_records = _normalize_response_records(raw_responses)
    answered_ids = {int(item["question_id"]) for item in response_records}
    remaining_dataset = [item for item in full_dataset if int(item["question_id"]) not in answered_ids]
    return remaining_dataset, full_dataset, response_path, predictions_path


def limit_category_1_to_first_n(
    max_questions: Optional[int],
    remaining_dataset: List[Dict[str, Any]],
    full_dataset: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep only the first ``N`` questions in ``full_dataset`` order (smoke / partial runs).

    Pending questions are intersected with that prefix so generation and
    prediction files stay consistent.

    Args:
        max_questions: If ``None``, return inputs unchanged. Otherwise must be >= 1.
        remaining_dataset: Questions still needing a model response.
        full_dataset: Full benchmark list in canonical order.

    Returns:
        ``(limited_remaining, limited_full)``.

    Raises:
        ValueError: If ``max_questions`` is set but not positive.
    """
    if max_questions is None:
        return remaining_dataset, full_dataset
    if max_questions < 1:
        raise ValueError("--max_questions must be >= 1 when set")
    limited_full = full_dataset[:max_questions]
    allowed_qids = {int(item["question_id"]) for item in limited_full}
    limited_remaining = [item for item in remaining_dataset if int(item["question_id"]) in allowed_qids]
    return limited_remaining, limited_full


def generate_and_persist_responses(
    args: argparse.Namespace,
    dataset: List[Dict[str, Any]],
    full_dataset: List[Dict[str, Any]],
    response_path: Path,
) -> List[Dict[str, Any]]:
    """Query the model on pending questions and incrementally save responses to disk.

    After each response (or after a full batch for vLLM models), merges the
    new record into the on-disk file, preserving the original question order
    from ``full_dataset``. If ``dataset`` is empty, just re-saves any existing
    records in canonical order and returns them.

    Args:
        args: Parsed command-line arguments. Uses ``model``, ``qa_prompt_mode``,
            ``seed``, and ``download_path``. Prompts are built with
            ``prompt_construction_category_1``.
        dataset: Questions that still need a response.
        full_dataset: All questions, used to determine the output ordering.
        response_path: File path where responses are read from and written to.

    Returns:
        Complete list of all response records (existing + newly generated),
        ordered by the original question order in ``full_dataset``.
    """
    response_path.parent.mkdir(parents=True, exist_ok=True)
    raw_existing = load_json_or_jsonl(str(response_path)) if response_path.exists() else []
    existing_records = _normalize_response_records(raw_existing)
    records_by_qid = {int(item["question_id"]): item for item in existing_records}
    dataset_qid_order = [int(item["question_id"]) for item in full_dataset]

    def build_ordered_records() -> List[Dict[str, Any]]:
        """Return all accumulated records sorted by the original dataset order."""
        return [records_by_qid[qid] for qid in dataset_qid_order if qid in records_by_qid]

    def save_records(records: List[Dict[str, Any]]) -> None:
        """Atomically overwrite the response file with the current record list."""
        with open(response_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def persist_current_state() -> List[Dict[str, Any]]:
        """Build ordered records, save them, and return the result."""
        records = build_ordered_records()
        save_records(records)
        return records

    if not dataset:
        return persist_current_state()

    prompt_list = [
        prompt_construction_category_1(item["question"], args.qa_prompt_mode) for item in dataset
    ]

    question_ids = [item['question_id'] for item in dataset]

    with tqdm(total=len(prompt_list), desc="Responses", unit="resp") as pbar:

        def handle_response(prompt_index: int, response_text: str) -> None:
            """Save a single API response and flush to disk."""
            qid = int(dataset[prompt_index]["question_id"])
            records_by_qid[qid] = {"question_id": qid, "response": response_text}
            persist_current_state()
            pbar.update(1)

        def handle_response_batch(responses: List[str]) -> None:
            """Save all vLLM responses at once and flush to disk."""
            for idx, response_text in enumerate(responses):
                qid = question_ids[idx]
                records_by_qid[qid] = {"question_id": qid, "response": response_text}
            persist_current_state()
            pbar.update(len(question_ids))
        
        if IF_Batch_Response[args.model]:
            get_responses(prompt_list=prompt_list, model=args.model, seed=args.seed, download_dir=args.download_path, on_response=handle_response_batch)
        else:
            get_responses(prompt_list=prompt_list, model=args.model, seed=args.seed, download_dir=args.download_path, on_response=handle_response)
    return persist_current_state()


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the Category 1 evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate category-1 benchmark data.")
    parser.add_argument(
        "--qa_prompt_mode",
        type=str,
        default="cot",
        choices=["cot", "one_shot", "one_shot_neg"],
        help='Prompting: "cot" (zero-shot) or "one_shot" (one worked example before each question).',
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        choices=MODEL_CHOICES,
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="local",
        choices=["local", "huggingface"],
        help='Load questions from local JSON under data/category_1/ or from the Hub (--hf_repo).',
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
    parser.add_argument("--task", type=str, default="dp-judge", choices=["dp-judge"])
    parser.add_argument(
        "--topic",
        type=str,
        default="laplace_mechanism",
        choices=CATEGORY_1_TOPICS,
        help="Category-1 topic to evaluate.",
    )
    parser.add_argument(
        "--download_path",
        type=str,
        default="/data/user/.cache",
        help="Path to where open-source model is downloaded.",
    )
    parser.add_argument("--seed", type=int, default=0)
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
    if args.topic not in CATEGORY_1_TOPICS:
        raise ValueError(
            f"This script is for category-1 only. Use topic in {sorted(CATEGORY_1_TOPICS)}; but got '{args.topic}'."
        )

    dataset, total_dataset, response_path, predictions_path = load_benchmark_data(args)
    dataset, total_dataset = limit_category_1_to_first_n(args.max_questions, dataset, total_dataset)
    print(f"Model: {MODEL_TO_CHECKPOINT[args.model]}")
    print(f"Task: {args.task}_{args.topic}_{CATEGORY_1_FILE_SUFFIX}")
    print(f"Question IDs to answer: {[item['question_id'] for item in dataset]}")
    responses = generate_and_persist_responses(args, dataset, total_dataset, response_path)

    predictions = get_predicted_labels_category_1(responses, total_dataset)
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
