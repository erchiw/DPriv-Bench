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


from utils.dpriv_data import (
    DEFAULT_HF_REPO_ID,
    load_category_2_answer_lookup_from_hf,
    load_category_2_answer_lookup_from_local,
)
from utils.eval_utils import (
    IF_Batch_Response,
    MODEL_TO_CHECKPOINT,
    get_predicted_labels_category_2,
    get_responses,
    get_responses_gpt_restricted_retrieval,
    get_responses_gemini_restricted_retrieval,
    load_json_or_jsonl,
)

def get_theorem_id(theorems: List[Dict[str, Any]], question_id: int) -> List[int]:
    """Return the theorem IDs linked to a given question.

    Args:
        theorems: Contents of ``question_theorem_link.json`` — a list of dicts
            with ``question_id`` and ``theorem_id`` (comma-separated string).
        question_id: The question whose linked theorems to retrieve.

    Returns:
        Ordered list of integer theorem IDs, or an empty list if the question
        has no entry in ``theorems``.
    """
    for item in theorems:
        if int(item.get("question_id", -1)) == int(question_id):
            raw = item.get("theorem_id")
            if raw is None:
                return []
            return [int(x) for x in str(raw).split(",") if str(x).strip()]
    return []


def prompt_construction(
    problem_description: str,
    theorem_description: str,
    qa_prompt_mode: str,
    task: str,
) -> str:
    """Construct the full prompt for a hard question, including theorem context.

    Combines the question text, the supporting theorem/definition block, and
    a chain-of-thought instruction that asks the model to cite theorems and
    put its final answer in ``\\boxed{}``.

    Args:
        problem_description: The raw LaTeX question text.
        theorem_description: Pre-formatted theorem/definition block (may be
            empty if the question has no linked theorems).
        qa_prompt_mode: Prompting strategy. Currently only ``"cot"`` is
            supported.
        task: Task identifier. Currently only ``"algo-judge-w-proof"`` is
            supported.

    Returns:
        The full prompt string to send to the model.

    Raises:
        NotImplementedError: If the ``task``/``qa_prompt_mode`` combination
            is not supported.
    """
    if task == "algo-judge-w-proof" and qa_prompt_mode == "cot":
        qa_inst = (
            "Please provide explanations or derivations first. "
            "If you use any theorem or definition, cite it by its ID in the reasoning. "
            "The very last line of your response must be exactly "
            "\\boxed{\\texttt{yes}} or \\boxed{\\texttt{no}}. "
            "Do not include citations or any other text on the last line."
        )

        helper = "You may use the following theorems or definitions if applicable."

        prompt = (
            problem_description
            + "\n\n"
            + helper
            + "\n\n"
            + theorem_description
            + "\n\n"
            + qa_inst
        )
        return prompt

    if task == "restricted-retrieval" and qa_prompt_mode == "cot":
        qa_inst = (
            "Please provide explanations or derivations first. "
            "You must use the theorem database before answering. "
            "First retrieve relevant sources from the theorem database. "
            "Then reason using the retrieved evidence. "
            "Finally return \\boxed{yes} or \\boxed{no}. "
            "Do not include citations or any other text on the last line. "
        )
        prompt = (
            problem_description
            + "\n\n"
            + qa_inst
        )
        return prompt

    if task == "zero-shot" and qa_prompt_mode == "cot":
        qa_inst = (
            "Please provide some explanations or deriviations first and then provide your final answer `yes` or `no` inside a Latex boxed format `\\boxed{}`."
        )
        prompt = (
            problem_description
            + "\n\n"
            + qa_inst
        )
        return prompt

    raise NotImplementedError(
        f"Task {task} with QA prompt mode {qa_prompt_mode} not implemented"
    )



def _load_theorem_is_definition(meta_path: Path) -> Dict[int, bool]:
    """Load meta_info_theorem.json and return theorem_id -> is_definition."""
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        int(item["theorem_id"]): bool(item.get("is_definition", False))
        for item in meta
    }


def load_question_ids(question_ids_path: Path) -> List[int]:
    """Load list of question IDs from a JSON file (e.g. difficult_question/test.json)."""
    with question_ids_path.open("r", encoding="utf-8") as f:
        ids = json.load(f)
    return [int(x) for x in ids]


def load_hard_question_data(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path, Path]:
    """Load difficult questions for IDs in test.json, attach theorems from question_theorem_link.json, and split into pending/all sets."""
    response_path = Path(
        f"response_hard_question/{args.task}_{args.model}_{args.qa_prompt_mode}_{args.seed}_responses.json"
    )
    predictions_path = Path(
        f"response_hard_question/{args.task}_{args.model}_{args.qa_prompt_mode}_{args.seed}_predictions.json"
    )

    question_ids = load_question_ids(Path(args.hard_question_ids_path))
    if args.data_source == "huggingface":
        answer_lookup = load_category_2_answer_lookup_from_hf(args.hf_repo, split=args.hf_split)
    else:
        answer_lookup = load_category_2_answer_lookup_from_local(args.question_root)

    with Path(args.theorem_link_path).open("r", encoding="utf-8") as f:
        theorem_link: List[Dict[str, Any]] = json.load(f)
    theorem_root = Path(args.theorem_root)
    theorem_is_definition = _load_theorem_is_definition(Path(args.meta_info_theorem_path))

    # Only run questions that have a theorem link (intersection of hard_question_ids and question_theorem_link)
    linked_question_ids = {int(item["question_id"]) for item in theorem_link if "question_id" in item}
    question_ids = [qid for qid in question_ids if qid in linked_question_ids]

    existing = load_json_or_jsonl(str(response_path)) if response_path.exists() else []
    answered_ids = {
        int(item["question_id"]) for item in existing if "question_id" in item
    }

    all_items: List[Dict[str, Any]] = []
    remaining_items: List[Dict[str, Any]] = []

    for qid in sorted(question_ids):
        answer = answer_lookup.get(qid)
        if answer is None:
            continue

        question = str(answer.get("question") or "")
        if not question.strip():
            continue

        theorem_ids = get_theorem_id(theorem_link, qid)
        theorem_statements = ""
        if theorem_ids:
            for th_id in theorem_ids:
                th_file_path = theorem_root / f"{th_id}.tex"
                with th_file_path.open("r", encoding="utf-8") as tf:
                    th_statement = tf.read()
                is_def = theorem_is_definition.get(th_id, False)
                if is_def:
                    theorem_statements += (
                        f"[DEFINITION {th_id}] " + th_statement + " [/DEFINITION]\n"
                    )
                else:
                    theorem_statements += (
                        f"[THEOREM {th_id}] " + th_statement + " [/THEOREM]\n"
                    )

        item = {
            "question_id": qid,
            "question": question,
            "theorem": theorem_statements,
            "label": answer.get("label"),
            "category": answer.get("subject"),
            "topic": answer.get("topic", "unknown"),
        }
        all_items.append(item)
        if qid not in answered_ids:
            remaining_items.append(item)

    return remaining_items, all_items, response_path, predictions_path


def limit_hard_question_to_first_n(
    max_questions: Optional[int],
    remaining_items: List[Dict[str, Any]],
    all_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep only the first ``N`` hard-track items in ``all_items`` order (smoke / partial runs).

    Pending items are intersected with that prefix. Order matches
    ``load_hard_question_data`` (sorted question IDs among linked hard IDs).

    Args:
        max_questions: If ``None``, return inputs unchanged. Otherwise must be >= 1.
        remaining_items: Questions still needing a model response.
        all_items: Full list built for this run.

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
    """Query the model on pending questions and append responses."""
    response_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_json_or_jsonl(str(response_path)) if response_path.exists() else []

    if not dataset:
        return existing

    prompt_list = [
        prompt_construction(
            item["question"], item["theorem"], args.qa_prompt_mode, args.task
        )
        for item in dataset
    ]
    new_entries: List[Dict[str, Any]] = []

    with tqdm(total=len(prompt_list), desc="Responses", unit="resp") as pbar:

        def _on_response(index: int, response_text: str) -> None:
            """Append a single API response and flush to disk."""
            row = {
                "question_id": dataset[index]["question_id"],
                "response": response_text,
            }
            new_entries.append(row)
            all_records = existing + new_entries
            with response_path.open("w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
            pbar.update(1)

        def _on_response_batch(responses: List[str]) -> None:
            """Append all vLLM responses at once and flush to disk."""
            question_ids = [item["question_id"] for item in dataset]
            if len(question_ids) != len(responses):
                raise ValueError(
                    "Length of question IDs and responses do not match "
                    f"({len(question_ids)} vs {len(responses)})"
                )

            for idx, response_text in enumerate(responses):
                row = {"question_id": question_ids[idx], "response": response_text}
                new_entries.append(row)

            all_records = existing + new_entries
            with response_path.open("w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
            pbar.update(len(new_entries))

        if args.task == "algo-judge-w-proof":
            if IF_Batch_Response[args.model]:
                get_responses(
                    prompt_list=prompt_list,
                    model=args.model,
                    on_response=_on_response_batch,
                    seed=args.seed,
                    download_dir=args.download_path,
                )
            else:
                get_responses(
                    prompt_list=prompt_list,
                    model=args.model,
                    on_response=_on_response,
                    seed=args.seed,
                    download_dir=args.download_path,
                )
        
        if args.task == "restricted-retrieval" and "gpt" in args.model:
            get_responses_gpt_restricted_retrieval(prompt_list=prompt_list, model=args.model, on_response=_on_response)
        elif args.task == "restricted-retrieval" and "gemini" in args.model:
            get_responses_gemini_restricted_retrieval(prompt_list=prompt_list, model=args.model, on_response=_on_response)
        elif args.task == "zero-shot" and "gpt" in args.model:
            if IF_Batch_Response[args.model]:
                get_responses(
                    prompt_list=prompt_list,
                    model=args.model,
                    on_response=_on_response_batch,
                    seed=args.seed,
                    download_dir=args.download_path,
                )
            else:
                get_responses(
                    prompt_list=prompt_list,
                    model=args.model,
                    on_response=_on_response,
                    seed=args.seed,
                    download_dir=args.download_path,
                )
    return existing + new_entries


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the hard question evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate hard (algorithm-level) DP questions with theorems."
    )
    parser.add_argument("--qa_prompt_mode", type=str, default="cot")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model alias to evaluate (see supported models in utils/eval_utils.py).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="algo-judge-w-proof",
        choices=["algo-judge-w-proof", "restricted-retrieval", "zero-shot"],
        help="DP task type.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hard_question_ids_path",
        type=str,
        default="difficult_question/hard_question_ids.json",
        help="Path to JSON array of question IDs to run (e.g. difficult_question/hard_question_ids.json).",
    )
    parser.add_argument(
        "--question_root",
        type=str,
        default="data/category_2",
        help="Local only: directory containing cate_2.json (question text, labels, subjects, topics).",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="local",
        choices=["local", "huggingface"],
        help="Use local cate_2.json or Hub cate_2 rows for question text and metadata.",
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
        "--theorem_link_path",
        type=str,
        default="difficult_question/question_theorem_link.json",
        help="Path to JSON linking questions to theorem IDs.",
    )
    parser.add_argument(
        "--theorem_root",
        type=str,
        default="difficult_question/theorem",
        help="Directory containing theorem/definition LaTeX files.",
    )
    parser.add_argument(
        "--meta_info_theorem_path",
        type=str,
        default="difficult_question/meta_info_theorem.json",
        help="Path to JSON with theorem_id and is_definition (used to label THEOREM vs DEFINITION).",
    )
    parser.add_argument(
        "--download_path",
        type=str,
        default="/data/user/.cache",
        help="Path to where open-source models are downloaded (for vLLM).",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        metavar="N",
        help="If set, only evaluate the first N questions in loaded order (smoke test).",
    )
    return parser


def main() -> None:
    """Entry point: parse arguments, run hard-question evaluation, and write predictions."""
    args = build_parser().parse_args()
    dataset, total_dataset, response_path, predictions_path = load_hard_question_data(
        args
    )
    dataset, total_dataset = limit_hard_question_to_first_n(
        args.max_questions, dataset, total_dataset
    )
    print(f"Model: {MODEL_TO_CHECKPOINT[args.model]}")
    print(f"Task: {args.task}_hard_question")
    print(f"Question IDs to answer: {[item['question_id'] for item in dataset]}")

    responses = generate_and_persist_responses(args, dataset, response_path)

    predictions = get_predicted_labels_category_2(responses, total_dataset)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
