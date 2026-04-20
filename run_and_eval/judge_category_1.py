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

"""Compute accuracy metrics from a Category 1 _predictions.json file.

Each record must contain 'pred' (1/0/-1) and 'label' (1/0).
Unparseable predictions (pred == -1) are treated as wrong.

You can either pass a pre-built predictions file, or recompute predictions from
a responses file plus ground truth (local JSON or Hugging Face), matching
``run_category_1.py --data_source``.

Usage:
    python run_and_eval/judge_category_1.py --predictions_path response_category_1/dp-judge_laplace_mechanism_hard_gpt-4o_cot_0_predictions.json

    python run_and_eval/judge_category_1.py --responses_path response_category_1/dp-judge_laplace_mechanism_hard_gpt-4o_cot_0_responses.json \\
        --data_source huggingface --topic laplace_mechanism
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from utils.dpriv_data import DEFAULT_HF_REPO_ID, load_category_1_from_hf
from utils.eval_utils import get_predicted_labels_category_1, load_json_or_jsonl

from run_category_1 import (
    CATEGORY_1_FILE_SUFFIX,
    CATEGORY_1_TOPICS,
    _normalize_response_records,
)


def compute_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    if total == 0:
        return {"n": 0, "unparseable": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    unparseable = sum(1 for p in predictions if p["pred"] == -1)
    y_true = [p["label"] for p in predictions]
    y_pred = [p["pred"] if p["pred"] != -1 else 1 - p["label"] for p in predictions]

    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
        precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    except ImportError:
        precision = recall = f1 = float("nan")

    acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / total
    return {"n": total, "unparseable": unparseable, "accuracy": acc,
            "precision": precision, "recall": recall, "f1": f1}


def _normalize_cat1_prediction_rows(raw: List[Any]) -> List[Dict[str, Any]]:
    """Keep only dict rows with int-coercible pred/label (Hub/JSONL-safe)."""
    out: List[Dict[str, Any]] = []
    for p in raw:
        if not isinstance(p, dict) or "pred" not in p or "label" not in p:
            continue
        try:
            pred_i = int(p["pred"])
            label_i = int(p["label"])
        except (TypeError, ValueError):
            continue
        rec: Dict[str, Any] = {"pred": pred_i, "label": label_i}
        if "question_id" in p:
            try:
                rec["question_id"] = int(p["question_id"])
            except (TypeError, ValueError):
                pass
        if "function_id" in p:
            try:
                rec["function_id"] = int(p["function_id"])
            except (TypeError, ValueError):
                pass
        out.append(rec)
    return out


def _load_cat1_ground_truth(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.data_source == "huggingface":
        return load_category_1_from_hf(args.hf_repo, args.topic, split=args.hf_split)
    data_path = Path(
        f"data/category_1/{args.task}_{args.topic}_{CATEGORY_1_FILE_SUFFIX}.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions_list(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.predictions_path is not None:
        raw = load_json_or_jsonl(args.predictions_path)
        return _normalize_cat1_prediction_rows(raw)
    raw_responses = load_json_or_jsonl(args.responses_path)
    responses = _normalize_response_records(raw_responses)
    dataset = _load_cat1_ground_truth(args)
    return get_predicted_labels_category_1(responses, dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge Category 1 predictions.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--predictions_path",
        default=None,
        help="Path to _predictions.json (or .jsonl).",
    )
    src.add_argument(
        "--responses_path",
        default=None,
        help="Path to _responses.json / .jsonl; ground truth from --data_source.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="local",
        choices=["local", "huggingface"],
        help="Ground-truth source when using --responses_path (ignored for --predictions_path).",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help="Hub repo id when --data_source huggingface.",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="test",
        help="Split name for datasets.load_dataset when --data_source huggingface.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dp-judge",
        choices=["dp-judge"],
        help="Filename prefix for local Category-1 JSON (dp-judge_<topic>_hard.json).",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        metavar="TOPIC",
        help="Required with --responses_path: Category-1 track (see run_category_1 topics).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.responses_path is not None:
        if not args.topic:
            parser.error("--topic is required when using --responses_path")
        if args.topic not in CATEGORY_1_TOPICS:
            parser.error(
                f"--topic must be one of {sorted(CATEGORY_1_TOPICS)}; got {args.topic!r}"
            )

    predictions = load_predictions_list(args)
    m = compute_metrics(predictions)
    print(f"Questions   : {m['n']}  (unparseable: {m['unparseable']})")
    print(f"Accuracy    : {m['accuracy']:.4f}")
    print(f"Precision   : {m['precision']:.4f}")
    print(f"Recall      : {m['recall']:.4f}")
    print(f"F1          : {m['f1']:.4f}")


if __name__ == "__main__":
    main()
