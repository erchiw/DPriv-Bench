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

"""Compute accuracy metrics from a Category 2 (or hard question) _predictions.json file.

Each record must contain 'pred' (1/0/-1), 'label' (1/0), and 'category'.
Unparseable predictions (pred == -1) are treated as wrong.

You can either pass a pre-built predictions file, or recompute predictions from
a responses file plus ground truth (local TeX+meta or Hugging Face), matching
``run_category_2.py --data_source``.

Usage:
    python run_and_eval/judge_category_2.py --predictions_path response_category_2/dp-judge_gpt-4o_cot_0_predictions.json

    python run_and_eval/judge_category_2.py --responses_path response_category_2/dp-judge_gpt-4o_cot_0_responses.json \\
        --data_source huggingface
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from utils.dpriv_data import DEFAULT_HF_REPO_ID, load_category_2_from_hf, load_category_2_from_local
from utils.eval_utils import (
    _ground_truth_label_to_int,
    get_predicted_labels_category_2,
    load_json_or_jsonl,
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


def _normalize_cat2_prediction_rows(raw: List[Any]) -> List[Dict[str, Any]]:
    """Coerce pred/label/category for JSON or Hub-exported rows."""
    out: List[Dict[str, Any]] = []
    for p in raw:
        if not isinstance(p, dict) or "pred" not in p or "label" not in p:
            continue
        try:
            pred_i = int(p["pred"])
            label_i = _ground_truth_label_to_int(p["label"])
        except (TypeError, ValueError):
            continue
        rec: Dict[str, Any] = {
            "pred": pred_i,
            "label": label_i,
            "category": str(p.get("category") or "unknown").strip() or "unknown",
        }
        if "question_id" in p:
            try:
                rec["question_id"] = int(p["question_id"])
            except (TypeError, ValueError):
                pass
        if "topic" in p:
            t = p.get("topic")
            rec["topic"] = str(t).strip() if t is not None else "unknown"
        out.append(rec)
    return out


def _normalize_response_records_cat2(raw_responses: List[Any]) -> List[Dict[str, Any]]:
    return [
        {"question_id": int(item["question_id"]), "response": str(item.get("response", ""))}
        for item in raw_responses
        if isinstance(item, dict) and "question_id" in item
    ]


def _load_cat2_full_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.data_source == "huggingface":
        return load_category_2_from_hf(args.hf_repo, split=args.hf_split)
    return load_category_2_from_local(args.question_root)


def load_predictions_list(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.predictions_path is not None:
        raw = load_json_or_jsonl(args.predictions_path)
        return _normalize_cat2_prediction_rows(raw)
    raw_responses = load_json_or_jsonl(args.responses_path)
    responses = _normalize_response_records_cat2(raw_responses)
    dataset = _load_cat2_full_dataset(args)
    return get_predicted_labels_category_2(responses, dataset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge Category 2 predictions.")
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
        "--question_root",
        type=str,
        default="data/category_2",
        help="Local only: directory containing cate_2.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions = load_predictions_list(args)
    m = compute_metrics(predictions)
    print("Overall")
    print(f"  Questions   : {m['n']}  (unparseable: {m['unparseable']})")
    print(f"  Accuracy    : {m['accuracy']:.4f}")
    print(f"  Precision   : {m['precision']:.4f}")
    print(f"  Recall      : {m['recall']:.4f}")
    print(f"  F1          : {m['f1']:.4f}")

    by_subject: Dict[str, List] = defaultdict(list)
    for p in predictions:
        by_subject[p.get("category", "unknown")].append(p)

    if len(by_subject) > 1:
        print(f"\n{'Subject':<45} {'N':>5}  {'Acc':>7}")
        print("-" * 60)
        for subject, preds in sorted(by_subject.items()):
            m = compute_metrics(preds)
            print(f"  {subject:<43} {m['n']:>5}  {m['accuracy']:>7.4f}")


if __name__ == "__main__":
    main()
