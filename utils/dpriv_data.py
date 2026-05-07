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

"""Load DPriv-Bench splits from the Hugging Face Hub or local pandas-format JSON files.

Hub loading requires the ``datasets`` package::

    pip install datasets

Local loading requires ``pandas``::

    pip install pandas
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Default matches the public dataset card (hyphenated repo id).
DEFAULT_HF_REPO_ID = ""

# Map ``run_category_1.py --topic`` values to Hub subset (config) names.
CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG: Dict[str, str] = {
    "selection_mechanism_expoMech_pureDP": "cate_1_ExpoMech_pureDP",
    "selection_mechanism_PF_pureDP": "cate_1_PF_pureDP",
    "selection_mechanism_LaplaceRNM_pureDP": "cate_1_LaplaceRNM_pureDP",
    "laplace_mechanism": "cate_1_Laplace_pureDP",
    "gaussian_mechanism_zCDP": "cate_1_Gaussian_zCDP",
    "gaussian_mechanism_GDP": "cate_1_Gaussian_GDP",
}


def _import_load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The `datasets` package is required for --data_source huggingface. "
            "Install with: pip install datasets"
        ) from exc
    return load_dataset


def _scalar_jsonable(value: Any) -> Any:
    """Convert common scalar/array types from Arrow/pandas/numpy to JSON-friendly Python."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    # numpy scalar
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    # pyarrow / HF scalar wrapper
    if hasattr(value, "as_py") and callable(getattr(value, "as_py")):
        try:
            return value.as_py()
        except Exception:
            pass
    return value


def _coerce_int(value: Any, field: str) -> int:
    v = _scalar_jsonable(value)
    if v is None:
        raise ValueError(f"Missing or null {field!r} in Hugging Face row.")
    return int(v)


def _coerce_optional_int(value: Any) -> Optional[int]:
    v = _scalar_jsonable(value)
    if v is None:
        return None
    return int(float(v))


def load_category_1_from_hf(
    repo_id: str,
    script_topic: str,
    *,
    split: str = "test",
) -> List[Dict[str, Any]]:
    """Load one Category-1 track from the Hub as the same list-of-dicts shape as local JSON.

    Each record includes at least ``question_id``, ``question``, ``label``, and
    ``function_id`` when present in the Hub schema.
    """
    load_dataset = _import_load_dataset()
    if script_topic not in CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG:
        raise KeyError(
            f"Unknown category-1 topic {script_topic!r} for Hugging Face loading. "
            f"Expected one of: {sorted(CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG)}"
        )
    config_name = CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG[script_topic]
    ds = load_dataset(repo_id, config_name, split=split)
    records: List[Dict[str, Any]] = []
    n = len(ds)
    for i in range(n):
        row = ds[i]
        qid = _coerce_int(row.get("question_id"), "question_id")
        question = str(_scalar_jsonable(row.get("question")) or "")
        label = _coerce_int(row.get("label"), "label")
        rec: Dict[str, Any] = {"question_id": qid, "question": question, "label": label}
        fid = _coerce_optional_int(row.get("function_id"))
        rec["function_id"] = fid if fid is not None else -1
        for opt in ("function", "function_sens"):
            if opt in row:
                val = _scalar_jsonable(row.get(opt))
                if val is not None:
                    rec[opt] = val
        records.append(rec)
    records.sort(key=lambda r: int(r["question_id"]))
    return records


def _str_or_unknown(value: Any) -> str:
    v = _scalar_jsonable(value)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "unknown"
    s = str(v).strip()
    return s if s else "unknown"


def load_category_2_from_hf(
    repo_id: str,
    *,
    split: str = "test",
) -> List[Dict[str, Any]]:
    """Load Category-2 examples from the ``cate_2`` config.

    Returns dicts aligned with :mod:`run_category_2` item shape:
    ``question_id``, ``question`` (LaTeX body), ``label`` (``1``/``0``),
    ``category`` (subject), ``topic``.
    """
    load_dataset = _import_load_dataset()
    ds = load_dataset(repo_id, "cate_2", split=split)
    items: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        row = ds[i]
        qid = _coerce_int(row.get("question_id"), "question_id")
        tex = str(_scalar_jsonable(row.get("question_tex")) or "")
        label = _coerce_int(row.get("label"), "label")
        subject = _str_or_unknown(row.get("subject"))
        topic = _str_or_unknown(row.get("topic"))
        items.append(
            {
                "question_id": qid,
                "question": tex,
                "label": label,
                "category": subject,
                "topic": topic,
            }
        )
    items.sort(key=lambda it: int(it["question_id"]))
    return items


def load_category_2_answer_lookup_from_hf(
    repo_id: str,
    *,
    split: str = "test",
) -> Dict[int, Dict[str, Any]]:
    """Return a ``question_id`` -> metadata map for Category 2 (Hub), including LaTeX text.

    Keys mirror local ``meta_info.json`` usage in ``run_hard_question``:
    ``question_id``, ``label`` (int), ``subject``, ``topic``, and ``question``
    (LaTeX statement, same as ``question_tex`` on the Hub).
    """
    rows = load_category_2_from_hf(repo_id, split=split)
    lookup: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        qid = int(row["question_id"])
        lookup[qid] = {
            "question_id": qid,
            "label": int(row["label"]),
            "subject": row["category"],
            "topic": row["topic"],
            "question": row["question"],
        }
    return lookup


# ---------------------------------------------------------------------------
# Local pandas-format JSON loaders
# ---------------------------------------------------------------------------

def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "The `pandas` package is required for local data loading. "
            "Install with: pip install pandas"
        ) from exc
    return pd


def load_category_1_from_local(
    data_dir: str,
    script_topic: str,
) -> List[Dict[str, Any]]:
    """Load one Category-1 track from ``data/category_1/cate_1_<config>.json``.

    The JSON file is a pandas-style records array (list of row objects).
    Returns the same list-of-dicts shape as :func:`load_category_1_from_hf`.

    Args:
        data_dir: Directory containing the ``cate_1_*.json`` files
            (typically ``data/category_1``).
        script_topic: One of the ``--topic`` values accepted by
            ``run_category_1.py`` (e.g. ``"laplace_mechanism"``).

    Returns:
        List of record dicts sorted by ``question_id``.
    """
    pd = _import_pandas()
    if script_topic not in CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG:
        raise KeyError(
            f"Unknown category-1 topic {script_topic!r} for local loading. "
            f"Expected one of: {sorted(CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG)}"
        )
    config_name = CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG[script_topic]
    path = Path(data_dir) / f"{config_name}.json"
    df = pd.read_json(path)
    records: List[Dict[str, Any]] = df.to_dict(orient="records")
    for r in records:
        r["question_id"] = int(r["question_id"])
        r["label"] = int(r["label"])
        fid = r.get("function_id")
        r["function_id"] = int(fid) if fid is not None else -1
    records.sort(key=lambda r: r["question_id"])
    return records


def load_category_2_from_local(
    data_dir: str,
) -> List[Dict[str, Any]]:
    """Load Category-2 examples from ``data/category_2/cate_2.json``.

    The JSON file is a pandas-style records array.  Column names follow the
    Hub schema (``question_tex``, ``subject``); they are renamed to match the
    ``run_category_2`` item shape (``question``, ``category``).

    Args:
        data_dir: Directory containing ``cate_2.json``
            (typically ``data/category_2``).

    Returns:
        List of ``{question_id, question, label, category, topic}`` dicts
        sorted by ``question_id``.
    """
    pd = _import_pandas()
    path = Path(data_dir) / "cate_2.json"
    df = pd.read_json(path)
    df = df.rename(columns={"question_tex": "question", "subject": "category"})
    df["question_id"] = df["question_id"].astype(int)
    df["label"] = df["label"].astype(int)
    items: List[Dict[str, Any]] = (
        df[["question_id", "question", "label", "category", "topic"]]
        .to_dict(orient="records")
    )
    items.sort(key=lambda it: it["question_id"])
    return items


def load_category_2_answer_lookup_from_local(
    data_dir: str,
) -> Dict[int, Dict[str, Any]]:
    """Return a ``question_id`` -> metadata map for Category 2 (local).

    Reads ``cate_2.json`` and returns the same dict shape used by
    ``run_hard_question``: keys ``question_id``, ``label``, ``subject``,
    ``topic``, and ``question`` (LaTeX text from ``question_tex``).

    Args:
        data_dir: Directory containing ``cate_2.json``
            (typically ``data/category_2``).
    """
    pd = _import_pandas()
    path = Path(data_dir) / "cate_2.json"
    df = pd.read_json(path)
    lookup: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        qid = int(row["question_id"])
        lookup[qid] = {
            "question_id": qid,
            "label": int(row["label"]),
            "subject": str(row.get("subject", "unknown") or "unknown"),
            "topic": str(row.get("topic", "unknown") or "unknown"),
            "question": str(row.get("question_tex", "") or ""),
        }
    return lookup
