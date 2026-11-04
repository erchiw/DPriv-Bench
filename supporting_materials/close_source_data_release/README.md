# Closed-source model responses and predictions

This folder stores closed-source model outputs used for rebuttal tables, plus a notebook that recomputes the reported metrics.

Paper results for **Table 2**, **Table 3**, **Figure 3**, **Table 5**, and **Figure 4** can be reproduced from [`reproduce_results.ipynb`](reproduce_results.ipynb).

Seeds used throughout: `17`, `42`, `89`, `231`, `764`.

## Layout

```text
responses_and_prediction_close_source/
  hard_files/
    one-shot/                 # one-shot hard-subset runs
    context-agumented/        # context-augmented hard-subset runs (folder spelling as on disk)
  judge_results/              # LLM-as-judge classifications (soundness / plausibility)
  cate_1_2_response_and_labels/
    cate-1/
      category_1/                       # raw responses
      category_1_extracted_labels/      # per-task evaluation JSONs (`avg_acc`)
    cate-2/
      category_2/                       # raw responses
      category_2_extracted_labels/      # prediction JSONs (`pred` / `label`)
  reproduce_results.ipynb     # end-to-end metric reproduction
  README.md
```

## What each subfolder is for

### `hard_files/one-shot`

Hard-subset one-shot runs. Prediction files look like:

```text
dp-judge_<task>_hard_<model>_one_shot_<seed>_predictions.json
```

Each prediction entry has `pred`, `label`, and related IDs. Used for paper **Table 5** (One-shot column), compared against Cate-1 Lap. RNM zero-shot accuracy.

### `hard_files/context-agumented`

Hard-subset context-augmented runs under three configurations:

| Config | Meaning |
|---|---|
| `zero-shot` | no extra context |
| `restricted-retrieval` | restricted retrieval context |
| `algo-judge-w-proof` | algorithm judge with proof context |

Filenames look like:

```text
<config>_<model>_cot_<seed>_predictions.json
```

Some files use `predicted_label` instead of `pred`; invalid predictions (`-1`) are treated as incorrect. The notebook reports **binary F1** (positive class = `1`) and plots a grouped bar chart of the three configs per model.

### `judge_results`

Judge classifications for reasoning quality. Primary files:

```text
dp-judge_<model>_cot_<seed>_judge_gpt-5.4_classification.json
```

Each item includes `extracted_label`, `ground_truth_label`, and `error_category`.

Used for paper **Figure 4** (`mean ± std` over seeds):

- **Accuracy**: `extracted_label == ground_truth_label`
- **Soundness**: fraction of answer-correct items with `error_category ∈ {correct, minor_gap}`
- **Plausibility**: fraction of answer-correct items with `error_category ∈ {correct, minor_gap, correct_direction_fixable}`

### `cate_1_2_response_and_labels`

Full Cate-1 / Cate-2 closed-source results.

**Cate-1** evaluations:

```text
dp-judge_<task>_hard_<model>_cot_<seed>_evaluations.json
```

Tasks: Laplace, Gauss (GDP), Gauss (zCDP), ExpMech, Lap. RNM, P-n-F.  
Notebook table: accuracy as `mean ± 1.96 × SE`, plus Task-Avg / Model-Avg.

**Cate-2** predictions:

```text
dp-judge_<model>_cot_<seed>_predictions.json
```

Notebook table: overall **F1 / Precision / Recall** as `mean ± 1.96 × SE` (positive class = `1`).

## Reproduce metrics

Paper results for the following figures/tables are produced by [`reproduce_results.ipynb`](reproduce_results.ipynb):

| Paper item | Notebook section |
|---|---|
| **Table 2** | Cate-1 accuracy (`mean ± 1.96 × SE`, Task-Avg / Model-Avg) |
| **Table 3** | Cate-2 F1 / Precision / Recall (`mean ± 1.96 × SE`) |
| **Figure 3** | Context-augmented hard-subset F1 grouped histogram |
| **Table 5** | One-shot in-context learning accuracy (Zero-Shot vs One-shot, `mean ± 1.96 × SE`) |
| **Figure 4** | Reasoning quality on Cate-2 (Accuracy / Plausibility / Soundness, `mean ± std`) |

1. Activate an environment with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and Jupyter (e.g. `conda activate dpllm`).
2. Open this folder and run `reproduce_results.ipynb` top to bottom.

The notebook sections are:

1. Hard files (one-shot accuracy; context-augmented F1 + histogram)
2. Judge accuracy / soundness / plausibility
3. Cate-1 and Cate-2 paper-style tables
4. Compact summary

## Models present here

Closed-source models in this package include:

- GPT-5-High / GPT-5-Minimal / GPT-5-Low
- Gemini-3-Pro (`gemini-3`)
- Gemini-2.5-Flash (`gemini-flash`)
- Claude-Sonnet-4.5 (`claude-sonnet`)
- Claude-Opus-4.5 (`claude-opus`)

Not every model appears in every sub-experiment (e.g. one-shot currently has Gemini-Flash and GPT-5-Minimal only).
