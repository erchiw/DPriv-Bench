# difficult_question/

Supporting data for the **hard question** evaluation track: a subset of Category 2 questions augmented with relevant theorems and definitions supplied in the prompt.

## Files

| Path | Description |
|---|---|
| `hard_question_ids.json` | JSON array of the 18 question IDs in the hard subset |
| `question_theorem_link.json` | Maps each question ID to one or more theorem IDs (comma-separated string) |
| `theorem/` | LaTeX source for each theorem or definition (`<theorem_id>.tex`) |
| `meta_info_theorem.json` | Per-theorem metadata: `is_definition`, source `reference` (URL), and `location` within that source |

## Schema

**`question_theorem_link.json`** — each entry links one question to one or more supporting items:

```json
{"question_id": 2, "theorem_id": "1,2"}
```

**`meta_info_theorem.json`** — per-theorem metadata:

```json
{
  "theorem_id": 1,
  "is_definition": false,
  "reference": "https://arxiv.org/pdf/1802.08908",
  "location": "Theorem 6"
}
```

- `is_definition`: controls whether the item is labeled `[THEOREM ...]` or `[DEFINITION ...]` in the prompt

## How It Is Used

`run_and_eval/run_hard_question.py` loads question text and labels from `data/category_2/cate_2.json` (via `load_category_2_answer_lookup_from_local`), filters to the IDs listed in `hard_question_ids.json`, and for each question appends the linked theorem/definition text from `theorem/`. The combined prompt is sent to the model.

Run from the root directory:

```bash
python run_and_eval/run_hard_question.py \
  --model gpt-5-minimal \
  --task algo-judge-w-proof \
  --seed 0 \
  --hard_question_ids_path difficult_question/hard_question_ids.json \
  --question_root data/category_2 \
  --theorem_link_path difficult_question/question_theorem_link.json \
  --theorem_root difficult_question/theorem \
  --meta_info_theorem_path difficult_question/meta_info_theorem.json
```

`--question_root` points to the directory containing `cate_2.json` (default: `data/category_2`). Pass `--data_source huggingface` to fetch question text and labels from the Hub instead.

To evaluate a custom subset, edit `hard_question_ids.json` with the desired question IDs (each must have a corresponding entry in `question_theorem_link.json`).
