# utils/

## `eval_utils.py`

Central utility module used by all evaluation scripts in `run_and_eval/`. Handles model dispatch, prompt construction, and yes/no answer parsing.

### Model Configuration

| Name | Description |
|---|---|
| `SUPPORTED_MODELS` | Set of valid model alias strings |
| `MODEL_TO_CHECKPOINT` | Dict mapping alias → full checkpoint/model-name string |
| `IF_Batch_Response` | Dict mapping alias → bool; `True` for vLLM models (all responses at once), `False` for API models (one at a time) |

### Public API

| Function | Description |
|---|---|
| `get_responses` | Main dispatcher; routes to the correct backend based on model alias |
| `load_json_or_jsonl` | Reads a JSON array or JSONL file; returns a list |
| `prompt_construction` | Wraps the question with a CoT instruction and `\boxed{}` answer format (Category 2 / hard track) |
| `prompt_construction_category_1` | Same as above, plus `one_shot` and `one_shot_neg` modes for Category 1 |
| `parse_yes_no_from_response` | Extracts yes/no from model output; returns `1`, `0`, or `-1` (unparseable) |
| `get_predicted_labels_category_1` | Converts raw response records to prediction dicts for Category 1 |
| `get_predicted_labels_category_2` | Converts raw response records to prediction dicts for Category 2 |

### Callbacks

The `on_response` parameter in `get_responses` and the per-backend functions is a callback invoked as responses arrive:

- **API models** (non-batch): `on_response(index: int, response_text: str)` — called once per prompt, used for incremental saving
- **vLLM models** (batch): `on_response(responses: List[str])` — called once with all responses after the full batch completes

### Yes/No Parsing Pipeline

`parse_yes_no_from_response` uses a three-step pipeline:

1. Search for `\boxed{yes}` or `\boxed{no}` in the response
2. Fall back to scanning the full response text for "yes" or "no"
3. If still unresolved: re-prompt GPT-4o as a judge (up to 3 recursive levels)
4. Return `-1` if still unparseable after all fallbacks

### Adding a New Model

**Registered model** (add to `SUPPORTED_MODELS`):

1. Add the alias string to `SUPPORTED_MODELS`
2. Add `alias → checkpoint` to `MODEL_TO_CHECKPOINT`
3. Add `alias → False` (API) or `True` (batch/vLLM) to `IF_Batch_Response`
4. Implement a `get_responses_<provider>(prompt_list, model, on_response)` function
5. Add a routing branch in `get_responses()`

**Custom model (no code changes needed)**:

- *Custom API* — set `CUSTOM_BASE_URL` (and optionally `CUSTOM_API_KEY`) in the environment, then pass any model name string to `--model`. Routed automatically to `get_responses_custom_api()`.
- *Custom local (vLLM)* — pass `--download_path /path/to/cache` with any HuggingFace checkpoint ID as `--model`. Routed automatically to `get_responses_custom_local()`.

### Notes

- All optional backends (`anthropic`, `google-genai`, `vllm`, `torch`) are imported inside `try/except` blocks and raise a clear error only when the corresponding model is actually used.
- DeepSeek models require both `DEEPSEEK_BASE_URL` and `DEEPSEEK_API_KEY` to be set in the environment.

---

### Constants

| Name | Description |
|---|---|
| `DEFAULT_HF_REPO_ID` | Default Hub repo (`"erchiw/DPriv-Bench"`) |
| `CATEGORY_1_SCRIPT_TOPIC_TO_HF_CONFIG` | Maps `--topic` argument values to Hub config names and local file stems |

### Public API — Hugging Face (requires `datasets`)

| Function | Description |
|---|---|
| `load_category_1_from_hf` | Load one Category 1 topic from the Hub; returns list of `{question_id, question, label, function_id}` dicts |
| `load_category_2_from_hf` | Load Category 2 from the `cate_2` Hub config; returns list of `{question_id, question, label, category, topic}` dicts |
| `load_category_2_answer_lookup_from_hf` | Same as above but returns a `question_id → metadata` dict (used by `run_hard_question.py`) |

### Public API — Local files (requires `pandas`)

All local loaders read pandas-format JSON records arrays from `data/`. They return the same dict shapes as their HF counterparts so callers need no branching logic beyond the `data_source` flag.

| Function | Description |
|---|---|
| `load_category_1_from_local` | Read `data/category_1/cate_1_<config>.json` via `pd.read_json`; returns list of `{question_id, question, label, function_id, …}` dicts sorted by `question_id` |
| `load_category_2_from_local` | Read `data/category_2/cate_2.json` via `pd.read_json`; renames `question_tex → question` and `subject → category`; returns list of `{question_id, question, label, category, topic}` dicts |
| `load_category_2_answer_lookup_from_local` | Same source as above but returns a `question_id → {question_id, label, subject, topic, question}` dict (used by `run_hard_question.py`) |
