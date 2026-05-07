# data/


All files are pandas-style JSON records arrays (`pd.read_json` compatible).

---

## `category_1/`

Mechanism-level yes/no questions. **6 files × 98 questions = 588 total.**

| File | `--topic` argument | DP formalism |
|---|---|---|
| `cate_1_Laplace_pureDP.json` | `laplace_mechanism` | Pure DP (ε-DP) |
| `cate_1_Gaussian_GDP.json` | `gaussian_mechanism_GDP` | Gaussian DP (GDP) |
| `cate_1_Gaussian_zCDP.json` | `gaussian_mechanism_zCDP` | Zero-concentrated DP (zCDP) |
| `cate_1_ExpoMech_pureDP.json` | `selection_mechanism_expoMech_pureDP` | Exponential mechanism, pure DP |
| `cate_1_LaplaceRNM_pureDP.json` | `selection_mechanism_LaplaceRNM_pureDP` | Laplace report-noisy-max, pure DP |
| `cate_1_PF_pureDP.json` | `selection_mechanism_PF_pureDP` | Permute-and-flip, pure DP |

Each record has:

| Field | Type | Description |
|---|---|---|
| `question_id` | int | Unique question identifier |
| `question` | str | Natural-language/LaTeX question text |
| `label` | int | Ground-truth: `1` = yes (DP holds), `0` = no |
| `function_id` | int | ID of the query function used |
| `function` | str | LaTeX expression for the query function |
| `function_sens` | str | Sensitivity of the function |

### `cate_1_function_bank.json`

Pool of 49 query functions referenced by `function_id` across all Category 1 files. Each record has `function_id`, `function`, and `function_sens`. For Category 1, we assume input data in $[0,1]$ and adopt the replace-one neighboring relation when calculating `function_sens`.

---

## `category_2/`

### `cate_2.json`

Algorithm-level yes/no questions drawn from the DP research literature. **125 questions across 18 topics.**

Each record has:

| Field | Type | Description |
|---|---|---|
| `question_id` | int | Unique question identifier |
| `question_tex` | str | LaTeX question text (run scripts rename this to `question`) |
| `label` | int | Ground-truth: `1` = yes, `0` = no |
| `subject` | str | Broad subject category (run scripts rename to `category`) |
| `topic` | str | Fine-grained topic within the subject |
| `citation` | str | Source paper citation |
| `pdf_link` | str | Link to the source paper |
| `publish_year` | int | Year of the source paper |
| `negative_mode` | str | How the negative example was constructed (if applicable) |
| `related_question` | int/null | `question_id` of a related question, if any |
| `section_number` | str | Section in the source paper |
| `comments` | str | Proof sketch of correctness |
