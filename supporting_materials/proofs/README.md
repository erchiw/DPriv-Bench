# proofs/

Supporting Proofs for DPriv-Bench: tightness proofs for Category 1 query functions, expanded counterexample proofs for Category 2 negatives, and a rendered catalog of all Category 2 questions.

These documents are **not** required to run the evaluation scripts. They document how ground-truth labels and sensitivities were justified.

## Files

| File | Contents |
|---|---|
| [`cate-1-tightness.pdf`](cate-1-tightness.pdf) | Replace-one sensitivity proofs for all **49** functions in the Category 1 function bank |
| [`cate-2-counterexamples-part1.pdf`](cate-2-counterexamples-part1.pdf) | Corrected counterexample proofs for **14** Category 2 negative questions |
| [`cate-2-counterexamples-part2.pdf`](cate-2-counterexamples-part2.pdf) | Corrected counterexample proofs for **7** further Category 2 negative questions |
| [`cate_2_metadata.pdf`](cate_2_metadata.pdf) | Reader-friendly rendering of all **125** Category 2 questions and their metadata |

---

## `cate-1-tightness.pdf`

**Title:** *Replace-One Sensitivity Proofs for the Category 1 Function Bank*

Proves the global replace-one sensitivity of every query in [`data/category_1/cate_1_function_bank.json`](../../data/category_1/cate_1_function_bank.json).

- Datasets are ordered tuples \(X \in [0,1]^n\).
- Adjacent datasets differ in at most one coordinate.
- A claimed bound \(s\) is **tight** only if it holds for all neighbors *and* is attained by some adjacent pair (a tight witness).

The write-up also records cases where the raw function-bank label is not unconditionally tight (e.g. it requires \(n \ge 3\), a particular median/quantile convention, or a nondegenerate parameter).

---

## Category 2 counterexample proofs

**Title:** *DPriv-Bench Category 2: Corrected Counterexample Proofs*

For selected **No** questions, the short `comments` field in [`data/category_2/cate_2.json`](../../data/category_2/cate_2.json) is only a sketch (often an upper bound, a missing hypothesis, or a conversion that is not itself a counterexample). These PDFs replace those sketches with explicit witnesses: neighboring datasets, output events, and a numerical or closed-form privacy-loss violation.

Each item includes:

1. the original problem statement;
2. the original metadata comment;
3. a corrected proof;
4. a **No** verdict.

### Part 1 (14 questions)

Q7, Q20, Q21, Q22, Q42, Q43, Q85, Q86, Q88, Q109, Q110, Q112, Q114, Q116.

Topics include permute-and-flip / zCDP, the exponential mechanism, Sheffet OLS, mean estimation, objective perturbation, Confident-GNMax, Propose–Test–Release, and the single-quantile mechanism.

### Part 2 (7 questions)

Q77, Q81, Q84, Q104, Q105, Q106, Q136.

Topics include noisy mean-ratio estimators, parallel composition, and the sparse vector technique. Part 2 also records independent numerical checks for the ratio-estimator examples.

---

## `cate_2_metadata.pdf`

A typeset catalog of every Category 2 question, grouped by subject and topic. For each question it shows the LaTeX statement, label (Yes/No), subject, topic, negative mode, related question, citation, section, year, PDF link, and comments.

This is the same reader-friendly view linked from the [top-level README](../../README.md).
