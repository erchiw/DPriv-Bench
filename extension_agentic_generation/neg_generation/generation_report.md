# Accepted-Positive Questions: Two-Agent Negative Generation

## Outcome

- Target positive sources: **10**
- Generator candidates: **10**
- Accepted negatives: **9**
- Rejected candidates: **1**
- Human-review cases: **0**
- Protected source SHA-256: `18a9f309bdda960c7a2c3fae5932b8f14c807ff85df27567949664ca3ee53845`

Agent A was technically isolated from existing negatives and all mutation history. Agent B received no confidence, expected label, expected_failure, or acceptance signal.

## Candidate decisions

| Candidate | Source | Mutation | Claim | Novelty | Verification | Decision | Disposition |
|---|---:|---|---|---|---|---|---|
| newneg_source_1_assumption_1 | 1 | assumption_perturbation | FALSE | NOVEL_BUT_RELATED | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_2_conclusion_1 | 2 | conclusion_perturbation | FALSE | NOVEL_BUT_RELATED | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_3_algorithm_1 | 3 | algorithm_perturbation | FALSE | NOVEL | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_4_algorithm_1 | 4 | algorithm_perturbation | FALSE | NOVEL_BUT_RELATED | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_5_assumption_1 | 5 | assumption_perturbation | FALSE | NOVEL | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_6_conclusion_1 | 6 | conclusion_perturbation | FALSE | NOVEL | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_7_conclusion_1 | 7 | conclusion_perturbation | FALSE | NOVEL_BUT_RELATED | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_8_assumption_1 | 8 | assumption_perturbation | FALSE | NOVEL | VERIFIED | ACCEPT | ACCEPTED |
| newneg_source_9_algorithm_1 | 9 | algorithm_perturbation | FALSE | DUPLICATE | VERIFIED | REJECT | REJECTED |
| newneg_source_10_algorithm_1 | 10 | algorithm_perturbation | FALSE | NOVEL_BUT_RELATED | VERIFIED | ACCEPT | ACCEPTED |

Accepted examples remain candidates for final human review before any benchmark merge.
