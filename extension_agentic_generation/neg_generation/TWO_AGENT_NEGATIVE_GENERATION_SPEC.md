# Fresh Two-Agent Negative Generation for Accepted Positive Questions

## Purpose

Create a new negative-question set from every positive record in:

```text
new_negative/accepted_positive_questions.json
```

Use two technically isolated Codex agents:

- Agent A freshly authors proof-guided candidate mutations from positive-only
  inputs.
- Agent B independently determines truth, verifies the semantic diff and proof,
  checks novelty against relevant existing negatives, and makes the final
  decision.

Do not call the OpenAI API or any external GPT API. Do not require
`OPENAI_API_KEY`.

Do not modify:

```text
new_negative/accepted_positive_questions.json
```

The protected source SHA-256 is:

```text
18a9f309bdda960c7a2c3fae5932b8f14c807ff85df27567949664ca3ee53845
```

## Target coverage

The isolated Agent A packet contains exactly the ten positive source IDs:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

Agent A must produce exactly one candidate for every packet record. Silent
omission, duplicate source coverage, extra source IDs, and placeholders are
forbidden.

For each source, Agent A chooses the strongest proof-ready mutation among:

```text
assumption_perturbation
algorithm_perturbation
conclusion_perturbation
```

If the first attempted mutation cannot support a concrete falsification
strategy, Agent A must retry a different single component before writing the
candidate.

## Agent A technical isolation

Prompt-only isolation is insufficient. Agent A must run in a filesystem sandbox
whose readable benchmark workspace contains only:

```text
TWO_AGENT_NEGATIVE_GENERATION_SPEC.md
input/positive_sources.json
input/source_context.json
output/
```

Agent A may read only the target positive records, their positive proof context,
this specification, and ordinary mathematical reference knowledge.

Agent A must not read or inspect:

- the protected source file;
- any label-0 question;
- any previous mutation, candidate, proof strategy, audit, verifier result,
  report, PDF, LaTeX rendering, or generated-negative artifact;
- any Agent B input or output;
- the surrounding repository or sibling workspace contents.

Agent A independently authors every candidate during this run. It must not
assign final truth, verification, novelty, or acceptance status.

## Agent A work

For every input record Agent A must:

1. read the original positive question and cited positive proof context;
2. identify one privacy-critical proof dependency;
3. mutate exactly one assumption, algorithm step, or conclusion;
4. preserve all unrelated definitions, quantifiers, parameter domains,
   neighboring relations, algorithm steps, outputs, privacy notions, and
   conclusions;
5. state a concrete counterexample, privacy-loss calculation, divergence
   calculation, tight lower bound, or reduction that Agent B can check;
6. write one complete candidate.

The candidate question must be the original question with exactly one declared
`before` span replaced by exactly one declared `after` span.

Agent A writes:

```text
output/generator_candidates.json
```

as a JSON array. Every candidate has exactly:

```json
{
  "candidate_id": "newneg_source_1_algorithm_1",
  "source_question_id": 1,
  "mutation_type": "algorithm_perturbation",
  "question_tex": "...",
  "mutation": {
    "component": "algorithm",
    "before": "...",
    "after": "...",
    "targeted_dependency_id": "...",
    "reason_for_mutation": "..."
  },
  "source_proof_summary": "...",
  "expected_failure": "...",
  "proposed_falsification_strategy": "...",
  "preliminary_novelty_matches": []
}
```

Every candidate ID begins with `newneg_`.
`preliminary_novelty_matches` is exactly `[]`.

Agent A must not include:

```text
confidence
expected_label
claim_assessment
novelty_status
verification_status
final_decision
accept
accepted
```

Agent A also writes:

```text
output/generator_authorship.json
```

with exactly:

```json
{
  "spec_file": "TWO_AGENT_NEGATIVE_GENERATION_SPEC.md",
  "source_sha256": "...",
  "positive_sources_sha256": "...",
  "agent_a_input_manifest_sha256": "...",
  "candidate_count": 10,
  "covered_source_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  "candidate_question_sha256": {},
  "history_read": false,
  "freshly_authored_during_this_run": true
}
```

## Agent B independence

Agent B receives only a sanitized packet containing:

- the original positive question and cited positive proof context;
- the candidate question;
- the exact before/after mutation;
- the proposed falsification strategy;
- relevant existing negatives introduced only after Agent A exits;
- deterministic similarity scores;
- earlier candidates from this run only for sequential novelty comparison.

Agent B must not receive Agent A confidence, expected label, `expected_failure`,
acceptance signals, hidden reasoning, or historical verifier decisions.

For each candidate Agent B neutrally determines:

1. `TRUE`, `FALSE`, `AMBIGUOUS`, or `INSUFFICIENT_INFORMATION`;
2. whether exactly one semantic component changed;
3. whether the proof is valid and all remaining assumptions hold;
4. whether the candidate duplicates a visible existing negative or an earlier
   candidate in the current packet;
5. `ACCEPT`, `REJECT`, or `REQUIRES_HUMAN_REVIEW`.

Agent B writes `output/verifier_results.json`, in packet order, with exactly:

```json
{
  "candidate_id": "newneg_source_1_algorithm_1",
  "claim_assessment": "FALSE",
  "single_component_change": true,
  "novelty_status": "NOVEL",
  "verification_status": "VERIFIED",
  "proof_type": "explicit_counterexample",
  "full_proof": "...",
  "remaining_assumptions_checked": [],
  "closest_existing_negatives": [],
  "closest_generated_negatives": [],
  "uncertainties": [],
  "final_decision": "ACCEPT"
}
```

`closest_existing_negatives` contains visible namespaced negative IDs as
strings. `closest_generated_negatives` contains only earlier current-run
candidate IDs.

Allowed novelty values:

```text
NOVEL
NOVEL_BUT_RELATED
DUPLICATE
UNCERTAIN
```

## Verification and acceptance

A false claim requires a rigorous proof using at least one:

- explicit DP counterexample;
- exact privacy-loss or divergence calculation;
- applicable tight lower bound;
- formal reduction to a concrete privacy violation.

It is insufficient to say only that the original proof no longer applies, an
assumption changed, less noise is used, or the claim appears implausible.

A candidate is accepted only when Agent B returns:

```text
claim_assessment == FALSE
single_component_change == true
novelty_status in {NOVEL, NOVEL_BUT_RELATED}
verification_status == VERIFIED
final_decision == ACCEPT
uncertainties == []
```

Ambiguous, incomplete, delicate, multi-component, or disputed cases must be
routed to `generated_negative_human_review.json`.

## Required outputs

Write final artifacts under:

```text
new_negative/two_agent_run/
```

The run includes:

```text
agent_a_input_manifest.json
generator_authorship.json
generator_candidates.json
verifier_results.json
generated_negative_candidates.json
generated_negative_audit.json
generated_negative_progress.json
generated_negative_human_review.json
generation_report.md
```

The validator fails unless:

- Agent A input IDs equal the ten target IDs;
- Agent A generated exactly one candidate for every target ID;
- every candidate is exactly one declared replacement of its source question;
- Agent B coverage and order exactly equal Agent A coverage and order;
- every accepted item satisfies the full acceptance rule;
- every ambiguous or disputed item is routed to human review;
- Agent A could not see mutation history or existing negatives;
- the protected source SHA-256 is unchanged;
- repeated deterministic merges are byte-identical.
