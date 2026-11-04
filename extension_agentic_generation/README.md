# Agentic Generation Experiments

> **Proof of concept only.** This directory is an experimental extension for automatically generating new DPriv-Bench questions. It is **not** part of our NeurIPS publication. Questions produced here have **not** been rigorously verified and should not be treated as official benchmark items.

This directory contains the protocols, source material, intermediate artifacts, and rendered outputs for a two-agent workflow that generates and verifies DPrivBench-style differential privacy questions.

The experiment has two stages:

1. **Positive generation** derives source-faithful questions whose correct answer is **Yes** and supplies checkable proofs.
2. **Negative generation** changes exactly one privacy-critical component of an accepted positive question, then verifies that the resulting claim is false and sufficiently novel.

In both stages, generation and verification are assigned to separate agents. Deterministic schemas, hashes, coverage checks, and acceptance gates are used to make the handoff auditable and to reduce self-verification bias.

## Directory layout

```text
agentic_generation/
├── README.md
├── pos_generation/
│   ├── .codex/agents/
│   │   └── dpriv-positive-generator.toml
│   ├── DPRIVBENCH_POSITIVE_TWO_AGENT_WORKFLOW.md
│   ├── dpriv_positive_workflow/schemas/
│   │   └── generator_output.schema.json
│   ├── references/                     # Four source PDFs retained locally
│   └── runs/positive-20260729-001/
│       ├── reference_manifest.json
│       ├── generator_output.json
│       └── sources/                    # Extracted text from five source papers
├── neg_generation/
│   ├── TWO_AGENT_NEGATIVE_GENERATION_SPEC.md
│   ├── generated_negative_candidates.json
│   ├── verifier_results.json
│   ├── generated_negative_audit.json
│   ├── generated_negative_human_review.json
│   └── generation_report.md
└── reports/
    ├── accepted_negative_examples.pdf
    ├── accepted_positive_questions.pdf
    ├── negative_example_generation.pdf
    └── positive_example_generation.pdf
```

## Positive-question workflow

The positive pipeline uses two mathematically focused roles:

- **Agent A (generator)** reads the locked reference packet, writes complete Yes-answer questions, and provides proofs with precise source anchors.
- **Agent B (verifier)** first solves each proof-hidden question independently, then audits the generator's original question, proof, assumptions, constants, parameter domains, adjacency relation, and privacy notion.

Only candidates that receive an independent Yes answer, pass every hard check, and receive an `ACCEPT` audit verdict are eligible for export. The workflow also locks source, question, proof, and independent-analysis hashes and limits claim-preserving revision loops.

The included run contains 10 positive candidates derived from five reference papers. Its rendered question set is available at `reports/accepted_positive_questions.pdf`; the detailed protocol and command sequence are in `pos_generation/DPRIVBENCH_POSITIVE_TWO_AGENT_WORKFLOW.md`.

## Negative-question workflow

The negative pipeline starts from accepted positive questions and enforces technical isolation between the two roles:

- **Agent A (generator)** sees positive-only inputs and freshly authors one assumption, algorithm, or conclusion perturbation per source. Each candidate must replace exactly one declared span and include a concrete falsification strategy.
- **Agent B (verifier)** independently determines whether the mutated claim is false, checks that only one semantic component changed, verifies a rigorous counterexample or calculation, and compares the candidate with existing and current-run negatives for novelty.

A negative candidate is accepted only if it is false, single-component, verified, novel or novel-but-related, free of unresolved uncertainty, and assigned the final decision `ACCEPT`.

The included negative run produced:

- 10 generator candidates;
- 9 accepted negatives;
- 1 rejected duplicate;
- 0 human-review cases.

See `neg_generation/generation_report.md` for the candidate-level decisions and `neg_generation/TWO_AGENT_NEGATIVE_GENERATION_SPEC.md` for the full isolation, schema, validation, and acceptance requirements.

## Key artifacts

| Artifact | Description |
|---|---|
| `reports/accepted_positive_questions.pdf` | Rendered accepted positive questions |
| `reports/positive_example_generation.pdf` | Positive-generation workflow report |
| `pos_generation/runs/positive-20260729-001/reference_manifest.json` | Hash-locked reference manifest |
| `pos_generation/runs/positive-20260729-001/generator_output.json` | Rich positive generator output |
| `neg_generation/generated_negative_candidates.json` | Accepted negative records |
| `neg_generation/verifier_results.json` | Independent truth, proof, and novelty decisions |
| `neg_generation/generated_negative_audit.json` | Consolidated audit trail |
| `neg_generation/generated_negative_human_review.json` | Candidates routed to human review (empty in this run) |
| `neg_generation/generation_report.md` | Candidate-level negative-generation results |
| `reports/accepted_negative_examples.pdf` | Rendered accepted negative examples |
| `reports/negative_example_generation.pdf` | Negative-generation workflow report |
