# ICMat-Qwen SFT v4 Contract

SFT v4 replaces the held rule-shaped v3 dataset with source-document-disjoint,
evidence-grounded engineering tasks over the licensed RAG v2 full-text export.
This contract does not call a teacher and does not train a model.

## Stages

1. `prepare`: validates the complete RAG v2 manifest/export, admits only
   `licensed_fulltext_readonly` + `CC BY 4.0` + `literature_knowledge`, assigns
   whole `source_id` document families to train/validation/calibration,
   `audit_challenge`, or final test, and optionally writes
   `teacher_requests.jsonl`.
   The 14-paper corpus is split by whole source document. Every finals namespace
   must retain at least one train family and one different non-training family.
   The expansion screening receipt is hash-bound: all four accepted electronic
   materials papers must be present, while every quarantined PMCID/XML hash is
   rejected even if it is accidentally copied into a candidate directory.
2. `validate`: consumes an external teacher runner's candidates. Every answer
   sentence must bind exact `chunk_id`, `locator`, chunk hash, span offsets, and
   span hash. Schema, binding, numeric grounding, and refusal policy are checked.
   This stage writes validated candidates, not a training dataset.
3. `materialize`: writes train/validation/calibration JSONL only when an
   external independent audit receipt has caller-supplied SHA-256, exact subject
   hashes, GO decision, no blockers/revocation, and explicit materialization and
   QLoRA-pilot scopes.

## Teacher request

`icmat_teacher_request.v4` contains:

- `request_id`, `split`, `task`, `family_id`, and deterministic build seed;
- model-visible `system` and `user` strings;
- `source_chunks` with source/chunk/namespace/locator/content hash/license;
- exact `evidence_spans` with offsets and span hash;
- a request-bound Draft 2020-12 `response_schema`;
- deterministic `generation_config`.

The rich v4 records are written to `teacher_request_bindings.v4.jsonl`.
They are directly consumable by `icmat_foundry/llm/local_teacher_v4.py`, whose
candidate envelope uses the same canonical request-object hash.
`teacher_requests.jsonl` is a lossless execution projection for the existing
`icmat_foundry/llm/local_teacher.py` contract: schema
`icmat_teacher_request.v1`, two `messages`, licensed span-level `evidence`, the
same response schema, and deterministic `generation`. The builder calls that
module's real validator before writing the manifest. Audit-challenge bindings
remain outside the training-side local-teacher execution file.

## Task set

- `evidence_grounded_explanation`
- `evidence_bounded_comparison`
- `computed_experimental_boundary`
- `next_measurement_or_tool`
- `refusal_counterfactual`

The audit challenge uses dedicated document families and matched supported /
unsupported assertions over the same evidence shape. It is disjoint from
calibration and is never training-eligible. Final-test artifacts contain only
family and chunk membership hashes, never text, prompts, answers, or metrics.

## Current boundary

The existing QLoRA trainer does not accept `manifest.v4.json`; that is an
intentional fail-closed boundary. A separate reviewed change may add v4 support
only after a real external audit GO. No v4 dataset, training run, BPU binary, X5
execution, or production integration is claimed by this contract.
