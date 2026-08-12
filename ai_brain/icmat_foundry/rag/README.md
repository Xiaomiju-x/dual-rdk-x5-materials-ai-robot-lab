# X5-ICMat Foundry Namespace RAG Candidate

This package is an independent finals candidate. It does not modify or replace
the frozen `spectrum_knowledge_shared` implementation.

## Contracts

- The researcher must explicitly select one of four fixed namespaces:
  `phosphor_xrd_pl`, `electronic_materials_property`,
  `fab_process_metrology_yield`, or `opto_packaging_reliability`.
- BM25 indexes are built separately. Optional vector providers must return the
  selected namespace, and reciprocal-rank fusion rejects any cross-namespace
  candidate.
- The frozen 25,228-chunk phosphor corpus is opened through a read-only adapter.
  Its expected SHA-256 and count are verified before use and rechecked after a
  build. The candidate neither rewrites nor re-ingests that file.
- Seed indexes contain only license-reviewed source-catalog metadata. They do
  not contain downloaded paper bodies or unlicensed full text.
- A `SUPPORTED` answer requires every declared key claim to cite a retrieved
  hit. A missing citation, a citation outside the hit set, a cross-namespace
  hit, or an unmet evidence type produces `UNKNOWN`.
- `literature_knowledge`, `real_measurement`, `structured_dataset`, and
  `source_metadata` are distinct evidence classes. Literature cannot satisfy a
  claim that explicitly requires a real measurement.

## Offline Build

Run:

```powershell
.\.venv-icmat\Scripts\python.exe tools\build_icmat_rag_namespaces.py
```

The deterministic evidence is written to `evaluation/icmat_foundry/rag/`.
Only seed records are persisted. Legacy retrieval smoke results persist IDs,
source metadata, and hashes, but not legacy text excerpts.

## Boundary

This layer verifies declared claim-to-citation relationships and provenance
classes. Without a semantic verifier or human review, it cannot prove that an
arbitrary natural-language sentence is scientifically true or that a caller
declared every claim. Callers must decompose all key statements into the
versioned claim contract. The current seed namespaces are metadata-scale
candidates, not a production semiconductor knowledge base, and no local fab or
instrument measurement has been ingested by this module.
