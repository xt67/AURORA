# Human review and canonical dataset preparation

A supervisor is not a software requirement. You may perform technical duplicate/source review as the sole researcher, using your own reviewer identifier and disclosing self-review. Published papers provide methods, not decisions for these images. Do not infer a diagnosis or relabel a genuine clinical conflict yourself without adequate evidence/expertise. Unresolved cases remain pending an explicitly documented resolution. See [literature and validation limits](LITERATURE_VALIDATION.md).

No decisions have been made or approved by this software. The existing local package contains 206 candidate pairs, 64 cross-label pairs and 13 high-similarity candidates. Similarity scores are evidence to inspect, not proof of identity.

Run the existing local interface from the repository root:

```powershell
python tools/provenance_review/review_server.py
```

Open http://127.0.0.1:8765 in your own browser. Keep this service local. It uses the existing reference images and contact sheets; it does not upload them. Stop with Ctrl+C when finished. A CSV/contact-sheet workflow remains equally valid.

Each save requires a reviewer identifier and records a UTC timestamp. Decisions are CONFIRMED_DUPLICATE, PROBABLE_DUPLICATE, SIMILAR_BUT_DISTINCT or UNRESOLVED. Label-conflict status is recorded separately. Save errors are displayed; prior decisions are retained in a local history JSONL file. All decision files and images remain gitignored.

The human must inspect cross-label cases and document their investigation in notes even when concluding that the images are distinct and there is no conflict. Probable, unresolved or conflicting cases block canonical preparation. Cross-label confirmed duplicates require explicit researcher adjudication; the tool refuses to silently relabel or exclude images. Record any approved reference corrections in a new versioned local reference manifest and retain the original.

When a complete human decision CSV exists, create a proposal (this command will refuse incomplete review):

```powershell
python -m src.study.data --reference data/reference/nlm_original/reference_manifest.csv --index outputs/provenance/manual_review_index.csv --decisions outputs/provenance/manual_review_decisions.csv --root data/reference/nlm_original --output outputs/canonical_proposal.json
```

The CLI checks review coverage, identities, reviewer attribution, timestamps, conflicts, source image hashes and image integrity. Confirmed duplicate relationships become connected components; exact matching hashes are also grouped. Groups stay together in the proposed cohort/label-aware 70/15/15 split. Group constraints can make realized counts differ from requested proportions. Review the counts and approve the proposal before freezing it. No patient-level independence is established.

The source CSV hashes are embedded in the proposal. The command uses exclusive creation so it cannot overwrite an existing proposal. Dataset approval and first publication run approval are separate steps documented in EXPERIMENT_PROTOCOL.md.
