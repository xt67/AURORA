# MediSyn 0.2.0 — software release candidate

This release contains software and research documentation, not a completed scientific study or published paper.

## Implemented
- Repaired GAN/WGAN-GP training, metadata checkpoints, PNG exports, classifier batching and honest demo states.
- Isolated experiment logging, reduced GAN memory overhead, fixed seed handling and protected canonical split files.
- Human-review validation and an existing local review UI with reviewer attribution and change history.
- Hash-verified canonical proposals, duplicate-group/cohort/label-aware splits and nested positive training budgets.
- Explicit publication approval gate; matched-update A0/A1/A2/A3/G1 runner; validation checkpoint/threshold selection; cohort metrics and reproducibility artifacts.
- Metric recomputation, grouped bootstrap intervals, paired seed summaries, calibration and image-proximity/diversity diagnostics; optional Inception FID/KID.
- Research manuscript working draft with verified foundational/medical references and unfilled results.
- Deployment candidate, model/data documentation, software citation and bounded local server health check.

## Pending or deliberately excluded
- Human adjudication of 206 pairs, label-conflict resolution, canonical/split/protocol approval and first publication-run approval.
- Publication model training, measured research results, external scientific review and submission.
- Diffusion and cross-cohort sensitivity training are optional future protocol extensions, not implemented comparisons.
- Full FID/KID extraction on real data has not been run; mathematical/export components have regression coverage.
- Container runtime validation requires Docker. Public hosting requires an account and deployment action. No paid infrastructure was provisioned.
- A software license, authorship details and data/checkpoint redistribution decisions require owner/researcher decisions. No license is invented.

Existing uncommitted data/split artifacts are preserved locally and excluded from software commits. No raw medical images, private decision files, credentials or model checkpoints are released.

## Literature and public-repository update

Added three supplied research references and explicit validation limits; added A3 exact-count duplication and selectable paired-analysis baseline. Removed obsolete exploratory scripts, mock API wrappers, duplicate planning documents, unimplemented diffusion requirements, tracked logs/results and the exploratory split. Local originals are preserved in an ignored archive or their original local paths. No Git history was rewritten.
