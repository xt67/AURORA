# Completed v2.1 experimental comparison

Completed 20 November 2026 on the RTX 4050 Laptop GPU. Execution commit: `c915a7b`. Configuration: `configs/study_publication_v2_1.yaml`. The [execution design](PUBLICATION_V2.md) and [numerical correction](PUBLICATION_V2_1.md) are frozen separately from historical protocols.

## Execution and outcomes

All 36 classifiers completed 1000 updates each; all nine DCGANs completed 200 epochs. The grid generated 1,101 images, with B/D counts matched at 52/105/210 per run. Every classifier checkpoint and threshold was selected using validation before any test evaluation. All test images were real. Total recorded fit time was 2.3724 hours, excluding analysis, interpreter startup and the initial failed attempt.

The generated grids from all nine fits showed repetitive noise-like patterns rather than convincing radiographs. This is a synthesis failure, not a successful medical-image generator. Mean D recall was lower than A at 25% and 100% budgets. At 50%, recall increased from 0.800 to 0.874 but specificity fell from 0.829 to 0.713. Full seed/cohort metrics, calibration, bootstrap intervals and D–A/D–B contrasts are in the [public results JSON](../paper/publication_v2/paper_results.json).

## Independent artifact checks

- Recomputed all 36 thresholds from saved validation predictions and all test metrics from saved probabilities.
- Checked selected checkpoints against minimum validation-loss history and verified exact 1000-update schedules.
- Verified all 1,101 exported-image hashes and all nine final GAN checkpoints and 200-epoch finite loss arrays.
- Confirmed generator source IDs match the selected real TB budgets with zero holdout overlap.
- Verified all 800 original-image hashes and 11 protected metadata/history hashes remain unchanged.

Machine-readable reconciliation is in [artifact verification](../paper/publication_v2/artifact_verification.json). Full source-level evidence remains local under `outputs/publication_v2/run_20261119_v2_1/`; source images and weights are not published.

The final 1000-replicate uncertainty analysis uses `src.study.fast_analysis`, a postprocessing optimization preserving the original RNG call order and estimators. Eight equivalence tests passed, covering tied scores, grouped images and single-class edge cases; a 100-replicate check on actual predictions agreed with the original implementation within 1e-12 across all five metrics. Analysis source hashes are included in the public JSON. The slower redundant analysis was stopped after the equivalent calculation completed; no model fit or prediction was changed. All nine saved generator grids are available in the [supplementary contact sheet](../paper/publication_v2/images/all_generator_grids.png).

The seven-page PDF was rendered and visually inspected. The editable Word copy passed structural checks for text, tables, figures and references; its pagination could not be previewed locally because LibreOffice is unavailable. The demo also displayed the actual complete study report without an exception.

## Failure record and software tests

The initial run at `befadbc` failed on an AMP-scaled gradient before its first validation checkpoint. A training-only diagnostic reproduced one non-finite gradient tensor at scale 65536 and none at scale 1, with equal loss. The corrected grid used scale 1 uniformly and a fresh output directory. The initial failure remains preserved; it was not overwritten or hidden. No test result guided the correction.

The initial focused suite had 51 passes and one overly strict floating-point assertion, corrected to a numerical tolerance. Three focused follow-up checks passed, including historical gate regressions. After the AMP correction, all 19 publication tests passed, including actual CUDA numerical/update-count coverage on an artificial fixture. Four demo navigation/empty-state checks passed after the status update. These software tests are separate from the real experiment evidence.

## Interpretation and remaining work

The primary comparison is complete and includes negative findings. It does not establish convincing image synthesis, diagnostic validity, privacy or a universally superior augmentation method. The original similarity-screening implementation/thresholds remain incompletely documented, patients are not identified, the split is fixed and Montgomery holdouts are small. A repaired generator requires a new documented evaluation rather than repeated tuning against this exposed test set. Faculty review, data-use/ethics determinations and submission remain necessary; no approval or acceptance is invented.
