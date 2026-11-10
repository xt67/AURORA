# Pre-feasibility verification — 11 November 2026

Status: passed read-only preparation checks; **no training launched or authorized**. The researcher accepts the 590-image dataset and split in principle, with final approval still required before the two-epoch feasibility run.

## Corrections

- Tracker wording now says **210 unique images**, not participants; patient identifiers remain unavailable.
- Primary v2 configuration sets `conventional_augmentation.horizontal_flip_probability: 0.0`. The protocol disables horizontal flipping. Rotation ±5°, translation up to 5%, brightness/contrast jitter .1 remain proposed, TB-training-only transforms.
- Original dataset, original audit, retained/excluded manifests, split JSON, selection reports and historical protocol/configuration were not modified.

## Verified data and allocation

Recomputed the union of flagged image identities, retained/excluded rows and every exclusion's pair membership from the original reference and 206-pair index. Rebuilt the split deterministically and compared every row/metadata field with the saved proposal. Checked the original reference/audit hashes, byte-identical audit snapshot and all 800 source-image hashes, dimensions, filename-label suffixes and readability. Verified the public report equals the local selection report and that every cohort/class/split count agrees.

| Partition | Normal | TB-labelled | Total |
|---|---:|---:|---:|
| Training | 203 | 210 | 413 |
| Validation | 43 | 46 | 89 |
| Test | 43 | 45 | 88 |
| Total | 289 | 301 | 590 |

| TB budget | Fixed real normal | Real TB | Added B | Added D |
|---|---:|---:|---:|---:|
| 25% | 203 | 52 | 52 | 52 |
| 50% | 203 | 105 | 105 | 105 |
| 100% | 203 | 210 | 210 | 210 |

Positive source IDs form nested subsets. Normal training IDs are identical at every budget. Both additions use the proposed selected-positive-count multiplier of 1.0, not the historical deficit-filling rule. All validation/test IDs map to original real images; none appears in any proposed generator-training source set. This confirms the prepared data separation, not an already executed v2 training pipeline. No actual model fitting occurred.

## Exact proposed configuration

Review [study_simplified_proposal.yaml](../configs/study_simplified_proposal.yaml) with the [corrected protocol](EXPERIMENT_PROTOCOL_SIMPLIFIED.md).

- Configuration SHA-256: `6a2b34cb819c805ba3e7523ed7ef65ef74dfb6b2c5ac9532155217e7354a950e`
- Protocol SHA-256: `ae72c4777851567eca3edb77070999a0d99a1df3eced0cf8e7a2521caf631de6`
- Proposed feasibility only: DCGAN/BCE, 52 real TB training images (25% budget), seed 42, **2 epochs**, batch 16, full precision, latent size 100, generator/discriminator widths 64, Adam learning rates .0002 and betas .5/.999, one discriminator step. No classifier fit or validation/test evaluation. The 200-epoch setting belongs to the later proposed publication grid, not feasibility.
- `training_authorized` remains `false`; no approval file or new frozen approval was created.

## Checks and remaining implementation blockers

`python -m pytest -q tests/test_study_selection.py tests/test_study_data.py`: **12 passed in 3.98s**. These tests do not fit models. The local verification record is `outputs/verification/pre_feasibility_2026-11-11.json`; it includes exact configuration/protocol digests and unchanged protected-artifact hashes.

Before feasibility: implement a v2 gate verifying the automated-exclusion source files, split and explicit new approval; add a bounded train-only feasibility entry point that overrides the publication epoch count to two, records its exact source IDs and never accesses holdouts. Test that path without real-data training. The old runner still requires v1 human decisions and must not be bypassed or supplied fictitious decisions.

Before the later publication grid: wire condition C's TB-only transforms with no horizontal flip; wire the identical selected-positive added-count rule for B/D and the A–D labels into the runner/report exporter; add integration checks for holdout exclusion and actual applied transforms/counts. The existing historical transform defaults and deficit count are not the v2 proposal and have not been silently changed.

Automated exclusion bias, absent patient identifiers, unadjudicated near-duplicates, small cohort holdouts and no clinical validation remain scientific limitations. This verification does not remove them. Stop for final researcher approval before any two-epoch feasibility run.
