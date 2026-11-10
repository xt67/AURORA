# AURORA / MediSyn completion status

The core toolkit is implemented and previously tested. The current task is the simplified v2 dataset/protocol preparation, now complete; the experimental study itself has not run.

- Verified 800 source images and preserved them all.
- Excluded 210 unique flagged participants from the primary subset without declaring them duplicates.
- Created separate retained/exclusion manifests for 590 / 210 images and preserved all 206 scored pairs.
- Proposed the deterministic 413/89/88 split and 52/105/210 positive training budgets.
- Specified four conditions, three seeds, matched classifier updates, one DCGAN and no required diffusion/cross-cohort experiment.
- Preserved v1.1 protocol/config history and documented automated selection bias and study limits.
- Passed 12 focused data-selection tests in 2.34s. Earlier full toolkit suite: 72 passed; later review UI focused suite: 23 passed. These are software checks, not medical validation.

The researcher must review the [selection report](DATASET_SELECTION_REPORT.md) and [simplified protocol](EXPERIMENT_PROTOCOL_SIMPLIFIED.md) before approving the first revised run. No new training approval is created. A full-800-image pair review is no longer a prerequisite for this selected-subset proposal.

After approval: adapt and test the existing execution gate for automated exclusion (not fictitious human adjudication), wire the TB-only conventional transforms and count-matched positive augmentation rule, perform the bounded feasibility check, then execute the agreed grid. Finally interpret actual results, complete the manuscript/declarations, choose license/release terms and configure/verify public hosting. No public deployment or paper submission is claimed.
