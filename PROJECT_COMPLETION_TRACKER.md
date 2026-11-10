# AURORA current execution status

Updated 11 November 2026. Working branch: `aurora-refinement`.

The researcher's simplified proposal replaces full-dataset human adjudication for the primary study. The first revised training run is explicitly pending review; earlier authorization is superseded.

| Item | Verified status |
|---|---|
| Original data | 800 images preserved, all hashes/dimensions/readability checked; 406 normal / 394 TB-labelled |
| Automated exclusion | All 210 unique images in the 206 flagged pairs excluded from primary selection; no confirmed-duplicate claim |
| Retained subset | 590 images: 289 normal / 301 TB-labelled |
| Split proposal | Seed 42; cohort/class stratification; 413 train / 89 validation / 88 test |
| Scarcity proposal | 52 / 105 / 210 nested TB training images; 203 normal training images fixed |
| Comparison proposal | Four conditions, three budgets, three seeds: 36 classifier fits and nine DCGAN fits |
| Local artifacts | Separate retained/exclusion manifests, full score-preserving audit copy, split JSON and selection report |
| Verification | 12 focused selection/data tests passed in 2.34s; independent actual-manifest accounting checks passed |
| Training | Not started; no real-data feasibility run, new approval file or frozen approval created |
| History | Exact v1.1 protocol/config copies retained under docs/protocol_history/; original audit and review artifacts untouched |

Review the [verified report](docs/DATASET_SELECTION_REPORT.md) and [v2 proposal](docs/EXPERIMENT_PROTOCOL_SIMPLIFIED.md). The proposed configuration is `configs/study_simplified_proposal.yaml`. The historical runner remains unchanged; automated-exclusion approval checks, TB-only transforms and the proposed positive-count rule will be wired/tested only after approval, before feasibility and publication execution.

Selection bias, unadjudicated duplicates, absent patient identifiers, limited cohort sizes and no clinical validation must be disclosed. Montgomery validation/test each have 19 images (eight TB-labelled): no forced cross-cohort study. Diffusion is deferred for scope/hardware constraints. No images were deleted or relabelled, and no AI suggestion was converted into human approval.

Stop point: return counts, distribution, split, protocol and risks to the researcher. No further experiments are authorized in this task.

11 November review: the researcher accepted the candidate dataset/split in principle. Horizontal flipping is disabled in the primary v2 configuration (probability 0.0); mild rotation/translation and brightness/contrast remain proposed. Final approval before the two-epoch feasibility run is still required. No patient identifiers or patient-level independence are inferred.

Pre-feasibility verification: original and selected manifests, exclusion memberships and every split row agree with the report; nested budgets and matching B/D counts verified; all validation/test IDs are real originals and excluded from proposed generator-training IDs. All 800 originals passed hash/dimension/readability checks; protected metadata/history hashes remained unchanged. Focused selection/data tests: 12 passed in 3.98s. See [exact configuration, checks and implementation blockers](docs/PRE_FEASIBILITY_CHECK.md). No training launched.
