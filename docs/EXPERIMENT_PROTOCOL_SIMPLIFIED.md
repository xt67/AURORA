# AURORA simplified experimental proposal v2

Prepared 10 November 2026. **Dataset preparation only; first training run is not approved.** This proposal supersedes the full-800-image adjudication/five-arm study for the primary analysis. The old protocol and configuration remain historical. The researcher's latest instructions explicitly require reviewing these outputs before approving training, superseding earlier run authorization.

Corrected 11 November 2026 following researcher review. The 590-image dataset and proposed split are accepted in principle. Horizontal flipping is disabled in the primary condition C; the other mild transforms remain proposed for review. Final approval is still required before the two-epoch feasibility run.

## Question and scope

How do selected augmentation methods affect TB-manifestation-labelled chest radiograph classification under controlled scarcity? This is an experimental comparison under defined conditions, not a claim of a universally best method. Positive, null and negative results are retained. TB recall is primary; AUPRC, F1, specificity and AUROC are supporting outcomes. No minimum improvement or clinical validation is required to report experimental findings.

## Automated dataset selection

The original NLM reference contains 800 images: 406 normal and 394 TB-manifestation-labelled. The immutable 206-pair audit involves 210 unique images. Exclude every image participating in any flagged pair, irrespective of label, score or later AI recommendation. This leaves **590 images: 289 normal and 301 TB-labelled**. This is automated similarity-based exclusion, not human adjudication and not proof of duplication. None of the 800 originals, original reference rows or audit scores is deleted, relabelled or overwritten.

The local retained manifest preserves all original metadata. The exclusion manifest contains each excluded filename, source cohort, original label, every associated pair ID and the exclusion reason. A byte-identical copy of the complete candidate-pair index preserves scores. Source SHA-256 hashes bind the proposal to the original reference and audit. All original image hashes, dimensions and readability were checked. Original and retained cohorts have no repeated exact SHA-256 values in this verified inventory; this is not evidence of patient independence or absence of transformed duplicates.

| Cohort / source label | Original | Retained | Excluded | Excluded % | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shenzhen normal | 326 | 214 | 112 | 34.36% | 150 | 32 | 32 |
| Shenzhen TB-labelled | 336 | 249 | 87 | 25.89% | 174 | 38 | 37 |
| Montgomery normal | 80 | 75 | 5 | 6.25% | 53 | 11 | 11 |
| Montgomery TB-labelled | 58 | 52 | 6 | 10.34% | 36 | 8 | 8 |
| Total | 800 | 590 | 210 | 26.25% | 413 | 89 | 88 |

Marginal exclusions: Shenzhen 30.06%; Montgomery 7.97%; normal 28.82%; TB-labelled 23.60%. These differences make selection bias a material limitation: the subset may differ in anatomy, acquisition, population or disease presentation. Findings concern the screened subset, not automatically the original distribution.

## Fixed split and preprocessing

Within each cohort/label stratum, sort filenames, shuffle with one Python random generator seeded 42 while processing strata in sorted order, then allocate floor(70/15/15 percent) counts. Allocate leftover images by descending fractional remainder; ties go train, validation, test. Realized counts are **413/89/88 (70.00%/15.08%/14.92%)**. The retained exact hashes are unique; any unexpected repeated hash makes this preparation command refuse rather than split copies silently. The split is a proposal, not frozen approval. Image identity is not patient identity; patient identifiers are unavailable.

All conditions use grayscale 128×128 square resizing. Classifier input repeats grayscale into three channels with ImageNet normalization; generator input uses one channel in [-1,1]. High-bit-depth inputs require an explicit conversion and are rejected otherwise. Square resizing/low resolution may discard relevant detail. Validation and test transforms are deterministic. Both partitions stay real and fixed; no generator fits, classifier updates or hyperparameter choices use the test set.

## Four comparisons

| Condition | Classifier training input |
|---|---|
| A | Selected real images only; uniform replacement sampling |
| B | A plus repeated real TB training images, chosen uniformly with replacement using the run seed |
| C | A with conventional transforms applied only to TB training images; normal images receive deterministic preprocessing |
| D | A plus synthetic TB images from one existing DCGAN trained only on the selected TB training images |

Proposed added count for **both B and D is one image per selected real TB training image** (52, 105 or 210); no other real image is introduced. This controls nominal added count when comparing D with B. This is explicit TB oversampling, not an assertion that every budget is class balanced. It avoids a no-op at the full budget, where TB training images already exceed normal images. This differs from v1's deficit-filling count and requires review as part of this proposal. No synthetic/conventional transforms are silently combined. A and C have unchanged dataset counts. All conditions draw exactly the same number of classifier batches.

C's horizontal flipping is disabled (probability 0.0). Remaining candidate transforms are rotation ±5°, translation up to 5%, brightness/contrast jitter .1, applied only to TB training images. They remain proposed experimental choices for review, not clinically validated transformations. D uses the existing 128px BCE DCGAN as the simplest repaired/tested implementation; it is not claimed empirically superior to WGAN-GP. No architecture search is planned. WGAN-GP is outside the primary comparison; diffusion is deferred for scope/hardware constraints, without claiming measured infeasibility.

## Scarcity and matched training

One shuffled positive training-image order with budget seed 42 defines nested prefixes. Counts are floor(fraction × 210). All 203 normal training images remain fixed. Validation contains 43 normal/46 TB-labelled; test contains 43 normal/45 TB-labelled.

| TB budget | Real normal | Real TB | Added TB in B/D | Total A/C | Total B/D |
|---|---:|---:|---:|---:|---:|
| 25% | 203 | 52 | 52 | 255 | 307 |
| 50% | 203 | 105 | 105 | 308 | 413 |
| 100% | 203 | 210 | 210 | 413 | 623 |

Use three seeds, 42/123/999, giving **36 classifier fits and nine GAN fits**. Each classifier is the same ImageNet-pretrained ResNet-18 with Adam, learning rate .0001, weight decay .0001, batch size 16, unweighted cross entropy, 1000 optimizer updates and validation every 100 updates. No early stopping changes compute exposure. CUDA AMP is allowed for classifiers; GAN training remains full precision. Select the classifier checkpoint by minimum validation loss; select its threshold by maximum validation F1 with highest-threshold tie-break. Freeze that threshold for all test metrics. This is not a clinical operating-point claim.

Candidate DCGAN settings: latent dimension 100, generator/discriminator width 64, Adam learning rates .0002, betas .5/.999, batch 16, one discriminator step, 200 epochs, final fixed-epoch checkpoint. These are proposed existing defaults, not optimized findings. The first approved feasibility check will use the 25% training subset, seed 42 and two epochs, without test evaluation. Check runtime, GPU memory, finite losses, checkpoint load/export and gross collapse; this is not a publication result or proof of image quality. Stop and document infeasibility rather than silently changing the method. **No real-data feasibility training is authorized in this preparation task.**

## Evaluation, reporting and limitations

Report all per-seed/condition/budget outcomes, mean/sample standard deviation, supporting metrics, calibration and paired D-minus-A and D-minus-B contrasts. Bootstrap image groups conditionally on each fitted model; these are not patient-level intervals. Three-seed summaries have substantial uncertainty. No automatic significance or superiority claim is made. Report computational failures and negative results; do not select ratios or protocols using test results. Optional existing nearest-neighbour/diversity/FID/KID diagnostics remain secondary, with declared reference sets/counts; they do not validate diagnosis or privacy.

Montgomery validation and test each contain only **19 images, including eight TB-labelled images**. A single TB error changes subgroup recall by 12.5 percentage points. Cohort metrics are descriptive with sample sizes/uncertainty; an independent cross-cohort training experiment is not forced into the primary study.

Disclose automated exclusion, differential exclusion rates, no human duplicate adjudication, possible retained near-duplicates/repeated patients, missing patient identifiers, unvalidated source labels, no clinical validation, limited cohort sizes and low resolution. No supervisor, clinician or expert approval is invented. Refer to the supplied literature as methodological support, not AURORA results. The scope remains an independent student experimental study.

## Artifacts and next boundary

Run `python -m src.study.selection --reference data/reference/nlm_original/reference_manifest.csv --index outputs/provenance/manual_review_index.csv --root data/reference/nlm_original --output outputs/provenance` to prepare new outputs. Existing outputs cause refusal; use a new output directory to reproduce rather than overwrite.

Local artifacts: `aurora_primary_manifest_590.csv`, `aurora_exclusion_manifest_210.csv`, `aurora_flagged_pairs_206.csv`, `aurora_primary_split_proposal.json`, and `aurora_dataset_selection_report.md`. Dataset/audit files remain local. `configs/study_simplified_proposal.yaml` records the proposed preprocessing/method settings and explicitly disables training authorization.

The historical executable runner still enforces v1 human-adjudication semantics. **Do not pass this v2 proposal to that CLI or fabricate empty human decisions.** After researcher approval, adapt its gate to verify automated-exclusion provenance, wire the TB-only conventional transforms and selected-positive count rule, run focused tests and the feasibility check, then execute the approved grid. This preparation task deliberately stops before those execution changes and before any training.
