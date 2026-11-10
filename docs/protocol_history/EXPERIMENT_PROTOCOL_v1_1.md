# AURORA proposed executable protocol (v1.1)

Status: implemented on artificial fixtures, **not approved for publication training**. The researcher must review this protocol and exact configuration before the first run. Parameters below are explicit defaults, not evidence that they are optimal.

## Question and data

Does synthetic augmentation improve TB-manifestation-labelled chest radiograph classification under controlled scarcity, and how does fidelity relate to utility? Recall is primary; AUPRC, F1, specificity and AUROC are supporting outcomes. No minimum improvement is promised.

The study is an experimental comparison of selected methods under defined conditions. Completion means reporting measured outcomes and limitations, including null or negative findings; it does not require superiority, a universally best method or clinical validation. With the default five arms, three budgets and three seeds, the run grid contains 45 classifier fits and nine G1 generator fits. Runtime has not been benchmarked for this full grid.

Use the original NLM Shenzhen/Montgomery collection. Preserve source cohorts. The pre-adjudication candidate inventory has 800 images; final counts remain pending. Do not substitute the historical mixed-source Kaggle dataset. Do not claim patient-disjoint partitions.

Human review must resolve all candidate pairs and investigate cross-label cases. `src.study.data` validates attributed decisions, creates connected components from confirmed duplicates and exact hashes, and proposes a deterministic cohort/label-aware split. Confirmed groups spanning cohorts form their own cohort-composition stratum. Groups are shuffled using the split seed, then assigned to the partition with the largest remaining target count. Targets are 70/15/15; grouped allocation is approximate. Both labels must be present in each overall partition. Report realized counts and subgroup sizes before approval. Cross-label duplicates are rejected pending explicit adjudication.

Every original image has a SHA-256 hash. Canonical loading verifies paths, hashes and readability; it rejects cross-split groups or identical hashes. The manifest is immutable by convention and hash-bound approval: any change requires new approval. Dataset and decision records stay local.

## Preprocessing and scarcity

Images are converted through PIL to grayscale, resized to 128 × 128, then converted to tensors. GAN input is one channel in [-1,1]. Classifier input repeats grayscale to three channels and applies ImageNet mean/std normalization. This square resize may alter aspect ratio and remove fine detail. High-bit-depth images require explicit preprocessing decisions before use; do not silently treat them as 8-bit.

Use 25%, 50%, 100% of the frozen TB-positive training image count, rounded down with a minimum of one. Keep all normal training images. One seeded shuffled positive-image order defines nested prefixes across budgets. `budget_seed=42` is fixed across classifier seeds. This is image-count scarcity; duplicate components remain within a partition, but multiple members of a component are not independent patients. Report that fact if duplicate groups remain.

Validation/test data are real, fixed and never used to train a generator. Cohort identity is used only for splitting and reporting, never as a classifier feature.

## Comparisons and matched optimization

| Arm | Training input |
|---|---|
| A0 | Selected real images; random sampling with replacement |
| A1 | Same real images; inverse-frequency sampling with replacement for equal expected class proportions |
| A2 | Same sampling as A0, plus conventional transforms |
| A3 | Selected real images plus repeated positive training rows; exact added count matched to G1 |
| G1 | Selected real images plus generated positive images |

Conventional transforms are horizontal flip (p=.5), affine rotation ±5 degrees, translation up to 5%, and brightness/contrast jitter .1. Their clinical plausibility needs reviewer approval. No conventional transforms are silently added to G1.

All arms use ResNet-18, Adam, the same batch size, learning rate and weight decay, unweighted cross entropy, exactly the same number of optimizer updates, and the same validation cadence. There is no early stopping in the study runner; validation loss selects the saved checkpoint without changing compute exposure. This differs intentionally from the historical exploratory epoch trainer. Default configuration: 1000 updates, batch 16, validation every 100 updates, learning rate .0001 and weight decay .0001. Default seeds: 42,123,999. These values need researcher sign-off.

At the selected checkpoint, maximize validation F1 over observed validation probabilities and endpoints 0/1; break ties with the highest threshold. Apply this fixed threshold to test and cohort subsets. Recall remains the primary reported outcome; this is not a clinical operating-point claim. The threshold rule must be approved before any test evaluation.

G1 trains an unconditional generator only on the positive images in that exact budget. Generate `ceil(max(0, normal_count - positive_count) * synthetic_ratio)` images, default ratio 1. A zero deficit means zero synthetic images and must be reported. The configured final generator epoch is used; no test-based generator selection occurs. Default candidate: existing 128px DCGAN. WGAN-GP is selectable with its configuration after feasibility review. GAN generation is full precision; classifier AMP is enabled on CUDA. Batch 16 and sample grids of 16 limit memory use on the 6GB GPU. OOM stops a run and is recorded rather than silently changing the method.

G2/diffusion is not implemented or claimed as evaluated. It is optional, requires separate feasibility work and an approved protocol amendment. Cross-cohort sensitivity training is also optional and not included in this pooled v1 runner.

## Approval and execution

Version 1.1 adds A3 following the count-control rationale of Pink and Sykes (2026); see [literature basis](LITERATURE_VALIDATION.md). A3 draws positive training rows uniformly with replacement using the run seed and adds exactly G1's deficit-based count. It records duplicate source IDs and does not claim independent new observations. At zero deficit it adds nothing. Its finite class counts match G1 even when the ratio is not one; A1 instead balances sampling in expectation. Our ratio definition differs from Pink and Sykes. The protocol/configuration amendment requires fresh researcher approval.

Analyze G1 against A0 and A3, retaining all results. Use the analysis command below with `--baseline A3` and the corresponding A3/G1 result paths for the count-matched contrast. The analyzer rejects unequal added counts. No test-based ratio selection is permitted.

After human review and proposal inspection, the researcher creates a local approval file using `configs/approval.example.json`. Do not change the committed example to imply approval. Set identities, timezone-qualified timestamp, explicit approval flags and SHA-256 values of the exact manifest, config and protocol. PowerShell `Get-FileHash -Algorithm SHA256 <path>` produces a hash; store lowercase values.

```powershell
python -m src.study.runner --config configs/study.yaml --approval outputs/approved_run.json --output outputs/publication_v1 --check-only
python -m src.study.runner --config configs/study.yaml --approval outputs/approved_run.json --output outputs/publication_v1
```

The gate checks approval flags, all three hashes, review-source hashes, completed decisions, reconstructed split equality and original image integrity before training. Output directories must be new. Failed runs retain a failure artifact and are not replaced with estimates. Software cannot authenticate a human signature; approval is a researcher attestation, not an automated ethics review.

## Analysis and provenance

After the entire configured grid completes, run `python -m src.study.report --run outputs/publication_v1 --output outputs/report_v1`. The exporter checks archived hashes, approval-to-provenance consistency, run completeness, common configuration and manifest, then recomputes metrics and writes a Markdown table and demo-compatible JSON. All arms are retained, including negative outcomes. The report contains aggregate/per-seed/cohort statistics and paired contrasts without raw image IDs or local paths. Archive integrity checks cannot authenticate a researcher signature or independently validate experimental conduct. Inspect the exported report before release. Failed or missing runs must be resolved or reported through a separately documented protocol deviation; they are not silently dropped.

Each run records exact selected real IDs, config, manifest digest, commit, device, Python/torch versions, runtime, threshold, checkpoint hash, test probabilities and subgroup metrics. Generator provenance records only its selected positive training IDs/hashes. The run root archives frozen config, protocol, approval and manifest.

```powershell
python -m src.study.analysis --results outputs/publication_v1/A0_b0.25_s42/result.json outputs/publication_v1/A1_b0.25_s42/result.json --output outputs/analysis_example.json
python -m src.evaluate.generative --real data/final/train/Tuberculosis --synthetic outputs/publication_v1/G1_b0.25_s42/synthetic --output outputs/memorization_screen.json
```

Supply all completed seed/arm artifacts to the analysis command for final reporting. It recomputes metrics from probabilities, reports sample mean/std across seeds, paired seed differences, group-stratified bootstrap intervals, Brier score and 10-bin ECE. Bootstrap intervals describe image-group uncertainty conditional on a fitted model; few-seed intervals are unstable. Missing runs are not imputed and no automatic significance claim is made.

Nearest-neighbour and diversity diagnostics use 64px grayscale RMSE. These are preliminary screens, not privacy tests. Optional `--inception` computes FID and unbiased polynomial-kernel KID on pretrained Inception pool3 features; it may download weights. For fidelity, use a separately declared real validation directory, with exact count/reference documented. KID subset standard deviation is not a confidence interval. Tiny reference sets make FID unstable. Neither FID/KID nor visual inspection establishes diagnostic validity. Never tune on the test set.
