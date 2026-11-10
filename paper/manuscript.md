# AURORA: Evaluating synthetic chest radiograph augmentation under controlled scarcity of TB-manifestation-labelled training images

**External-review working draft, 10 November 2026. Not ready for submission.**

Authors, affiliations, supervisor, corresponding author and target venue: to be supplied by the researchers. There are no completed publication experiments under this protocol.

## Abstract

**Background:** Synthetic images may expand training datasets, but fidelity and downstream value are separate questions. **Objective:** AURORA will evaluate whether augmentation improves TB recall when real positive training images are limited. **Methods:** MediSyn implements manifest-driven cohort/duplicate-aware partitioning, nested training budgets, matched classifier-update comparisons and validation-based threshold selection. **Results:** Pending human dataset adjudication and approved experiments. **Conclusion:** No conclusion about augmentation effectiveness is currently supported.

## 1. Introduction

A reproducible study of synthetic augmentation must distinguish an increase in training-image count from an increase in useful information. GANs learn a distribution through adversarial optimization [1]; DCGAN provides a convolutional image-generation baseline [2]. These mechanisms motivate augmentation experiments, but they do not establish that generated radiographs preserve disease-related features.

The NLM chest radiograph collections were released to support pulmonary-disease research, including images with manifestations of TB [3]. AURORA studies classification of these image labels rather than microbiologically confirmed infection. The proposed experiment reduces positive training-image availability while keeping evaluation images fixed. Its primary question is whether synthetic augmentation improves TB recall relative to real-only training and simpler augmentation controls, and how that utility relates to synthesis diagnostics.

The intended contribution is a controlled, auditable evaluation pipeline and its eventual measured findings. We do not claim a novel generator architecture, clinical readiness or publication novelty before a fuller task-specific literature review and completed experiments.

## 2. Related Work

Frid-Adar and colleagues evaluated GAN-based augmentation for liver-lesion classification using CT regions of interest [4]. Their study supports investigating synthetic augmentation as a task-dependent intervention; its findings cannot be transferred directly to full chest radiographs or TB-manifestation labels. Schütte and colleagues evaluated GAN-generated chest radiographs and brain CT images through downstream predictive performance, reporting sensitivity to label structure and limited sample availability [5]. This motivates treating utility, dataset composition and training-data budget as explicit experimental factors.

WGAN-GP replaces weight clipping with a critic-gradient penalty and provides an alternative adversarial training formulation [6]. MediSyn retains it as an available candidate, without assuming that all implemented architectures belong in the final study. FID compares distributions in a pretrained feature space [7], while KID uses a kernel-based discrepancy estimator [8]. These metrics address distributional similarity, not clinical correctness. The study therefore separates feature-based fidelity, pixel-based proximity/diversity screens and real-test classification outcomes. A deeper comparison with TB-specific synthesis and classification studies remains necessary before submission.

## 3. Materials and Methods

### 3.1 Candidate data and review status

The candidate inventory comprises Shenzhen (326 normal, 336 TB-manifestation-labelled) and Montgomery (80 normal, 58 TB-manifestation-labelled) images, consistent with the dataset publication [3]. The approximately balanced pooled inventory is not the earlier mixed-source 5:1 dataset. Final sample counts and exclusions depend on adjudication. No patient identifiers have been established in the project, and patient-level independence is not claimed.

Local provenance work identified 206 perceptually similar candidate pairs, including 64 cross-label pairs and 13 high-similarity candidates. These are unresolved audit observations, not confirmed duplicates. No qualified reviewer has signed them off. MediSyn records attributed human decisions, timestamps, notes and separate label-conflict status. Probable/unresolved decisions and cross-label duplicate conflicts block canonical finalization. The software does not make diagnosis, exclusion or relabelling decisions.

### 3.2 Implemented manifest and split procedure

A reviewed proposal records source-file hashes, relative image paths, cohort, binary label, connected duplicate group and partition. Exact hashes and confirmed duplicate relationships are grouped; cross-label groups are rejected. Groups are stratified by label and cohort composition, shuffled with a fixed seed, and allocated to the partition with the largest remaining target count under a 70/15/15 target. This group-aware procedure can produce approximate proportions. Realized counts and subgroup sizes must be approved and reported.

The training gate reconstructs the proposal from reviewed sources, verifies image hashes and checks that approval binds the exact manifest, configuration and protocol. The default training CLI has no bypass for an unresolved publication dataset. Software tests use separate artificial images.

### 3.3 Preprocessing and nested budgets

PIL grayscale conversion and square resizing produce 128 × 128 inputs. High-bit-depth inputs are rejected pending an explicit intensity-conversion method. GAN tensors are normalized to [-1,1]; classifier tensors repeat grayscale to three channels and use ImageNet normalization. Square resizing and low resolution are acknowledged limitations.

The implemented budget function retains all normal training images and takes nested prefixes of one seeded permutation of positive training images. Budgets are 25%, 50% and 100% of the frozen positive training count, rounded down with a minimum of one. The budget seed is fixed across training seeds. Budgets count images rather than independent patients or duplicate groups.

### 3.4 Candidate synthesis and classification methods

MediSyn implements a 128px convolutional GAN and a WGAN-GP critic without batch normalization or a sigmoid output. G1 trains the configured generator solely on real positive images in the corresponding training budget. The final fixed epoch is used, with no test-based selection. The default synthetic count fills the positive-image deficit relative to normal training images; actual counts and zero-deficit cases are recorded. These are proposed settings awaiting researcher approval, not empirically selected best settings.

A0 uses real-only random sampling with replacement. A1 uses inverse-class-frequency sampling with replacement, yielding equal expected class proportions. A2 adds conventional transforms to A0: horizontal flips, small affine rotation/translation, and brightness/contrast jitter. G1 adds generated positive images without silently adding A2 transforms. A reviewer must approve the biological plausibility of these transforms.

All conditions use ResNet-18, Adam, unweighted cross entropy and identical batch size, optimizer hyperparameters, validation cadence and total optimizer-update budget. The study runner does not early-stop. Validation loss selects the classifier checkpoint; validation F1 selects the decision threshold, with ties resolved by the highest threshold. Test and cohort subsets receive that fixed threshold. This threshold rule is not a validated clinical operating point. Default settings and all approval requirements are specified in docs/EXPERIMENT_PROTOCOL.md and configs/study.yaml.

Diffusion is optional and unimplemented in this release. Cross-cohort sensitivity training is not part of the pooled v1 runner. Neither is described as completed or infeasible based on unperformed experiments.

### 3.5 Evaluation and analysis implementation

TB recall is primary. Supporting metrics are AUPRC, F1, specificity and AUROC; the code retains per-image probabilities and reports Shenzhen and Montgomery subsets separately. Undefined ranking metrics for single-class subsets are represented as missing rather than invented values. Calibration reporting includes Brier score and a ten-bin expected calibration error.

Analysis recomputes metrics from saved probabilities, summarizes seed means and sample standard deviations, and verifies paired real training/test identities before calculating arm-minus-A0 differences. Confidence intervals use a percentile bootstrap of duplicate groups within cohort-composition/label strata. Paired seed differences have a separate descriptive seed bootstrap. Small-seed uncertainty is explicitly limited, and the software makes no automatic statistical-significance claim.

Optional Inception diagnostics calculate FID and polynomial-kernel KID; KID subset variability is not a confidence interval. Pixel diagnostics use resized grayscale RMSE for nearest training neighbours and diversity. These proximity screens neither establish privacy nor replace pathological review. Reference sets and sample counts must be declared; validation-based fidelity tuning must not consume the test set.

## 4. Results

**Publication results are unavailable.** Insert only measured, versioned artifacts after approval: final cohort/label/split counts; per-seed baseline and augmentation metrics; confidence intervals; cohort findings; generator fidelity/diversity/proximity summaries; failures and runtime. Software regression tests demonstrate execution behavior on artificial inputs, not medical quality or classifier improvement. Historical confounded smoke metrics are excluded from primary evidence.

## 5. Discussion

Interpretation awaits results. The final discussion should assess whether any benefit exceeds random oversampling and conventional augmentation, which budgets show benefit or harm, whether conclusions change by cohort, and whether fidelity tracks downstream utility. Mixed or negative findings must be retained. No advantage is claimed in this draft.

## 6. Limitations and Responsible Use

The dataset is small and drawn from two cohorts. Absence of established patient identifiers prevents a patient-disjointness claim. Candidate duplication and label conflicts require human resolution. Source, acquisition and cohort effects may confound apparent disease classification. Low-resolution grayscale preprocessing may discard relevant detail. Feature-space similarity and saliency/visual plausibility do not prove diagnosis-relevant fidelity. Models may memorize training content. No formal privacy guarantee, clinical validation, prospective evaluation or diagnostic-use authorization is established.

Original images, contact sheets, private review records and checkpoints are excluded from this code release. Dataset/checkpoint redistribution terms require explicit review; public availability is not treated as a blanket redistribution license. Local institutional requirements, ethics statements and any disclosure requirements must be completed by the researchers.

## 7. Conclusion and Future Work

The software provides an executable controlled-study framework. A conclusion about augmentation effectiveness remains pending. Future extensions may include a separately validated diffusion comparator, independent cohorts, higher-resolution preprocessing and formal privacy evaluation. None is claimed as achieved.

## 8. References

[1] Goodfellow IJ et al. Generative Adversarial Networks. 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661).

[2] Radford A, Metz L, Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. 2015. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434).

[3] Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G. Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. Quant Imaging Med Surg. 2014;4(6):475–477. [doi:10.3978/j.issn.2223-4292.2014.11.20](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20). [Author-hosted full text](https://www.lhncbc.nlm.nih.gov/LHC-publications/PDF/pub9356.pdf).

[4] Frid-Adar M, Diamant I, Klang E, Amitai M, Goldberger J, Greenspan H. GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification. Neurocomputing. 2018;321:321–331. [doi:10.1016/j.neucom.2018.09.013](https://doi.org/10.1016/j.neucom.2018.09.013). [Author preprint](https://arxiv.org/abs/1803.01229).

[5] Schütte AD, Hetzel J, Gatidis S, Hepp T, Dietz B, Bauer S, Schwab P. Overcoming barriers to data sharing with medical image generation: a comprehensive evaluation. npj Digit Med. 2021;4:141. [doi:10.1038/s41746-021-00507-3](https://doi.org/10.1038/s41746-021-00507-3). [Author preprint](https://arxiv.org/abs/2012.03769).

[6] Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. Improved Training of Wasserstein GANs. 2017. [arXiv:1704.00028](https://arxiv.org/abs/1704.00028).

[7] Heusel M, Ramsauer H, Unterthiner T, Nessler B, Hochreiter S. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS. 2017. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500).

[8] Bińkowski M, Sutherland DJ, Arbel M, Gretton A. Demystifying MMD GANs. ICLR. 2018. [arXiv:1801.01401](https://arxiv.org/abs/1801.01401).

## 9. Reproducibility Appendix

The repository includes configuration, review schema, immutable manifest checks, run provenance, metric recomputation and regression tests. The run root archives its approved protocol/configuration/manifest and approval attestation; each run records the commit, selected image IDs, checkpoint hash, seed, device, software version and runtime. The published paper must identify an exact commit and artifact release rather than a moving branch.

Before external submission, complete author contributions, affiliations, funding, competing interests, data-use/ethics statements, code license, model-release terms, task-specific literature coverage, final protocol deviations and all missing results. This working draft does not imply journal acceptance or endorsement.
