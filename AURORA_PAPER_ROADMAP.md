# AURORA Paper Roadmap

**Status:** Proposal only. No experiments have been run for the revised NLM-based protocol.

**Acronym note:** The expansion "Augmented Underrepresented Rare-disease Observations using Realistic Artificial Imaging" is provisional. The study concerns TB-associated radiographic manifestations in an approximately class-balanced NLM collection, so the "rare-disease" wording should be reconsidered before final submission.

## 1. Introduction
Define the problem of limited and imbalanced TB chest X-ray data, the goal of synthetic augmentation, and the contribution: a controlled, reproducibility-focused evaluation of synthetic TB image utility. State the research question and why negative or mixed results are acceptable.

## 2. Related Work
Summarize prior work on medical image synthesis (GANs and diffusion), synthetic data augmentation for imbalanced medical datasets, conventional augmentation, fidelity metrics, and limitations of FID in medical imaging. Include memorization/privacy concerns and small-dataset risks. Citations must be verified before submission.

## 3. Materials and Methods
- **Dataset:** Candidate primary data are the original NLM collections: Shenzhen (326 normal, 336 TB-labelled) and Montgomery (80 normal, 58 TB-labelled), totaling 800 images. The original NLM collection is approximately class-balanced. TB-positive training-data scarcity is created experimentally by reducing the number of real TB-labelled training images; the original dataset should not be described as naturally imbalanced. Do not describe the dataset as 3,500 normal and 700 TB.
- **Labels:** Image-level TB-manifestation labels derived from official documentation. No patient identifiers are available.
- **Preprocessing:** Document resizing, grayscale handling, normalization, and any duplicate/near-duplicate handling.
- **Splitting:** Use a proposed cohort-aware, duplicate-aware 70/15/15 split stratified by cohort and label. The 206 near-duplicate candidate pairs remain under human adjudication, so the final dataset and split are not yet approved or duplicate-safe. Report exact counts only after adjudication. State clearly that no patient-level independence can be claimed.
- **Scarcity simulation:** Define 25%, 50%, and 100% budgets relative to the final available TB-labelled training partition after dataset adjudication and splitting. Use reproducible nested subsets across budgets so that each smaller budget is contained within the larger one. Do not change the underlying dataset.
- **Generative models:** Mark model selection as pending. Do not assume DCGAN, WGAN-GP, or diffusion models are final selections.
- **Baselines:** Real-only training, random oversampling, conventional augmentation, GAN-based augmentation, and diffusion-based augmentation if feasible. Ensure fair comparisons by holding classifier architecture, preprocessing, optimizer settings, training-update budget, and evaluation procedures consistent across all conditions.
- **Classifier:** Fixed architecture and hyperparameters across conditions; class weights and early stopping where applicable.
- **Leakage controls:** Synthetic data generated from training data only; validation and test sets remain real and fixed. Include separate Shenzhen and Montgomery subgroup evaluation in the pooled primary analysis. Treat cross-cohort evaluations as separate sensitivity analyses.

## 4. Results
Placeholder only. Do not fabricate results. Future content should include dataset audit summaries, split counts, fidelity/diversity/memorization analyses, and downstream classifier metrics. Required metrics: TB recall as primary outcome, plus AUPRC, F1, specificity, and AUROC.

## 5. Discussion
Placeholder until experiments are complete. Interpret whether synthetic augmentation helps, when it helps, and how fidelity relates to downstream utility. Compare generative families only if both are evaluated. Discuss source confounding and cohort-related limitations.

## 6. Limitations and Responsible Use
- No patient identifiers; image-level split only.
- Unresolved near-duplicate candidates and any confirmed duplicates.
- Cohort differences between Shenzhen and Montgomery.
- Small Montgomery subset and limited external validity.
- Single disease/dataset scope.
- Resolution and preprocessing constraints.
- FID/Inception limitations in medical imaging.
- Potential memorization or privacy risk from synthetic data.
- No clinical validation or diagnostic claims.

## 7. Conclusion and Future Work
Placeholder until results are available. State conclusions only from measured evidence. Future work may include patient-level datasets, external validation, higher resolution, improved fidelity/diversity evaluation, and privacy/memorization analysis.

## 8. References
Use only verified citations. Do not invent references. Add bibliographic details for all datasets, models, metrics, and prior studies during drafting.

## 9. Reproducibility Appendix
Include seeds, configuration, dataset versions/provenance, split definitions, duplicate-audit summaries, preprocessing details, training settings, hardware (RTX 4050 6 GB VRAM, 32 GB RAM), runtime notes, and artifact paths. Report unresolved decisions and any deviations from the pre-registered protocol.

## Corrections to earlier draft assumptions
- Do not describe the primary dataset as 3,500 normal and 700 TB.
- Do not describe the NLM dataset as having a natural 5:1 class imbalance.
- TB scarcity is simulated through controlled training budgets.
- Replace random class-stratified splitting with cohort-aware, duplicate-aware splitting.
- Keep absence of patient identifiers as an explicit limitation.
- Mark generative model selection as pending.
- Include random oversampling as a separate baseline.
- Keep abstract, results, and conclusion as placeholders until experiments are completed.