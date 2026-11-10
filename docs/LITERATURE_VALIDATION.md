# Literature basis and validation limits

Reviewed 10 November 2026 from three researcher-supplied PDFs. Papers supply evidence and methodological comparisons, not instructions, approval signatures or AURORA results. No attached PDF is redistributed in this repository.

| Source | Relevant evidence | Application and limits in AURORA |
|---|---|---|
| Folguera-Profitós et al. (2026), systematic review, JMIR AI 5:e81472. [DOI](https://doi.org/10.2196/81472) | Reviews 160 studies; heterogeneous methods and risk of bias limit generalization. Its descriptive synthesis is not a pooled causal effect estimate. | Separate synthesis quality, downstream utility and clinical validity. Keep negative outcomes, provenance and limitations. Review-level findings cannot validate our images or predict improvement. |
| Morís et al. (2024), Medical & Biological Engineering & Computing 62:2189–2212. [DOI](https://doi.org/10.1007/s11517-024-03056-5) | Studies adapted latent diffusion and CUT on Montgomery, Shenzhen and JSRT; screening tests use original images. Sections 3–4 describe methods and substantial A100 hardware. | Supports real-only evaluation and a TB-specific literature comparison. Our DCGAN/WGAN-GP, resolution, protocol and GPU differ; their accuracy is neither our result nor evidence that their training fits our hardware. Diffusion remains optional and unimplemented. |
| Pink & Sykes (2026), PMLR 318:954–961. [Proceedings record](https://proceedings.mlr.press/v318/pink26a.html) | Sections 3–4 and Appendix compare synthetic augmentation with real duplication under matched classifier updates; superior fidelity did not establish superior downstream utility in that study. | Add A3, an exact added-count duplicate-real comparator for G1, alongside A1 class-balanced sampling. This is related prior work, not a new contribution invented here. Their mixed-source dataset, DenseNet, resolutions and ratio definition differ from ours. |

The supplied Pink PDF has a placeholder `DOI: 0`; the verified PMLR proceedings record above is the citation authority. Do not cite the placeholder as a DOI. Its author listing uses Ed Sykes; the PDF uses Edward R. Sykes.

## Implemented response

The proposed protocol now includes A3. For each budget/seed it repeats positive **training** rows with replacement, adding exactly the number of images G1 would generate. Its class composition and nominal dataset size match G1 at any configured ratio. All arms retain equal classifier updates. A1 balances class sampling in expectation; A3 explicitly records finite duplicated source IDs. Neither creates independent real patients. At zero deficit A3 adds nothing. Analysis supports `--baseline A3`, rejects unequal A3/G1 added counts and retains the existing real identity and classifier checks.

This is a protocol amendment awaiting researcher approval. Changed protocol/config hashes invalidate any old approval. It is not permission to start publication training. Our ratio multiplies the positive-class deficit; it is **not** the synthetic-to-real ratio used by Pink and Sykes. Do not equate numeric ratios across studies.

FID/KID are optional distribution diagnostics; pixel nearest-neighbour screens are preliminary memorization checks. We do not add SSIM between unrelated generated/real images as if each had a known ground-truth target. No diagnostic correctness, privacy guarantee or clinical validation follows from these measures.

## Working independently

A supervisor is not required by this software. The sole researcher can inspect technical duplicate identity, document source metadata, approve their protocol and run reproducible experiments. Record the actual reviewer identity and disclose self-review; do not describe it as independent expert review. Clinical relabelling cannot be justified by resemblance, another paper's results or an AI-generated diagnosis. Investigate source metadata for cross-label candidates; genuinely unresolved cases remain unresolved until an explicitly documented adjudication or exclusion protocol is approved and implemented.

The 206 local candidate pairs remain unreviewed. These papers have not resolved them. No canonical dataset or publication run is approved, and no effectiveness results exist. Do not manufacture a supervisor, clinician, ethics approval or affiliation. Any institution/venue requirements must be checked for the eventual submission.

## Contribution and manuscript

The defensible proposed scope is a reproducible, cohort-aware scarcity evaluation on original NLM images with transparent duplicate review and fixed-update controls. Controlled synthetic augmentation and duplication controls already exist in the literature. Claiming novelty or utility requires completed experiments and a broader literature comparison. Negative or inconclusive results are reportable; publication acceptance is not guaranteed.

Before submission, report actual exclusions, split/group counts, seeds, compute, all arms, uncertainty, failures, cohort results and limitations. Software regression tests validate execution only. The manuscript must retain pending results until measured artifacts exist.
