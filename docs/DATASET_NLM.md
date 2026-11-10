# Candidate primary dataset: NLM chest radiographs

This card supersedes the mixed-source Kaggle card for the proposed primary study. The older card and exploratory results remain historical records.

| Cohort | Normal | TB-manifestation-labelled | Candidate total |
|---|---:|---:|---:|
| Shenzhen | 326 | 336 | 662 |
| Montgomery | 80 | 58 | 138 |
| Combined | 406 | 394 | 800 |

These are pre-adjudication counts, not an approved final analysis population. Original sources and label meanings are described in [Jaeger et al., 2014](https://www.lhncbc.nlm.nih.gov/LHC-publications/PDF/pub9356.pdf). The task is image-label classification; the labels do not establish microbiological confirmation for this study.

Local originals are stored in `data/reference/nlm_original/{shenzhen,montgomery}/`. `reference_manifest.csv` records source URLs, dimensions and SHA-256 hashes. No images or private review records are included in the software release. Obtain data from the [official NLM resource](https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html) under its current terms; preserve the downloaded notices. Broad redistribution rights for images or trained checkpoints have not been established by this project, so no public data download is provided.

The existing audit reports zero exact SHA-256 duplicates and 206 similarity candidates: 64 cross-label and 13 high-similarity candidates. No human review has been completed. Follow HUMAN_REVIEW.md; similarity alone does not establish identity. The proposed canonical tool rejects unresolved cases and cross-label duplicate conflicts.

The final manifest will preserve cohort identities, hashes, reviewed duplicate groups and frozen partitions. No patient identifiers have been established, so image/group-disjoint partitions are not called patient-disjoint. Controlled scarcity is created inside the final training partition; the source collection is approximately balanced.

Preprocessing is grayscale square resize to 128px. GAN inputs use [-1,1]; classifier inputs use replicated grayscale and ImageNet normalization. High-bit-depth inputs are rejected until an explicit intensity mapping is approved. Square resize may distort aspect ratio; low resolution may discard detail. External validity, source effects, duplicate uncertainty and clinical validity require investigation.
