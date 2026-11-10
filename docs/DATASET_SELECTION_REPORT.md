# AURORA automated dataset selection — proposal v2

Verified 800 originals and 206 flagged pairs: retained 590, excluded 210 unique images.
No original image/reference/audit was modified. Exclusions are similarity-based, not confirmed duplicates. No human adjudication or training occurred.

| Cohort / source label | Original | Retained | Excluded | Excluded % | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| montgomery / normal | 80 | 75 | 5 | 6.25% | 53 | 11 | 11 |
| montgomery / tuberculosis | 58 | 52 | 6 | 10.34% | 36 | 8 | 8 |
| shenzhen / normal | 326 | 214 | 112 | 34.36% | 150 | 32 | 32 |
| shenzhen / tuberculosis | 336 | 249 | 87 | 25.89% | 174 | 38 | 37 |

## Marginal exclusion rates

| Group | Original | Retained | Excluded % |
|---|---:|---:|---:|
| montgomery | 138 | 127 | 7.97% |
| shenzhen | 662 | 463 | 30.06% |
| normal | 406 | 289 | 28.82% |
| tuberculosis | 394 | 301 | 23.60% |

Split seed: 42. Counts: train 413, validation 89, test 88.
Within each cohort/label stratum: seeded shuffle, floor 70/15/15 targets, allocate remainders by descending fractional part (ties train, val, test). One fixed split; labels unchanged.

## Nested TB training budgets

| Budget | Normal training images | TB training images |
|---|---:|---:|
| 25% | 203 | 52 |
| 50% | 203 | 105 |
| 100% | 203 | 210 |

## Limitations and next boundary

Exclusion may change acquisition, anatomy and disease distributions and preferentially remove common-looking images. Exclusion rates differ across cohorts/classes; findings apply to this selected subset. Retained images are not proven duplicate-free, and patient identifiers are unavailable: patient-level independence cannot be claimed. Source labels are not clinically revalidated. Montgomery holdouts are small; cohort results are descriptive and no independent cross-cohort experiment is required.

The simplified four-condition protocol is proposed in docs/EXPERIMENT_PROTOCOL_SIMPLIFIED.md. Diffusion is deferred for scope/hardware reasons. Researcher review of these counts, split and protocol is required before any first training run. Earlier approval does not authorize this revised run.

## Source hashes

- original_reference: `c8592c06e852f8166cebdde73b937b85b56110941c7a3fc03b3a67f943201347`
- candidate_pair_index: `a0e46a33ea0cf6a3df1f2fb3f7260a7e79ff206a0268f8ac24bc725f466e3266`
