# Software verification report — 10 November 2026

## Environment
Windows; existing `aurora-cuda` Conda environment. Python 3.11.17, PyTorch 2.2.0+cu121, torchvision 0.17.0+cu121, NumPy 1.26.4, scikit-learn 1.4.2, Streamlit 1.32.0. CUDA is available on the local RTX 4050 laptop GPU. No environment reinstall was performed.

## Measured checks

| Check | Observed outcome |
|---|---|
| Independently reproduced original suite | 30 passed in 11.37s |
| Core repairs plus input/logging regressions | 36 passed in 13.70s |
| Review/manifest focused tests | 9 passed in 2.82s |
| Study runner and analysis focused tests | 13 passed in 14.09s |
| Final complete suite | **72 passed in 24.90s**, 13 dependency deprecation warnings |
| Local wheel build | `aurora-0.2.0-py3-none-any.whl` built without downloading dependencies |
| Wheel installation/import | Installed into an isolated project folder; imported installed study runner successfully |
| `python -m pip check` | No broken requirements found |
| Bounded Streamlit HTTP check | `/_stcore/health` returned HTTP 200 and `ok`; server stopped |

An intermediate final-suite run found one CUDA memory-query compatibility defect (64 passed, 1 failed). PyTorch 2.2 requires an explicit device index for `mem_get_info`; the code now supplies it. The complete rerun passed. Deprecation warnings originate in matplotlib/pyparsing and were not suppressed.

Latest pass includes A3 fixture training, deterministic duplication, non-integer/invalid ratios and A3/G1 count-matched analysis rejection. Previously recorded wheel/app checks predate this amendment; they were not rerun for unchanged app behavior.

Latest reporting checks cover complete-grid enforcement, rejection of exploratory/config-mismatched inputs, private-field omission, frozen-file hash checks, overwrite refusal, and measured-report display in Streamlit.

## Coverage
Dataset transforms and labels; immutable split preservation; high-bit-depth rejection; finite binary metric inputs; isolated logs; CPU/CUDA GAN and WGAN-GP optimization; critic group independence; checkpoint round trips; deterministic generation; valid PNG encoding; classifier synthetic-batch progression; CPU and CUDA fixed-update classification; all A0/A1/A2/A3/G1 fixture arms; nested budgets; reviewed duplicate grouping; changed-image/hash leakage detection; missing/changed approval rejection; validation-selected thresholds; grouped bootstrap, paired identity checks and calibration; KID estimator arithmetic; nearest-neighbour matching; local review history/path safety; all demo pages, actual fixture-model generation and corrupt checkpoint handling.

All training tests use small artificial images and untrained/tiny fixture models. They do not use unresolved NLM images to claim experimental success. Human review decisions in tests are explicitly artificial fixtures and are never written into the actual review package.

## Not verified / not claimed
- No publication experiments, real-data synthesis quality, TB classification benefit, clinical validity or privacy guarantee.
- No human adjudication of the real 206 pairs; no canonical approval or first-run approval.
- No clean-machine dependency download/install test, container build/runtime test or public hosting deployment. Docker is unavailable locally.
- No full pretrained Inception feature extraction/FID/KID run; mathematical estimator and tensor export paths have tests. Downloading feature weights remains an explicit optional action.
- No diffusion implementation or cross-cohort sensitivity experiment.

Run the suite from the repository root with `python -m pytest -q`. The default system Python is not the tested Conda interpreter. In the Codex Windows sandbox, temporary-file access required running these tests with approval outside the sandbox.
