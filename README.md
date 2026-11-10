# AURORA / MediSyn

A research prototype for evaluating synthetic chest X-ray augmentation under limited TB-labelled training data. MediSyn is the companion Streamlit demo.

This is an experimental comparison: it reports what selected methods do under defined conditions. Positive, negative and inconclusive findings are valid outcomes. Demonstrating a universally best method or clinical readiness is not a completion requirement.

**Status: simplified v2 dataset and protocol prepared; training awaits researcher review.** See [execution tracker](PROJECT_COMPLETION_TRACKER.md) and [test report](docs/TEST_REPORT.md) for measured software checks and remaining limitations. The demo displays no invented performance metrics.

## Research scope

The current v2 proposal retains 590 of 800 original NLM images after automated exclusion of all 210 images participating in the 206 flagged candidate pairs. These are not confirmed duplicates; no original image was deleted. See the [verified dataset report](docs/DATASET_SELECTION_REPORT.md), [simplified protocol](docs/EXPERIMENT_PROTOCOL_SIMPLIFIED.md) and [proposed configuration](configs/study_simplified_proposal.yaml). Patient independence and a duplicate-free subset are not claimed.

Synthetic images are learned from real training images. They do not automatically guarantee privacy, diagnostic validity or improved classifier performance. The current implementation is for chest X-ray research; MRI and clinical deployment have not been validated.

## Environment

Use Python 3.10–3.12 and the pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

On the current development machine, the existing `aurora-cuda` Conda environment has the required dependencies and CUDA PyTorch. The default system Python lacks the test dependencies.

## Demo

```powershell
python -m streamlit run app/streamlit_app.py
```

The app works without a checkpoint and explains that generation/results are unavailable. See [deployment preparation](docs/DEPLOYMENT.md) for environment variables and the untested container candidate.

## Historical toolkit GAN training and export (not authorized for v2 yet)

Only use a directory containing the approved **training partition**, with class folders such as `Normal/` and `Tuberculosis/`. Do not point this command at the full dataset. These commands do not perform data adjudication or establish leakage-free splits for you.

```powershell
python -m src.gan.train_gan --config configs/dcgan.yaml --data-dir data/final/train --target-class Tuberculosis --output-dir outputs/gan
python -m src.gan.inference --checkpoint outputs/gan/models/generator_latest.pth --output-dir data/synthetic/run_01 --count 100 --seed 42
```

Use `configs/wgan_gp.yaml` for WGAN-GP. Both GAN paths currently use full precision. Checkpoints include architecture metadata; the most recent checkpoint is not necessarily the best-quality model. Exported files are explicitly named synthetic and include a generation provenance manifest.

## Components

- `src/data_loader.py`: exploratory loaders and GAN/classifier transforms. Random splitting is not the final NLM study protocol.
- `src/gan/`: DCGAN/WGAN-GP training, checkpoint inference and PNG export.
- `src/classifier/`: ResNet-18 training and evaluation.
- `src/evaluate/`: classification metrics and FID.
- `src/study/`: reviewed manifests, publication gate, controlled A0/A1/A2/A3/G1 experiments and reproducible analysis.
- `app/streamlit_app.py`: research demo.
- `tests/`: model, training, data and app regression tests.
- `paper/manuscript.md`: external-review working draft with verified references and implemented methods; research results pending.

The revised study runner is implemented and tested on artificial fixtures. Diffusion is an optional, unimplemented extension. Publication experiments have not started. Obsolete exploratory scripts, duplicate planning documents and generated artifacts have been removed from the public branch; local research records are preserved.

## Current preparation boundary

The simplified proposal compares real-only training, random real-TB oversampling, TB-only conventional augmentation and one DCGAN method. There are three nested TB budgets and three seeds: 36 proposed classifier fits and nine GAN fits. No revised publication or feasibility training has started.

```powershell
python -m src.study.selection --reference data/reference/nlm_original/reference_manifest.csv --index outputs/provenance/manual_review_index.csv --root data/reference/nlm_original --output outputs/provenance
```

The command verifies original hashes/dimensions, preserves all source files, writes retained/exclusion manifests and a proposed 413/89/88 split, and refuses existing output files. Use a new directory for reproduction. Review these outputs before approving the first training run. The v2 configuration is proposal-only; the historical runner has not been adapted to its automated-exclusion gate, TB-only transforms or revised count rule. Do not use fabricated human decisions to pass the old gate. [Prior protocol history](docs/protocol_history/EXPERIMENT_PROTOCOL_v1_1.md) is preserved.

Further documentation: [architecture](docs/ARCHITECTURE.md), [NLM dataset card](docs/DATASET_NLM.md), [model card](docs/MODEL_CARD.md), [release notes](docs/RELEASE_NOTES.md), [software citation](CITATION.cff). Raw/reference datasets and private reviewer records stay local.

For completed historical-runner experiments, the existing reporting utility exports measured tables and demo reports (v2 integration follows approval):

```powershell
python -m src.study.report --run outputs/publication_v1 --output outputs/report_v1
```

The exporter requires the complete configured run grid, checks frozen-file hashes and recomputes metrics. It writes `results_table.md` and `paper_results.json`, including all arms, seed variability, cohort results and A0/A3 contrasts. Source image IDs and local paths are omitted. Set `AURORA_RESULTS` to the exported JSON after inspecting it. Exporting does not independently validate the science or submit the paper.

## Release and license

No software license has yet been selected. Data and model redistribution rights must be checked before publication. Do not interpret publicly accessible source images as unrestricted redistribution permission.
