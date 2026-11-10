# AURORA / MediSyn

A research prototype for evaluating synthetic chest X-ray augmentation under limited TB-labelled training data. MediSyn is the companion Streamlit demo.

**Status: software release candidate; publication experiments remain gated by human review.** See [execution tracker](PROJECT_COMPLETION_TRACKER.md) and [test report](docs/TEST_REPORT.md) for measured software checks and remaining limitations. The demo displays no invented performance metrics.

## Research scope

The current proposal uses the original NLM Shenzhen and Montgomery collections with experimentally reduced TB-labelled training budgets. The local dataset review has 206 unresolved image-pair candidates. Final duplicate groups, exclusions and cohort-aware splits must be established before publication experiments. Follow [the current paper roadmap](AURORA_PAPER_ROADMAP.md), not the historical mixed-source Kaggle protocol.

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

## GAN training and export

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
- `src/study/`: reviewed manifests, publication gate, controlled A0/A1/A2/G1 experiments and reproducible analysis.
- `app/streamlit_app.py`: research demo.
- `tests/`: model, training, data and app regression tests.
- `paper/manuscript.md`: external-review working draft with verified references and implemented methods; research results pending.

The revised study runner is implemented and tested on artificial fixtures. Diffusion is an optional, unimplemented extension. Publication experiments have not started. Historical status documents are retained for context, not as evidence of current readiness.

## Controlled research workflow

1. Complete the [local human review](docs/HUMAN_REVIEW.md). No review decisions are supplied by the software.
2. Create and inspect a hash-verified canonical proposal. Approve the dataset, frozen split and [executable protocol](docs/EXPERIMENT_PROTOCOL.md).
3. Supply a researcher-approved local file based on `configs/approval.example.json`. The committed example intentionally refuses publication training.
4. Run the preflight, then the exact approved experiment configuration:

```powershell
python -m src.study.runner --config configs/study.yaml --approval outputs/approved_run.json --output outputs/publication_v1 --check-only
python -m src.study.runner --config configs/study.yaml --approval outputs/approved_run.json --output outputs/publication_v1
```

Use a new output directory. The runner records failures, provenance, fixed-update training, validation-selected thresholds and real test predictions. See the protocol for analysis and synthesis-diagnostic commands. No medical images or checkpoints are committed.

Further documentation: [architecture](docs/ARCHITECTURE.md), [NLM dataset card](docs/DATASET_NLM.md), [model card](docs/MODEL_CARD.md), [release notes](docs/RELEASE_NOTES.md), [software citation](CITATION.cff). Raw/reference datasets and private reviewer records stay local.

## Release and license

No software license has yet been selected. Data and model redistribution rights must be checked before publication. Do not interpret publicly accessible source images as unrestricted redistribution permission.
