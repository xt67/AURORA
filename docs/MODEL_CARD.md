# AURORA candidate model card

Release status: software only. No publication-trained checkpoint or clinically validated model is released.

## Intended use
Controlled research on synthetic augmentation of TB-manifestation-labelled chest radiographs. The Streamlit demo displays clearly synthetic images from a locally configured checkpoint. No diagnosis, screening recommendation or patient-management decision is supported.

## Architectures and inputs
DCGAN and WGAN-GP share the existing 128px grayscale generator with a noise-vector input. WGAN-GP uses an unbounded critic without batch normalization. Generator training runs in full precision. ResNet-18 consumes replicated grayscale RGB tensors with ImageNet normalization. The study runner uses AMP only for CUDA classifier training.

## Data and selection
Candidate primary data are the original NLM cohorts, pending human review. The generator is restricted to the selected positive real training subset. The configured final generator epoch is retained; it is not called a best-quality checkpoint. Classifier checkpoints use validation loss; operating thresholds use validation F1. Test data are excluded from training and selection.

## Evidence and limitations
Software fixtures test CPU/CUDA execution, checkpoints, image export and metrics. They are not evidence of medical fidelity, privacy or downstream improvement. Small-data memorization, cohort confounding, low resolution and lack of patient identifiers remain limitations. FID/KID and pixel-neighbour screens do not prove clinical validity or privacy.

## Required checkpoint release metadata
Before releasing a trained checkpoint, fill in: commit/config/manifest hashes; training budget and seed; generator family and epoch; checkpoint hash; image counts; dataset permissions; fidelity/diversity/proximity results; downstream per-seed and cohort results; failure cases; review identity/date; compute/runtime; allowed uses and exclusions. No values are fabricated in this card.

Load only project-produced or otherwise trusted model artifacts. No checkpoint-upload feature is provided. Dataset and checkpoint redistribution permissions are separate from access to the software source.
