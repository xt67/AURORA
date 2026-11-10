# Separate v2 DCGAN feasibility entry point

Implemented 11 November 2026. **Not executed; final researcher approval remains required.** This entry point is separate from the historical publication runner and its v1 gate. No human review decisions are created or used by v2 automated-exclusion approval.

## Gate behaviour

`src.study.v2_gate.feasibility_preflight` requires an existing JSON approval with schema 2, scope `aurora_v2_dcgan_feasibility_only`, a nonblank researcher identity, timezone-qualified timestamp and all four flags explicitly true: `dataset_approved`, `split_approved`, `protocol_approved`, `feasibility_run_approved`. Approval is a researcher attestation; software does not authenticate a signature. A draft, missing file, v1 approval or publication approval does not authorize feasibility.

Approval binds SHA-256 values of seven inputs: proposal configuration, protocol, original reference, candidate-pair index, retained manifest, exclusion manifest and split JSON. It also binds the ordered list of 52 source IDs, selected-row digest, exact effective-generator-config digest, Git commit and a hash of the listed implementation files. A changed input, selection or implementation fails closed; runtime files must be committed. The approved device is explicit (`cuda` or `cpu`); no automatic fallback is allowed.

The gate reconstructs the full flagged-image union and verifies all retained/excluded metadata and pair memberships. It reconstructs the deterministic split and checks the accepted 800 originals, 206 pairs, 210 exclusions, 590 retained images, cohort/class counts and 413/89/88 split. It selects only the seed-42 nested 25% TB training budget: exactly 52 sources. No manual-decision file is read, synthesized or bypassed. Existing exclusion records retain their immutable historical wording; the gate accepts only the exact older reason or its corrected image-based wording, checking all other fields exactly. Newly generated records say "image".

## Image-access boundary and bounded work

The preflight reads metadata for all partitions, not their image bytes. `ApprovedImageReader` accepts only the exact approved IDs and train/positive rows. It rejects nonselected IDs, altered paths, holdout rows, path escapes/redirections and changed source-image hashes. It hashes and decodes the same byte buffer for each permitted image.

Only those 52 images are preloaded, using deterministic grayscale 128px preprocessing and normalization to [-1,1]. The GAN receives an in-memory `TensorDataset`, with no dataset root, image paths, holdout loader or original manifest. This is an application-level boundary around the trusted trainer, not an operating-system filesystem sandbox.

The effective configuration is constructed explicitly: BCE DCGAN; seed 42; **two epochs**; latent dimension 100; widths 64/64; batch size 16; Adam rates .0002, betas .5/.999; one discriminator step; full precision; label smoothing .9; checkpoint at epoch two. The loader has four batches (16/16/16/4), zero workers and no dropped samples. The publication setting of 200 epochs is never inherited. There is no resume option, classifier training, validation/test evaluation, architecture search or automatic retry.

## Commands for the approved workflow

From the repository root in the existing `aurora-cuda` environment, prepare a **false-flag template** after the implementation is committed:

```powershell
python -m src.study.v2_gate --config configs/study_simplified_proposal.yaml --device cuda --draft outputs/feasibility_v2_approval.draft.json
```

This validates metadata and writes bindings/source IDs, but opens no image and grants no approval. It refuses overwrite. After final researcher authorization, save a separate local `outputs/feasibility_v2_approved.json` with the real reviewer identity/time and explicit approvals. Keep `training_authorized: false` in the proposal: feasibility permission is separate from publication permission. Regenerate bindings if any bound input or implementation commit changes.

Check approval and the 52 source images without training:

```powershell
python -m src.study.feasibility --config configs/study_simplified_proposal.yaml --approval outputs/feasibility_v2_approved.json --check-only
```

**Only after final approval**, execute the two-epoch feasibility run into a new directory:

```powershell
python -m src.study.feasibility --config configs/study_simplified_proposal.yaml --approval outputs/feasibility_v2_approved.json --output outputs/feasibility_v2/run_01
```

The check-only path cannot invoke the trainer or create run outputs. Execution rejects existing output directories and any output under the original image root. Originals, manifests, audits, prior experiments and historical protocols are read-only inputs. No approval file has been granted by this implementation task.

## Recorded artifacts

Each execution archives exact input bytes and approval, effective generator YAML/hash, all selected IDs and image hashes, actual read IDs, input/selection/implementation digests, commit, seed, source count, software/device details and non-publication scope. Success records total runtime, training runtime, checkpoint hash and epoch/config checks. CUDA records starting free/total memory and peak allocated/reserved memory; an approved CPU run records memory as not applicable. Training/device failures retain `failure.json` with runtime/provenance; invalid preflight/source inputs fail before output creation. No failures or prior outputs are overwritten.

## Remaining blockers

Focused verification on 11 November 2026: **39 passed in 13.86s**, with 13 existing dependency deprecation warnings. This covers the v2 gate/reader/orchestration, selection/data helpers and two historical v1 gate regressions. Trainer execution and CUDA counters are stubbed; no optimizer or real training ran.

Final approval has not been supplied; the real two-epoch run must not start. Actual runtime, GPU capacity, losses, image quality and checkpoint behaviour on these real sources remain unmeasured. Stubbed software tests establish orchestration and rejection behaviour only.

The later four-condition publication runner still needs its separate v2 integration: TB-only conventional transforms with flips disabled, matched selected-positive B/D counts, A–D reporting and separate publication authorization. Feasibility approval cannot authorize that later study. No diffusion work or changes to the v1 approval gate are included here.
