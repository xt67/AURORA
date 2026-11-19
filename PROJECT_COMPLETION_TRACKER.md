# AURORA project status

Updated 19 November 2026. Default branch: `master`. Repository: https://github.com/xt67/AURORA.

## Completed

- Preserved all 800 original images and the scored similarity audit.
- Selected 590 images by excluding all 210 unique images in 206 flagged pairs, without claiming confirmed duplicates.
- Prepared the stratified 413/89/88 image split and nested 52/105/210 TB training budgets.
- Implemented a separate v2 feasibility approval gate, exact source checks and a tensor-only generator loader.
- Completed one two-epoch DCGAN run on 52 TB training images, seed 42, using an RTX 4050; no holdout images were read by the generator.
- Recorded source IDs, hashes, commit, checkpoint, losses, runtime and memory. See [execution evidence](docs/FEASIBILITY_V2.md#first-real-feasibility-execution).
- Consolidated GitHub into one branch, added Apache 2.0 and prepared a [faculty-review manuscript](paper/professor_review/manuscript.pdf).

## Evidence boundary

The run confirms that the specified software path executes. It does not establish image quality, privacy, clinical validity or downstream classification improvement. The historical v1 approval gate remains unchanged. Dataset selection is image-level; patient identifiers are unavailable.

Recorded focused feasibility tests: 39 passed (11 November 2026). These software tests used stubs where documented and are separate from the real training result.

## Remaining research and release work

1. Integrate and test the v2 publication runner, including TB-only conventional transforms and matched B/D added counts.
2. Obtain separate authorization for, then execute, the planned 36-classifier/nine-GAN grid.
3. Analyze measured classification outcomes and declared image-quality/memorization diagnostics; retain failures and negative results.
4. Complete initial similarity-screening documentation, author declarations and venue-specific submission requirements.
5. Configure and verify public hosting if deployment is required. No public deployment is claimed.

Original data, historical protocols and prior experiment artifacts remain unchanged. Earlier preparation-stage instructions are historical; the current record above includes the subsequently approved feasibility execution.
