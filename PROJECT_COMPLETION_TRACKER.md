# AURORA project status

Updated 20 November 2026. Default branch: `master`. Repository: https://github.com/xt67/AURORA.

## Completed

- Preserved all 800 original images and the scored similarity audit.
- Selected 590 images by excluding all 210 unique images in 206 flagged pairs, without claiming confirmed duplicates.
- Prepared the stratified 413/89/88 image split and nested 52/105/210 TB training budgets.
- Implemented a separate v2 feasibility approval gate, exact source checks and a tensor-only generator loader.
- Completed one two-epoch DCGAN run on 52 TB training images, seed 42, using an RTX 4050; no holdout images were read by the generator.
- Recorded source IDs, hashes, commit, checkpoint, losses, runtime and memory. See [execution evidence](docs/FEASIBILITY_V2.md#first-real-feasibility-execution).
- Consolidated GitHub into one branch, added Apache 2.0 and prepared a [faculty-review manuscript](paper/professor_review/manuscript.pdf).
- Implemented the separate v2 publication runner and tested approval, source, transform and holdout rejection paths.
- Completed the authorized 36-classifier/nine-GAN grid at commit `c915a7b`, with 36,000 classifier updates and 1,101 generated training images.
- Recomputed all validation thresholds and test metrics, verified checkpoint selection and confirmed zero generator/holdout overlap. All 800 originals and 11 protected metadata/history artifacts still match their pre-run hashes.
- Completed 1000-replicate uncertainty analysis, per-seed/cohort/calibration reporting and matched D–A/D–B contrasts. Prepared the [new measured-results manuscript](paper/publication_v2/manuscript.pdf) and editable Word copy; the prior feasibility manuscript remains historical.

## Evidence boundary

The experiment grid completed, but the generator did not produce convincing radiographs in inspected fixed-order grids: all nine fits showed repetitive noise-like patterns. The classification comparisons describe this failed-synthesis implementation, not successful disease-preserving augmentation. The historical v1 approval gate remains unchanged. Dataset selection is image-level; patient identifiers are unavailable.

Recorded focused feasibility tests: 39 passed (11 November 2026). These software tests used stubs where documented and are separate from the real training result.

## Remaining research and release work

1. Obtain faculty/scientific review of the measured results and manuscript. Computational verification and manuscript preparation are complete; this is not paper acceptance or clinical validation.
2. If the intended product must generate convincing radiographs, repair and separately evaluate the generator. That product objective remains unmet. Do not repeatedly tune against the now-exposed test set.
3. Recover the original candidate-screening implementation/thresholds if available; its frozen pair index and deterministic exclusion rule are preserved, but end-to-end screening reproducibility remains limited.
4. Obtain faculty/venue review and applicable data-use/ethics determinations. The author confirmed corresponding email, no funding and no competing interests; no institutional approval is invented.
5. Configure public hosting only if required. No public deployment is claimed.

Original data, historical protocols and prior experiment artifacts remain unchanged. Earlier preparation-stage instructions are historical; the current record above includes the subsequently approved feasibility execution.
