# AURORA / MediSyn execution status

Updated 10 November 2026. Working branch: `aurora-refinement`.

| Area | Status |
|---|---|
| Software | GAN training/export, classifier, review UI, canonical manifests, gated fixed-update runner and analysis implemented |
| Literature | Three supplied papers reviewed; methods and limits documented in [literature notes](docs/LITERATURE_VALIDATION.md); references added to manuscript |
| Experimental control | A3 repeats positive training rows with exactly G1's added count; source IDs recorded; A3/G1 paired analysis checks matching counts |
| Reporting | Complete-grid exporter recomputes results, checks frozen provenance and writes paper tables/demo JSON without private row IDs or paths |
| Verification | Artificial fixture and regression checks documented in [test report](docs/TEST_REPORT.md); they do not validate medical images |
| Repository | Obsolete download/baseline scripts, duplicate planning docs, stale requirements, generated logs/results and exploratory split removed from public tracking; local copies preserved |
| Research | 206 pair decisions unresolved; canonical dataset, split, protocol and first run require actual researcher approval |
| Manuscript | Working draft with methods and references; results and conclusions pending experiments; no supervisor or clinical review claimed |
| Deployment | Local package/app checks passed previously; Docker and account-based public hosting still unverified |

No publication experiments or paper submission have occurred. A sole researcher can perform technical review and approve their study; published papers cannot supply case-specific decisions or clinical validation. The protocol amendment is proposed, not approved.

## Remaining scope

Core software is implemented; experimental completion is pending. The default grid has 45 classifier fits (five arms × three budgets × three seeds), including nine generator fits. No full-grid runtime estimate is available. Following technical dataset review and approval, remaining work is to execute that grid, inspect and interpret measured results, finish manuscript declarations/results/discussion, and configure public hosting. A universally best method, positive gains and clinical validation are not requirements for this experimental project. Diffusion is optional and is not added to the completion scope.

Historical files are retained locally under ignored `outputs/repository_archive/2026-11-10/` and in Git history. Local images, checkpoints, review decisions, `data/splits.json` and machine-specific files remain private. Cleanup does not rewrite Git history or alter the default branch.

Next required research action: complete [technical review](docs/HUMAN_REVIEW.md), inspect the canonical proposal and explicitly approve the exact [protocol](docs/EXPERIMENT_PROTOCOL.md). Public deployment also needs the account-owner hosting setup described in [deployment instructions](docs/DEPLOYMENT.md). Author identity, release license and final manuscript declarations remain researcher decisions.
