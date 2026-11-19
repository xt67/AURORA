# Draft evidence review — 19 November 2026

This is a local editorial check, not independent scientific or clinical review.

| Claim | Evidence | Allowed interpretation |
|---|---|---|
| 800 original / 590 retained / 210 excluded | Dataset selection report and frozen manifests | Images, not unique patients; exclusions not confirmed duplicates |
| 413/89/88 split | Split JSON and cohort/class report | Image-level stratification; no patient independence claim |
| 52 sources, two epochs, seed 42 | Local feasibility result and checkpoint/config checks | One bounded execution, not converged training |
| 1.32 s trainer, 6.19 s total | `outputs/feasibility_v2/run_01/feasibility_result.json` | Single-run recorded intervals; excludes interpreter/import startup from total |
| 216.92 MiB allocation | CUDA allocator peak in result | Not total GPU/process footprint |
| Finite epoch losses | `gan/g_losses.npy` / `gan/d_losses.npy` | Not fidelity, diversity or clinical quality |
| 39 software tests passed | Recorded focused test result in feasibility documentation | Stubbed orchestration tests, separate from the actual run |
| Four conditions / 36 classifier fits | Simplified v2 proposal | Planned; no classifier outcomes exist |

Primary literature records were checked for NLM dataset publication (PMC4256233), DCGAN (arXiv:1511.06434), Morís et al. (DOI 10.1007/s11517-024-03056-5 / PMID 38499946), Pink and Sykes (PMLR 318), and Folguera-Profitós et al. (JMIR AI e81472). No other paper's numerical performance is presented as an AURORA result. No exhaustive novelty review is claimed.

The historical manuscript is retained unchanged under `paper/history/`. Historical protocols and experiment artifacts were not rewritten. Initial candidate-screening thresholds/algorithm documentation, completed publication experiments, image-quality analysis, author declarations and venue selection remain outstanding.

Paper Orchestra CLI help was exercised without model calls. The local launcher rejects missing dependencies/LaTeX tools and requires an explicit execution flag. The input bundle uses an explicit aggregate-field allowlist; hashes were checked. No source IDs or approval identity fields are in its experimental log. Input bundles and external-model outputs remain ignored; only the reviewed manuscript and reproducible tooling are public.
