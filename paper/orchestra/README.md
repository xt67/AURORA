# Paper Orchestra integration for AURORA

Prepared 19 November 2026 for Rayan Rahman, Marathwada Mitra Mandal Institute of Technology.

Upstream: https://github.com/google-research/paper-orchestra
Pinned revision: `d9bae997e793a015ed38bb1605be9eb58c00284f` (Apache 2.0).
The upstream code is cloned locally under ignored `outputs/tools/paper-orchestra`; it is not vendored into AURORA.
An isolated Python environment is under `outputs/tools/paper-orchestra-env`.

## Current deliverables and truthful status

- `paper/manuscript.md`: active Codex-assisted v2 protocol/feasibility draft, with confirmed author/college.
- `paper/manuscript.html`: self-contained reading copy.
- `paper/manuscript.tex`: Pandoc-generated editable LaTeX, not yet compiled.
- `paper/history/manuscript_v1_2026-11-10.md`: unchanged historical draft.
- `template/`: neutral article template and scientific writing constraints, matching upstream's required `template.tex` / `guidelines.md` interface.
- `outputs/paper_orchestra/inputs_2026-11-19`: aggregate-only local input bundle in upstream's required `idea_sparse.md`, `experimental_log.md`, `figures/` format.

**No successful Paper Orchestra model run or compiled PDF is claimed.** Dependency installation, a supported model credential and `pdflatex`/`bibtex` must pass preflight. The upstream multi-agent pipeline sends writing materials to external services, can make many model/search calls and does not provide a hard dollar budget in its CLI. Only sanitized inputs are supplied. Raw radiographs, exact source IDs, private approvals, source PDFs and credentials are excluded.

## Setup and execution

From the AURORA root (PowerShell):

```powershell
# If installation needs to be resumed:
outputs/tools/paper-orchestra-env/Scripts/python.exe -m pip install --timeout 60 -r outputs/tools/paper-orchestra/requirements.txt

# Rebuild into a NEW directory if the manuscript or evidence changes:
C:/Users/onlys/anaconda3/envs/aurora-cuda/python.exe scripts/prepare_paper_orchestra.py --output outputs/paper_orchestra/inputs_revision_02

# Check only; makes no model calls:
outputs/tools/paper-orchestra-env/Scripts/python.exe scripts/run_paper_orchestra.py --inputs outputs/paper_orchestra/inputs_2026-11-19 --output outputs/paper_orchestra/generated_01

# Execute once tools and credentials are ready:
outputs/tools/paper-orchestra-env/Scripts/python.exe scripts/run_paper_orchestra.py --inputs outputs/paper_orchestra/inputs_2026-11-19 --output outputs/paper_orchestra/generated_01 --execute
```

Install a trusted TeX distribution providing `pdflatex` and `bibtex` on PATH; additional packages used by the generic template may be required. No LaTeX installation has been assumed from Pandoc availability. Set `GEMINI_API_KEY` in the local upstream `.env` file, or configure Vertex project/location and credentials. Do not commit or paste secrets into chat. `OPENAI_API_KEY` supports OpenAI writing models, but the inspected upstream citation-title parser still calls Gemini explicitly. The optional Semantic Scholar key can help literature API access. SMTP/email settings are not configured or used.

Defaults match the inspected upstream CLI (`gemini-3.1-pro-preview` for writing/reflection). Availability must be checked against the configured account; no model call has validated it here. Plotting is disabled and existing measured figures supplied. Research cutoff is explicitly November 2026. The wrapper refuses changed upstream revision, missing credentials/tools, changed bundle hashes and existing output directories. It does not impose a spend limit or guarantee generated claims.

## Evidence review before submission

The only actual experiment is the two-epoch, 52-image feasibility run. No classifier performance, FID/KID, privacy or clinical-validation claim is supported. The proposed 36-classifier/nine-GAN grid remains future work. Future AI output must be reviewed against the run artifacts and primary literature. The historical five-arm protocol must not silently replace v2's four conditions or its selected-positive added-count rule.

Remaining author details: email, department if desired, contributions, funding/conflicts and applicable data-use/ethics determination. No declaration is fabricated. A target venue and its formatting/disclosure requirements remain undecided. Before an effectiveness submission, complete the experiments, document initial similarity screening and review all citations. The working paper can currently report protocol and execution feasibility only.
