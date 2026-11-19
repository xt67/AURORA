# Paper Orchestra integration

Upstream: https://github.com/google-research/paper-orchestra

Reviewed revision: `d9bae997e793a015ed38bb1605be9eb58c00284f` (Apache 2.0).

AURORA used the upstream outline and section-writing prompts with Gemini Flash through a standard-library REST adapter. Curated references replaced automated literature search. The resulting text was checked and edited against the recorded experiment evidence. The full unmodified pipeline and automated refinement stage were not completed. The current deliverable is the [faculty-review paper](../professor_review/manuscript.pdf), not an unreviewed model response.

## Reusable components

- `template/`: neutral LaTeX article template, writing constraints and source bibliography.
- `scripts/prepare_paper_orchestra.py`: creates a new aggregate-only input bundle from the current manuscript and local feasibility artifacts. No raw medical images, exact source IDs or private approvals are exported.
- `scripts/run_paper_orchestra.py`: preflight and optional invocation of the pinned upstream CLI. It checks tools, credentials, input hashes and a new output location before execution.

The upstream clone, Python environments, API responses and generated input bundles are local under ignored `outputs/`. Credentials stay in the ignored upstream `.env` file. Nothing in this folder grants training approval or sends email.

## Commands from the repository root

```powershell
# Requires the local measured feasibility artifacts; choose a new output directory.
python scripts/prepare_paper_orchestra.py --output outputs/paper_orchestra/inputs_new

# Use an environment with the upstream requirements installed.
# Requires pdflatex and bibtex on PATH; no model call without --execute.
python scripts/run_paper_orchestra.py --inputs outputs/paper_orchestra/inputs_new --output outputs/paper_orchestra/generated_new --writer-model gemini-3-flash-preview --reflection-model gemini-3-flash-preview
```

The configured key accepted a minimal Flash request; the Pro model returned zero free-tier quota. This does not guarantee sustained pipeline quota. Upstream citation parsing also uses Gemini. The CLI has no hard spend cap. Only add `--execute` when intentionally starting external model calls. Keep unmeasured classifier outcomes, clinical claims and missing declarations out of generated text.
