# AURORA / MediSyn current completion status

Updated 10 November 2026. See PROJECT_COMPLETION_TRACKER.md at the repository root for current execution status and docs/TEST_REPORT.md for verification evidence.

The software is a locally tested release candidate. The research study is not complete, deployed publicly or ready for paper submission.

## Completed independent work

- Repaired synthesis, checkpoint/inference, PNG export, classifier batching, metrics and honest demo behavior.
- Implemented reviewed canonical proposals, duplicate-aware cohort/label splitting, image hashes and nested training budgets.
- Implemented an explicit publication approval gate and fixed-update A0/A1/A2/A3/G1 experiment runner with validation-based selection, subgroup evaluation and provenance.
- Implemented analysis, grouped confidence intervals, paired comparisons, calibration, pixel proximity/diversity and optional FID/KID diagnostics.
- Strengthened the existing local human-review interface without creating or approving any real decisions.
- Prepared a manuscript working draft, architecture/protocol/data/model documentation, packaging, deployment candidate and software citation.
- Final regression suite: 72 passed in 24.90s. Local package build/import, dependency check and actual Streamlit HTTP health check passed.

- Completed the measured-results exporter for paper tables and demo JSON, with complete-grid/provenance checks and omission of private row IDs/local paths.
- Updated scope throughout: comparative experimental results, including negative outcomes; no universal superiority or clinical validation requirement.

## Remaining human/external gates

1. Complete the 206-pair review and investigate conflicting labels. No reviewer decisions exist yet.
2. Inspect/approve the resulting canonical dataset, frozen split and exact protocol/configuration. Explicitly approve the first publication run.
3. Run the approved multi-seed experiments and inspect genuine results, including failures and negative outcomes. No publication experiments have been launched.
4. Complete manuscript results, discussion, conclusion, authorship and declarations; disclose sole-researcher review and absence of clinical validation. Submission requires explicit instruction.
5. Select a software license and establish data/checkpoint release permissions. No raw images or private reviewer files are included in software commits.
6. Validate Docker/container behavior on a machine with Docker, and perform account-based hosting setup. No public deployment URL exists. The local demo can run without a model and reports that state honestly.

Diffusion and cross-cohort sensitivity training are optional, unimplemented extensions requiring a protocol amendment. They are not claimed as completed. Full Inception metric extraction has not been exercised on real data; software-level metric/export checks do not establish synthesis quality.

## Current documentation

Use the executable protocol, NLM dataset card, literature validation notes and manuscript. Historical planning documents and mixed-source smoke results were removed from the public branch. The local archive and Git history preserve them; they are not primary evidence. A supervisor is not a software prerequisite, and papers do not replace actual dataset review.
