# AURORA research manuscript

The current version is the [faculty-review PDF](professor_review/manuscript.pdf).

- [Editable Markdown source](professor_review/manuscript.md)
- [Review notes and evidence boundary](professor_review/REVIEW_NOTES.md)
- [Bibliography](orchestra/template/references.bib)
- [Paper Orchestra integration](orchestra/README.md)

The report covers dataset preparation and a completed two-epoch DCGAN feasibility run. Classifier comparisons remain future work. Superseded drafts and intermediate exports were removed from the current tree; they remain recoverable from Git history and a local ignored archive.

The Markdown source, bibliography and stylesheet are sufficient to regenerate a reading copy with Pandoc:

```powershell
pandoc paper/professor_review/manuscript.md --standalone --citeproc --bibliography=paper/orchestra/template/references.bib --resource-path=paper/professor_review --css=paper/professor_review/style.css --embed-resources -o outputs/manuscript.html
```

Print the generated HTML to A4 PDF and inspect pagination before replacing the published PDF. No model calls or new training are needed to render the current source.
