# AURORA research manuscript

The current [PDF](publication_v2/manuscript.pdf) reports the completed four-condition comparison: 36 classifiers and nine GANs. Its [editable source](publication_v2/manuscript.md), [aggregate results](publication_v2/paper_results.json) and [bibliography](publication_v2/references.bib) are included.

The generator samples were noise-like rather than convincing radiographs. The paper reports failed synthesis and classification trade-offs; it does not claim successful disease-preserving generation. Faculty review, data-use/ethics determinations and venue submission remain separate.

The earlier feasibility-only manuscript is preserved in `professor_review/`. Do not send that earlier version as the completed comparison. A local editable Word copy of the new manuscript is also prepared; `.docx` files remain ignored by Git.

The new source was assembled from measured artifacts using `scripts/build_publication_manuscript.py`. Render with Pandoc and the existing bibliography; inspect the pages before distribution. [Paper Orchestra integration](orchestra/README.md) documents the earlier drafting assistance; no full literature/refinement pipeline is claimed.
