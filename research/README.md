# Research — Email-Link Publication Pipeline

This directory contains research analysis conducted in preparation for publishing Email-Link as an academic paper.

## Analysis Files

| File | Description |
|------|-------------|
| `analysis/executive-summary.md` | Publishability assessment — strengths, weaknesses, O-1 verdict |
| `analysis/introspection.md` | Deep code analysis — 607 lines covering every module in the pipeline |
| `analysis/reading-list.md` | 20 annotated papers — threat level, relevance tier, citation strategy |
| `analysis/gap-analysis.md` | Novelty claims ranked by strength + vulnerability scenarios with mitigations |
| `analysis/landscape-summary.md` | Competitive landscape map + venue strategy + action items |

## Key Findings

1. **The gap is real.** Zero papers combine email data + knowledge graph construction + TransE/KGE embeddings. The intersection is unoccupied.
2. **MAILEX (EMNLP 2023)** is the closest related work — email event extraction, but no KG, no embeddings.
3. **Evaluation needs work.** Link prediction tested on only 100 triples (~2% sample). Standard benchmarks test against ALL entities.
4. **Strongest publishable angle:** TransE predicting missing event participants from graph structure.
5. **Target venues:** arXiv preprint first → EMNLP 2026 workshop → CIKM 2026 → IEEE BigData 2026.

## Next Steps

- [ ] Expand link prediction evaluation to full entity set
- [ ] Add BM25/TF-IDF baseline comparison
- [ ] Address overclaims (LLM interface, ML classifiers)
- [ ] Write Related Work section (cite MAILEX + Guo 2020)
- [ ] Draft arXiv preprint

---

*Analysis by Commander Riker (research agent) — February 2026*
