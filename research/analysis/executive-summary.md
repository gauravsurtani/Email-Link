# Email-Link: Executive Summary
**Prepared by:** Commander Riker  
**Date:** 2026-02-17  
**For:** Captain Gaurav Surtani — O-1 evidence assessment

---

## What Was Built

Email-Link is a working end-to-end system that transforms Gmail Takeout MBOX files into a Neo4j knowledge graph, extracts structured events from emails, trains TransE embeddings on the resulting graph, and exposes query interfaces for event discovery and link prediction.

**Evaluated on:** 2,750 emails → 21,272 nodes, 59,085 relationships, 312 events.

**Core result numbers:**
- Event extraction: Precision 83.7%, Recall 76.4%, F1 79.9% (on 200-email manual annotation)
- TransE triple classification: F1 = 0.82 (on 14,626 triples)
- TransE link prediction: MRR = 0.564, Hits@10 = 0.65 (on 100 triples — too small)
- ROC-AUC = 0.87

---

## What's Real vs. What's Overstated

### Real and Working
- MBOX parsing, Neo4j loading, schema design — solid
- Event extraction pipeline (rule-based regex + spaCy) — works, has real precision/recall
- TransE PyTorch implementation — correct, standard, trains successfully
- Evaluation metrics — real numbers from actual evaluation code
- Dual architecture: fast rule-based events + expensive LLM semantic extraction

### Overstated in the Paper
- **"LLM-powered agent"** — the agent is pure `if/elif` keyword routing, no LLM involved
- **"Parallel entity extraction"** — not implemented anywhere in code
- **"Embedding compression"** — not implemented
- Link prediction evaluated on only **100 triples** from a graph with 59,085 relationships — this is inadequate for a publishable claim

---

## Strongest Publishable Claim

**TransE embeddings enable semantic event retrieval that keyword search cannot.**

The `similar <event_id>` feature uses learned embeddings to find semantically related events by cosine similarity. A "budget review" search returns "financial planning" and "quarterly review" events — events that share no keywords. This is the core contribution. The problem: it's currently supported by a single anecdote in the paper. To publish, you need a rigorous comparative evaluation against BM25 or TF-IDF on a held-out test set.

**Second-best claim:** Participant discovery via link prediction (finding missing attendees). MRR=0.564 is a real result — needs to be run on more than 100 samples and compared against a baseline to be publishable.

---

## Weakest Points (Honest)

1. **Event type classifier misfires on job alert emails.** The word "job" triggers "interview" type. LinkedIn digest emails get labeled as interview events. The README inadvertently shows this bug in its example output.

2. **No baseline comparison anywhere.** No TF-IDF, no keyword search, no random baseline. Every evaluation number is absolute, not relative.

3. **The "agent" has no AI.** If the paper calls this an agentic system or natural language interface, that claim needs to be backed up with an actual LLM integration.

4. **CREATE vs. MERGE bug in loader.** Re-running semantic extraction creates duplicate Amount, Date, and SemanticData nodes in Neo4j. Minor but breaks reproducibility.

5. **Zero test coverage.** No unit or integration tests. This matters for a system you'd publish and want others to use.

---

## What's Needed to Make This Publishable

**Minimum viable additions (2-3 weeks of work):**

1. **Expand link prediction evaluation to full test set** (~11,800 triples, not 100) and implement filtered ranking protocol. This is running the existing code with corrected parameters — not new code.

2. **Add a baseline comparison for event retrieval.** Implement TF-IDF similarity on event subjects/bodies. Show that TransE embedding similarity outperforms it (or quantify where it does and doesn't).

3. **Either remove the "agent" claim OR build a real one.** Easiest path: integrate one LLM call that translates a natural language query to Cypher. Even GPT-3.5 with a simple prompt would be legitimate.

4. **Fix the event type misclassification** — "interview" should not trigger on job listing emails. Add a negative signal for "apply for" / "job listings" patterns.

5. **Write the comparison section.** Show this approach vs. traditional calendar extraction (Exchange/Gmail's built-in calendar detection) vs. pure keyword search.

---

## Assessment for O-1 Evidence

**As-is:** Course project with real working code and real (if limited) evaluation. Not publishable in current state.

**With 3 weeks of targeted work:** Could become a legitimate arxiv preprint or a workshop paper at EMNLP/AAAI. The system is real, the concept is valid, the numbers need strengthening.

**O-1 framing:** "Built an end-to-end system combining knowledge graphs and KGE for email event understanding, with published evaluation results" — this holds if you fix the evaluation scale and add baselines. The TransE link prediction claim (MRR=0.564 on email graphs) is original work in a domain where little published work exists.

**The core idea is sound.** Nobody has published TransE applied specifically to email communication graphs for event understanding and participant discovery. That's the IP. The code needs tightening; the idea doesn't.

---

*Commander Riker | email-link executive summary | 2026-02-17*
