# Email-Link: Annotated Reading List
**Compiled by:** Commander Riker  
**Date:** 2026-02-17  
**Phase:** Literature Scout (Phase 1 of Email-Link research pipeline)  
**Purpose:** Inform the Related Work section of the arXiv paper

---

> **Legend:**
> - ⚠️ **THREAT** — May challenge novelty claim. Read carefully before submission.
> - 🤝 **ALLY** — Validates our approach. Cite to support claims.
> - 📐 **FOUNDATION** — Must-cite foundational work.
> - 🔬 **PARALLEL** — Similar problem, different domain. Safe to reference.
> - 🧭 **CONTEXT** — Background/survey. Sets the landscape.

---

## TIER 1: Direct Competitors / Closest Related Work

---

### 1. ⚠️ MAILEX: Email Event and Argument Extraction
**Authors:** Saurabh Srivastava, Gaurav Singh, Shou Matsumoto, Ali Raz, Paulo Costa, Joshua Poore, Ziyu Yao  
**Year:** 2023  
**Venue:** EMNLP 2023 (Conference on Empirical Methods in Natural Language Processing)  
**Citations:** 7  
**URL:** https://arxiv.org/abs/2305.13469  

**What they did:** Introduced the first benchmark dataset for event extraction from conversational email threads. Proposed a taxonomy covering 10 event types and 76 argument roles in the email domain. Dataset contains 1.5K email threads and ~4K emails annotated with ~8K event instances. Compared three approaches: fine-tuned sequence labeling (BERT-based), fine-tuned generative extraction, and few-shot in-context learning.

**Key findings:** Email event extraction is harder than general event extraction due to non-continuous triggers, shared trigger spans across conversational turns, and non-named entity arguments. None of their three approaches achieves satisfying performance, leaving significant room for improvement.

**Relation to Email-Link:**  
⚠️ **THIS IS THE MOST IMPORTANT PAPER TO ENGAGE WITH.** It is the closest existing work to Email-Link. However, the threat is LIMITED: MAILEX uses purely sequence-based NLP models (BERT, T5, GPT-3) for event extraction — **it does not construct a Knowledge Graph, does not use TransE embeddings, and does not model relationships between extracted events**. Email-Link's contribution is orthogonal: we use NLP as a preprocessing step, then build a Knowledge Graph in Neo4j, and apply TransE embeddings to reason over the relational structure of email events. The key differentiator is that Email-Link captures **inter-event and entity-event relationships** through the graph structure, while MAILEX only identifies event spans in isolation. Our paper MUST cite MAILEX in Related Work and explicitly articulate this distinction.

**Action:** Read the full paper, particularly Section 4 (experiments) and Section 5 (analysis of failure modes). Their failure modes (non-continuous triggers, conversational context) are exactly what KG-based approaches can address.

---

### 2. 📐 Translating Embeddings for Modeling Multi-relational Data (TransE)
**Authors:** Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko  
**Year:** 2013  
**Venue:** NeurIPS 2013 (Neural Information Processing Systems)  
**Citations:** 14,000+  
**URL:** https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html  

**What they did:** Introduced TransE — the translation-based knowledge graph embedding method that models relationships as translations in vector space: **h + r ≈ t** (head entity + relation ≈ tail entity). Demonstrated effectiveness on benchmark KGs (Freebase, Wordnet). Simple, scalable, and interpretable compared to prior methods.

**Relation to Email-Link:**  
📐 **MUST CITE. This is the core embedding methodology of Email-Link.** The paper must explain why TransE was chosen for email event extraction: its interpretability (translation semantics map naturally to event participation relationships like "sender → wrote → email"), scalability to the size of personal email data, and effectiveness on sparse graphs typical of personal communication networks.

---

### 3. 📐 RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
**Authors:** Zhiqing Sun, Zhihong Deng, Jian-Yun Nie, Jian Tang  
**Year:** 2019 (ICLR 2019)  
**Venue:** International Conference on Learning Representations (ICLR)  
**Citations:** 2,564  
**URL:** https://arxiv.org/abs/1902.10197  

**What they did:** Proposed RotatE, which models each relation as a rotation from head to tail entity in complex vector space. Demonstrated superior performance over TransE on standard KG benchmarks (FB15k-237, WN18RR) by modeling symmetric, antisymmetric, inversion, and composition relation patterns that TransE cannot handle.

**Relation to Email-Link:**  
📐 **Must cite as a TransE variant comparison.** The paper should acknowledge RotatE as a more expressive alternative. Key argument: for email event extraction, the relationship types are relatively simple (participation, temporal ordering, communication) — TransE's linear translation semantics are sufficient and computationally cheaper at personal-data scale. RotatE would be a strong "future work" extension for asymmetric email relations (e.g., forward chains, reply chains).

---

### 4. 🔬 Knowledge Graph Enhanced Event Extraction in Financial Documents
**Authors:** Kaihao Guo, Tianpei Jiang, Haipeng Zhang  
**Year:** 2020  
**Venue:** IEEE International Conference on Big Data (BigData 2020)  
**Citations:** 12  
**URL:** https://arxiv.org/abs/2109.02592  

**What they did:** Proposed a framework that embeds a knowledge graph via Graph Neural Network and integrates the embedding with standard features for document-level event extraction. Applied to Chinese financial announcements. Outperformed state-of-the-art by 5.3% in F1-score specifically for multiple event extraction.

**Relation to Email-Link:**  
🤝 **KEY ALLY. This is the closest methodological parallel.** They do KG + embeddings for domain-specific document event extraction — exactly Email-Link's paradigm but in the financial domain. Key differences: (1) they use GNN over the KG, while Email-Link uses TransE; (2) their documents are financial announcements (structured, formal), while Email is informal/conversational; (3) they don't use Neo4j as a storage layer. Cite as validation that "KG-enhanced event extraction has shown gains in domain-specific document corpora."

---

## TIER 2: Core Methodology Validation

---

### 5. 🤝 Information Extraction Pipelines for Knowledge Graphs
**Authors:** M. Y. Jaradeh, Kuldeep Singh, M. Stocker, A. Both, S. Auer  
**Year:** 2023  
**Venue:** Knowledge and Information Systems (KAIS), Springer  
**Citations:** 25  
**URL:** https://doi.org/10.1007/s10115-022-01826-x  

**What they did:** Developed Plumber, a framework with 40 reusable components for KG completion subtasks (coreference resolution, entity linking, relation extraction). Generates 432 distinct IE pipelines dynamically using a transformer classifier. Evaluated on DBpedia, Wikidata, and Open Research KG.

**Relation to Email-Link:**  
🤝 **ALLY — validates the pipeline architecture.** Email-Link's NLP→KG→Embedding pipeline is consistent with the modular IE pipeline paradigm validated here. Cite to support the "pipeline approach to KG construction from text" claim.

---

### 6. 🤝 A Survey of Information Extraction Based on Deep Learning
**Authors:** Yang Yang, Zhilei Wu, Yuexiang Yang, Shuangshuang Lian, et al.  
**Year:** 2022  
**Venue:** Applied Sciences (MDPI)  
**Citations:** 90  
**URL:** https://doi.org/10.3390/app12199691  

**What they did:** Comprehensive survey of deep learning methods for information extraction covering entity recognition, relation extraction, and event extraction. Reviews transformer-based approaches, GNN-based methods, and their applications.

**Relation to Email-Link:**  
🧭 **Background citation for the Related Work section.** Sets the context for why moving from pure NLP to graph-structured representations is beneficial for capturing relational context in IE tasks.

---

### 7. 🤝 CollabKG: A Learnable Human-Machine-Cooperative Information Extraction Toolkit for (Event) Knowledge Graph Construction
**Authors:** Xiang Wei, Yufeng Chen, Ning Cheng, Xingyu Cui, et al.  
**Year:** 2023  
**Venue:** LREC 2023 (Language Resources and Evaluation Conference)  
**Citations:** 5  
**URL:** https://arxiv.org/abs/2307.00769  

**What they did:** Built an interactive annotation toolkit for constructing event-centric and entity-centric knowledge graphs from text with human-in-the-loop learning. Supports both supervised and active learning modes for improving extraction quality iteratively.

**Relation to Email-Link:**  
🤝 **ALLY — validates the event KG construction approach.** Shows that (event extraction → KG construction) is a recognized research direction. The human-in-the-loop aspect is relevant to Email-Link's potential for personal productivity applications where user feedback could refine the graph.

---

### 8. 🤝 Is a Large Language Model a Good Annotator for Event Extraction?
**Authors:** Ruirui Chen, Chengwei Qin, Weifeng Jiang, Dongkyu Choi  
**Year:** 2024  
**Venue:** AAAI 2024 (Association for the Advancement of Artificial Intelligence)  
**Citations:** 57  
**URL:** https://doi.org/10.1609/aaai.v38i16.29730  

**What they did:** Investigated whether GPT-4 can replace human annotators for event extraction. Found that while LLMs can generate reasonable event annotations, they fall short of human-level consistency on complex event types and argument linking. Suggests a hybrid human-LLM annotation pipeline is optimal.

**Relation to Email-Link:**  
🤝 **ALLY — validates the need for structured approaches.** The finding that pure LLM-based event extraction is unreliable for complex events supports Email-Link's graph-structured approach. LLMs excel at identifying event triggers but struggle with argument linking — precisely what a KG + embedding approach handles through relational modeling. Cite to argue that "LLM-based event extraction alone is insufficient for structured personal event understanding."

---

## TIER 3: Temporal Knowledge Graph Methods

---

### 9. 🤝 TeAST: Temporal Knowledge Graph Embedding via Archimedean Spiral Timeline
**Authors:** Jiang Li, Xiangdong Su, Guanglai Gao  
**Year:** 2023  
**Venue:** ACL 2023 (Annual Meeting of the Association for Computational Linguistics)  
**Citations:** 36  
**URL:** https://doi.org/10.18653/v1/2023.acl-long.862  

**What they did:** Proposed a temporal KG embedding model using Archimedean spirals to model the progression of time in knowledge graphs. Entities are placed on spirals whose position reflects their temporal characteristics. Outperforms previous temporal KGE methods on standard benchmarks.

**Relation to Email-Link:**  
🧭 **Relevant context for the "temporal" dimension of email data.** Email events are inherently temporal (meetings, deadlines, follow-ups). TeAST shows that adding explicit temporal structure to KG embeddings improves event reasoning. Email-Link's use of static TransE misses this temporal signal — a limitation to acknowledge and a direction for future work. Cite as: "Recent work on temporal KG embeddings (TeAST, 2023) suggests that incorporating explicit timeline modeling could further improve event extraction from time-sensitive communication data."

---

### 10. 🤝 TARGAT: A Time-Aware Relational Graph Attention Model for Temporal Knowledge Graph Embedding
**Authors:** Zhiwen Xie, Runjie Zhu, Jin Liu, et al.  
**Year:** 2023  
**Venue:** IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)  
**Citations:** 25  
**URL:** https://doi.org/10.1109/TASLP.2023.3282101  

**What they did:** Proposed a time-aware relational graph attention model for temporal KG embedding that captures evolving entity representations over time via attention mechanisms. Particularly effective on knowledge graphs where entity attributes change over time.

**Relation to Email-Link:**  
🧭 **Relevant context for future directions.** Like TeAST, this validates that temporal awareness in KG embeddings is a productive research direction for communication data. Email-Link's current TransE approach is static; TARGAT-style temporal attention is a natural next step.

---

### 11. 🧭 A Survey on Temporal Knowledge Graph Embedding: Models and Applications
**Authors:** Yuchao Zhang, Xiangjie Kong, Zhehui Shen, et al.  
**Year:** 2024  
**Venue:** Knowledge-Based Systems (Elsevier)  
**Citations:** 19  
**URL:** https://doi.org/10.1016/j.knosys.2024.112454  

**What they did:** Comprehensive survey of temporal knowledge graph embedding models, covering evolution from static methods (TransE, RotatE) to time-aware models. Reviews applications in QA, event prediction, and recommendation.

**Relation to Email-Link:**  
🧭 **BACKGROUND. Cite in Related Work to contextualize Email-Link's use of static TransE within the broader TKG landscape.** Can frame Email-Link's TransE choice as appropriate for "extracting event structure from personal email archives as a time-static snapshot" while acknowledging temporal KGE as future work.

---

## TIER 4: Competing Paradigms (LLM-First, GraphRAG)

---

### 12. ⚠️ Event Extraction in Large Language Model
**Authors:** Bobo Li, Xudong Han, Jiang Liu, Yuzhe Ding, et al.  
**Year:** 2025  
**Venue:** arXiv preprint  
**Citations:** 1  
**URL:** https://arxiv.org/abs/2512.19537  

**What they did:** Survey of LLM-based approaches to event extraction. Shows that prompting LLMs (GPT-4, LLaMA) can produce structured event outputs in zero/few-shot settings without task-specific training. Reviews the shift from fine-tuned sequence labeling to prompt-based extraction.

**Relation to Email-Link:**  
⚠️ **MODERATE THREAT. Reviewers may ask "why not just use GPT-4?"** Preemptive response: (1) LLM-based extraction lacks the structured relational representation that KG + embeddings provide — an LLM can extract events but cannot model event-entity RELATIONSHIPS across emails without explicit graph structure; (2) KG + TransE provides interpretable embeddings usable downstream for link prediction, event clustering, and recommendation — pure LLM extraction does not; (3) Privacy: processing personal email through commercial LLMs raises privacy concerns that a local Neo4j + embedding pipeline avoids.

---

### 13. ⚠️ Graph Retrieval-Augmented Generation: A Survey
**Authors:** Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, et al.  
**Year:** 2024  
**Venue:** ACM Transactions on Information Systems  
**Citations:** 298  
**URL:** https://arxiv.org/abs/2408.08921  

**What they did:** Comprehensive survey of GraphRAG approaches that combine knowledge graphs with vector retrieval and LLMs for generation tasks. Reviews graph construction, indexing, and retrieval strategies. Covers 100+ papers across QA, summarization, and information extraction.

**Relation to Email-Link:**  
⚠️ **COMPETING PARADIGM. GraphRAG (MS GraphRAG, 2024) represents the emerging mainstream approach for KG-based information retrieval.** However, Email-Link was designed before this wave and represents a different philosophy: structured embedding-based event representation vs. RAG-based text generation. Frame Email-Link as a "structured KG extraction" approach that feeds INTO GraphRAG-style systems — our KG is the foundation, not the endpoint. This actually positions Email-Link well: we build the KG, GraphRAG systems consume it.

---

### 14. 🔬 HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction
**Authors:** Bhaskarjit Sarmah, et al.  
**Year:** 2024  
**Venue:** International Conference on AI in Finance (ICAIF 2024)  
**Citations:** 144  
**URL:** https://arxiv.org/abs/2408.04948  

**What they did:** Proposed HybridRAG which combines KG-based retrieval with vector-similarity retrieval for financial document QA. Shows that KG + vector retrieval outperforms either method alone. Applied to financial earnings call transcripts.

**Relation to Email-Link:**  
🔬 **PARALLEL DOMAIN, validates KG+embedding for structured documents.** They show that KG structure captures relations that vector similarity misses. Email-Link operates on email data (vs. financial transcripts), uses TransE embeddings (vs. vector retrieval) — different implementations of the same core insight. Cite as: "Concurrent work combining KG structure with embeddings for domain-specific documents has shown strong results (HybridRAG, 2024)."

---

## TIER 5: Email-Adjacent Work

---

### 15. 🔬 Focus on the Action: Learning to Highlight and Summarize Jointly for Email To-Do Items Summarization
**Authors:** Kexun Zhang, Jiaao Chen, Diyi Yang  
**Year:** 2022  
**Venue:** ACL Findings 2022  
**Citations:** 4  
**URL:** https://doi.org/10.18653/v1/2022.findings-acl.323  

**What they did:** Proposed a joint learning approach for email to-do item generation that simultaneously highlights relevant email segments and generates to-do summaries. Framed as email action understanding — identifying what the recipient needs to do.

**Relation to Email-Link:**  
🔬 **ADJACENT. Shares the email event understanding goal.** They focus on to-do items (action events) from individual emails using sequence models; Email-Link extracts a broader range of events and models relationships between them via KG. Not a direct competitor. Cite to show the importance of email event understanding in the NLP community.

---

### 16. 🔬 Document-Level Event Extraction — A Survey of Methods and Applications
**Authors:** Qi Liu, Zhen Luan, Kunlong Wang, Ye Zou, et al.  
**Year:** 2023  
**Venue:** Journal of Physics: Conference Series  
**Citations:** 7  
**URL:** https://doi.org/10.1088/1742-6596/2504/1/012008  

**What they did:** Survey of document-level event extraction methods, covering cross-sentence, cross-document, and argument-role identification. Distinguishes sentence-level event extraction (most benchmarks) from document-level extraction (more realistic, harder).

**Relation to Email-Link:**  
🧭 **Background context. Email is a document-level extraction task** — events span multiple messages, conversation turns, and time periods. Cite to position Email-Link as document-level event extraction applied to the email domain.

---

## TIER 6: Personal Knowledge Graph Construction

---

### 17. 🤝 PKGRec: Personal Knowledge Graph Construction and Mining for Federated Recommendation Enhancement
**Authors:** Haochen Yuan, Yang Zhang, Quan Z. Sheng, Lina Yao, et al.  
**Year:** 2025  
**Venue:** CIKM 2025 (International Conference on Information and Knowledge Management)  
**Citations:** 1  
**URL:** https://doi.org/10.1145/3746252.3761301  

**What they did:** Framework for constructing and mining Personal Knowledge Graphs (PKGs) from individual user data for federated recommendation. Shows that user-specific KGs outperform generic KGs for personalized tasks. Addresses privacy via federated architecture.

**Relation to Email-Link:**  
🤝 **ALLY — validates the "personal KG" paradigm.** Email-Link is fundamentally a personal knowledge graph system — it builds a KG from a user's email data to extract their events and relationships. PKGRec validates that personal KGs are a real and growing research area. Cite to position Email-Link within the personal AI / personal KG literature.

---

### 18. 🤝 PROM: Personal Knowledge Graph Construction with Large Language Models
**Authors:** Aishan Maoliniyazi, Chaohong Ma, Xiaofeng Meng, Bingbing Xu  
**Year:** 2025  
**Venue:** IC3K 2025 (International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management)  
**Citations:** 0  
**URL:** https://doi.org/10.5220/0013830900004000  

**What they did:** Proposed an LLM-powered framework for constructing personal knowledge graphs from user notes, documents, and communications. Uses GPT-4 for entity/relation extraction and Neo4j for storage (same stack as Email-Link).

**Relation to Email-Link:**  
⚠️ **MODERATE FLAG. Same stack (LLM + Neo4j), overlapping goal (personal KG construction), but published after Email-Link's codebase.** Key differentiators: Email-Link uses TransE embeddings for relational reasoning (PROM does not use KG embeddings), and Email-Link targets email specifically as a domain rather than general personal documents. This paper validates our stack choice (Neo4j) but does not preempt our contribution (the TransE embedding layer for event-specific reasoning).

---

### 19. 🔬 SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations
**Authors:** Md. Saidul Hoque Anik, Ariful Azad  
**Year:** 2025  
**Venue:** Conference on Machine Learning and Systems (MLSys 2025)  
**Citations:** 3  
**URL:** https://arxiv.org/abs/2502.16949  

**What they did:** Proposed efficient sparse matrix implementations for training TransE-family models (TransE, TransR, TransH). Achieves significant speedups (3-10x) on large knowledge graphs without accuracy loss. Demonstrates TransE remains competitive with more complex embeddings at scale.

**Relation to Email-Link:**  
🤝 **ALLY — shows TransE family is still actively developed and competitive.** Critics might dismiss TransE as "outdated" — this 2025 paper at a top venue shows the translation-based embedding family is alive and being optimized. Strengthens the case for Email-Link's use of TransE.

---

### 20. 🔬 Biomedical Event Extraction on Graph Edge-conditioned Attention Networks with Hierarchical Knowledge Graphs
**Authors:** Kung-Hsiang Huang, Mu Yang, Nanyun Peng  
**Year:** 2020  
**Venue:** EMNLP Findings 2020  
**Citations:** 30  
**URL:** https://arxiv.org/abs/2009.09335  

**What they did:** Incorporated domain knowledge from UMLS (medical ontology) into event extraction via a graph edge-conditioned attention network. Showed significant F1 improvements for biomedical event extraction by modeling domain-specific entity relationships.

**Relation to Email-Link:**  
🔬 **PARALLEL DOMAIN. Same core idea (domain-specific KG → improved event extraction) applied to biomedical text.** Shows that domain-specific knowledge graph injection is a proven technique for hard event extraction tasks. Email-Link applies the same principle to the personal email domain. Cite as a strong precedent for "domain-specific KG-enhanced event extraction."

---

## Honorable Mentions (Relevant but Lower Priority)

| Paper | Year | Why it Matters |
|-------|------|----------------|
| Action-Item-Driven Summarization of Long Meeting Transcripts (Golia & Kalita, 2023) | 2023 | Adjacent task: action item extraction from conversational data |
| HyperComplEx (Gajjar et al., 2025) | 2025 | Advanced KG embedding combining multiple spaces — future work direction |
| Knowledge Graph Reasoning with Logics and Embeddings: Survey (Zhang et al., 2022) | 2022 | Background survey on KG reasoning methods |
| Document-Level Event Extraction Guided by Event Schemas and Knowledge Graphs (Sun et al., 2025) | 2025 | Recent parallel: schema-guided KG event extraction |
| CompoundE: KGE with Translation, Rotation and Scaling (Ge et al., 2022) | 2022 | Extended TransE with multiple geometric operations |

---

## Quick Reference Table (Top 20)

| Rank | Paper | Year | Venue | Relation | Priority |
|------|-------|------|-------|----------|----------|
| 1 | MAILEX: Email Event and Argument Extraction | 2023 | EMNLP | ⚠️ THREAT (manageable) | **MUST READ** |
| 2 | TransE: Translating Embeddings for Multi-relational Data | 2013 | NeurIPS | 📐 Foundation | **MUST CITE** |
| 3 | RotatE: KGE by Relational Rotation | 2019 | ICLR | 📐 Foundation | **MUST CITE** |
| 4 | KG-Enhanced Event Extraction in Financial Docs | 2020 | IEEE BigData | 🤝 Ally (Closest Parallel) | **MUST READ** |
| 5 | Information Extraction Pipelines for KGs | 2023 | KAIS | 🤝 Ally | High |
| 6 | Survey of IE Based on Deep Learning | 2022 | Applied Sci | 🧭 Context | High |
| 7 | CollabKG: IE Toolkit for Event KG Construction | 2023 | LREC | 🤝 Ally | Medium |
| 8 | Is LLM a Good Annotator for Event Extraction? | 2024 | AAAI | 🤝 Ally (vs LLM-only) | High |
| 9 | TeAST: Temporal KGE via Spiral Timeline | 2023 | ACL | 🧭 Context/Future Work | Medium |
| 10 | TARGAT: Time-Aware Relational Graph Attention | 2023 | TASLP | 🧭 Context/Future Work | Medium |
| 11 | Survey on Temporal KGE | 2024 | KBS | 🧭 Context | Medium |
| 12 | Event Extraction in LLM | 2025 | arXiv | ⚠️ Competing Paradigm | **READ** |
| 13 | GraphRAG Survey | 2024 | ACM TOIS | ⚠️ Competing Paradigm | **READ** |
| 14 | HybridRAG: KG + Vector RAG for Finance | 2024 | ICAIF | 🔬 Parallel | Medium |
| 15 | Focus on Action: Email To-Do Summarization | 2022 | ACL Findings | 🔬 Adjacent | Medium |
| 16 | Document-Level Event Extraction Survey | 2023 | JPC | 🧭 Context | Low-Med |
| 17 | PKGRec: Personal KG for Recommendation | 2025 | CIKM | 🤝 Ally (Personal KG) | Medium |
| 18 | PROM: Personal KG with LLM | 2025 | IC3K | ⚠️ Flag (Same Stack) | **READ** |
| 19 | SparseTransX: Efficient Translation KGE | 2025 | MLSys | 🤝 Ally (TransE viability) | Low-Med |
| 20 | Biomedical Event Extraction + KG | 2020 | EMNLP Findings | 🔬 Parallel Domain | Medium |

---

## Search Methodology Notes

- **Databases searched:** Semantic Scholar API, arXiv search, direct arXiv abstracts
- **Search queries (20+ executed):** knowledge graph embeddings information extraction, email event extraction NLP, TransE/RotatE/ComplEx comparison, temporal knowledge graph embedding, email knowledge graph construction, personal knowledge graph, GraphRAG survey, LLM event extraction, email to-do action item detection, document-level event extraction survey, event schema induction, email NLP benchmark
- **Date range:** Primary focus 2022-2026; foundational papers included without date filter
- **Limitation:** Semantic Scholar rate-limited; some queries timed out. A second pass with full API access or Google Scholar would likely yield additional relevant papers in 2024-2025.
- **Known gap:** Chinese-language NLP conferences (ACL China, NLPCC) may have email/communication event extraction papers not fully indexed in English-language sources. Manual search recommended.

---

*Filed by Commander Riker — O-1 Strategy Division*  
*Next action: Read MAILEX full text + the Guo 2020 financial paper. Those two define our position in the literature.*
