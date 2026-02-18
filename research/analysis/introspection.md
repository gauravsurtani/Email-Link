# Email-Link Codebase — Deep Introspection Report
**Inspector:** Commander Riker  
**Date:** 2026-02-17  
**Repo:** https://github.com/gauravsurtani/Email-Link  
**Commit state:** As of 2026-02-17 (repo exists on disk at /tmp/email-link-introspect/)

---

## PREFACE: What This Actually Is

This is a graduate class project (SJSU CMPE/CS, based on "Rubric Project.pdf" in repo) co-authored by Gaurav Surtani and Somesh Bagadiya. The paper is titled *"Leveraging Knowledge Graphs and TransE Embeddings for Event Extraction from Email Data."* It is a real, working system with real evaluation metrics, but it is a **course project, not a published paper** yet. The Captain needs to know what's here before deciding whether to develop this into something submittable.

Total code: ~7,244 lines across 33 Python files. There are **zero test files**.

---

## a. ARCHITECTURE MAP

Full data flow:

```
Gmail Takeout MBOX file
        │
        ▼
[email_parser/mbox_parser.py]
  - Python mailbox.mbox()
  - Extracts: message_id, thread_id, subject, date, labels,
    from/to/cc/bcc (as {name, email} dicts), body (plain text preferred),
    attachments (filename, content_type, size), references/in_reply_to
  - Outputs: List of email dicts OR saves to JSON file
        │
        ▼
[email_parser/email_extractor.py]
  - decode_str(): handles encoded headers (RFC 2047)
  - extract_email_addresses(): getaddresses() → {name, email} dicts
  - get_email_body(): walks multipart, prefers text/plain over HTML
  - extract_attachments(): metadata only (no content stored)
  - extract_references(): regex on References/In-Reply-To headers
        │
        ├─── (Optional) ──────────────────────────────────────────────────┐
        │   [analysis/semantic_extractor.py — SemanticExtractor]          │
        │     - Sends subject + body to GPT-3.5-turbo                     │
        │     - Prompt extracts: type, organizations, people, products,   │
        │       locations, amounts, dates, actions                         │
        │     - Falls back to spaCy en_core_web_sm NER on API failure     │
        │     - Saves enriched emails to JSON batches                     │
        └────────────────────────────────────────────────────────────────-┘
        │
        ▼
[graph_db/loader.py — EmailGraphLoader]
  - Connects to Neo4j (bolt://localhost:7687)
  - Creates constraints & indexes (schema.py)
  - _create_email_nodes(): MERGE Email, Thread, Person nodes
    Creates: SENT, RECEIVED, COPIED, BCC, BELONGS_TO, REPLY_TO,
    HAS_LABEL, HAS_ATTACHMENT relationships
  - _create_person_relationships(): creates Domain nodes, BELONGS_TO
  - _create_semantic_nodes(): creates SemanticData + entity nodes
    (Organization, Location, Product, Action, Amount, Date)
        │
        ▼
[Neo4j Knowledge Graph]
  Scale: 21,272 nodes, 59,085 relationships (on 2,750 email dataset)
        │
        ├─── (Event Extraction Pipeline) ─────────────────────────────────┐
        │   [event_extraction/extract_events.py — EventExtractor]         │
        │     - Rule-based: keyword matching + regex on body              │
        │     - _is_likely_event_email(): checks event keywords + phrases │
        │     - _identify_event_type(): keyword counting per category     │
        │     - _extract_datetime_info(): dateparser + regex patterns     │
        │     - _extract_location(): regex for "location:", rooms, addrs  │
        │     - _extract_virtual_meeting_info(): Zoom/Teams URL extraction│
        │     - _extract_attendees(): spaCy PERSON NER + email regex      │
        │   Outputs: List of event dicts per email                        │
        │                                                                 │
        │   [event_extraction/event_to_graph.py — EventGraphEnhancer]    │
        │     - Creates Event nodes in Neo4j                              │
        │     - Connects: Email-[:CONTAINS_EVENT]->Event                  │
        │     - Event-[:LOCATED_AT]->Location                             │
        │     - Person-[:ATTENDS]->Event (with role property)             │
        │     - Person-[:ORGANIZES]->Event                                │
        └────────────────────────────────────────────────────────────────-┘
        │
        ▼
[embeddings/graph_embeddings.py — GraphEmbeddings + TransE]
  - extract_triples(): Cypher query to pull all (source, relation, target)
    triples from Neo4j for node types: Person, Email, Domain, Thread,
    Label, Attachment, Event, Location
  - Entity IDs formatted as "NodeType_identifier"
    (e.g., "Person_gaurav@example.com", "Event_abc123_1234567890")
  - train_transe(): TransE model (PyTorch), dim=100, epochs=50-100,
    batch=128, lr=0.001, margin=1.0, Adam optimizer
  - Saves: entity_embeddings.csv, relation_embeddings.csv,
    entity_mapping.csv, triples.json, entities.json, relations.json
        │
        ├─── Evaluation ──────────────────────────────────────────────────┐
        │   [evaluate_embeddings.py]                                      │
        │     - Triple classification (0.81 F1 on 14,626 triples)         │
        │     - Link prediction (MRR=0.564, Hits@10=0.65 on 100 triples) │
        │     - ROC-AUC: 0.87                                              │
        │     - Generates: confusion matrix, ROC curve, score dist plots  │
        └────────────────────────────────────────────────────────────────-┘
        │
        ▼
[agent/agent.py — EmailGraphAgent]  ← keyword routing, NOT LLM
[event_query.py — EventQuery]       ← Cypher-based query CLI
[analysis/queries.py — EmailGraphQueries] ← Pre-built Cypher queries
```

Entry points:
- `main.py` → parse + load pipeline
- `event_pipeline.py` → event extraction + embeddings
- `event_query.py` → query CLI
- `evaluate_embeddings.py` → evaluation suite
- `extract_semantic_data.py` → standalone semantic extraction
- `analyze_semantic_data.py` → semantic data analysis

---

## b. KNOWLEDGE GRAPH SCHEMA

### Node Types

| Node Label | Primary Key | Key Properties | Purpose |
|---|---|---|---|
| Person | email | name, type | Email sender/recipient + mentioned people |
| Email | message_id | subject, date, body, has_attachments | Core email node |
| Thread | thread_id | — | Gmail thread grouping |
| Domain | name | — | Email domain (@company.com) |
| Label | name | — | Gmail labels (Inbox, Sent, etc.) |
| Attachment | (none) | filename, content_type, size | File metadata |
| SemanticData | id | type, source | LLM-extracted entity container |
| Organization | name | — | Companies/institutions from LLM |
| Location | name | — | Places from LLM + events |
| Product | name | — | Products/services from LLM |
| Action | name | — | Verbs/actions from LLM |
| Amount | value | — | Monetary values from LLM |
| Date | value | — | Dates from LLM (as text strings) |
| Event | event_id | event_type, subject, event_date, event_time, duration, virtual_platform, virtual_link | Extracted calendar events |

**Note on Person duality:** Person nodes serve double duty — as `type='email_participant'` (from email headers) and as `type='entity_mention'` (from LLM body extraction). The merge logic is: `MERGE (p:Person:EntityMention {name: $name})` — this creates a Person node keyed on *name*, not *email*, which can create duplicates when the same person appears with different name spellings. This is a schema design flaw.

### Relationship Types

**Email communication graph:**
```
(Person)-[:SENT]->(Email)
(Person)-[:RECEIVED]->(Email)
(Person)-[:COPIED]->(Email)       # CC
(Person)-[:BCC]->(Email)
(Email)-[:BELONGS_TO]->(Thread)
(Email)-[:REPLY_TO]->(Email)
(Email)-[:HAS_LABEL]->(Label)
(Email)-[:HAS_ATTACHMENT]->(Attachment)
(Person)-[:BELONGS_TO]->(Domain)
```

**Semantic layer:**
```
(Email)-[:HAS_SEMANTIC_DATA]->(SemanticData)
(SemanticData)-[:MENTIONS_ORGANIZATION]->(Organization)
(SemanticData)-[:MENTIONS_PERSON]->(Person)
(SemanticData)-[:MENTIONS_PRODUCT]->(Product)
(SemanticData)-[:MENTIONS_LOCATION]->(Location)
(SemanticData)-[:MENTIONS_AMOUNT]->(Amount)
(SemanticData)-[:DESCRIBES_ACTION]->(Action)
(SemanticData)-[:MENTIONS_DATE]->(Date)
```

**Event layer:**
```
(Email)-[:CONTAINS_EVENT]->(Event)
(Event)-[:LOCATED_AT]->(Location)
(Person)-[:ATTENDS {role: 'organizer'|'attendee'|'cc'}]->(Event)
(Person)-[:ORGANIZES]->(Event)
```

### Design Rationale

The Email-as-hub design is intentional and well-documented in the paper. Every entity connects *through* the email, which:
1. Preserves provenance (which email mentioned which entity)
2. Enables thread-based traversal
3. Allows the TransE triples to encode "who communicated with whom about what"

The **SemanticData intermediate node** is a useful indirection layer — it separates the LLM extraction results from the email itself and allows multiple semantic extractions per email. However, using CREATE (not MERGE) for SemanticData means re-running extraction creates duplicate nodes.

---

## c. TRANSE IMPLEMENTATION

### Key Code: `embeddings/graph_embeddings.py`

**Model parameters (hardcoded in `train_transe()`, default args in `event_pipeline.py`):**
- `embedding_dim = 100`
- `batch_size = 128`
- `epochs = 50` (pipeline default) / `100` (paper claims)
- `lr = 0.001` (Adam)
- `margin = 1.0`

**Initialization:**
```python
# Entity embeddings: uniform in [-6/sqrt(d), 6/sqrt(d)] then L2-normalized
self.entity_embeddings.weight.data.uniform_(-6/np.sqrt(embedding_dim), 
                                             6/np.sqrt(embedding_dim))
self.entity_embeddings.weight.data = nn.functional.normalize(
    self.entity_embeddings.weight.data, p=2, dim=1
)
# Relation embeddings: same uniform init, NOT normalized
self.relation_embeddings.weight.data.uniform_(...)
```
This matches the original Bordes et al. (2013) initialization.

**Scoring function:**
```python
scores = torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=1, dim=1)
```
L1 norm, standard TransE. The paper says "||h + r - t||" which matches.

**Loss function:**
```python
return torch.mean(torch.relu(self.margin + pos_scores - neg_scores))
```
Margin-based ranking loss: L = max(0, γ + d(h+r,t) - d(h'+r,t')). Standard.

**Negative sampling (in `get_corrupted_scores()`):**
```python
# Half batch corrupts heads, half corrupts tails
corrupt_heads = torch.randint(0, self.entity_count, (batch_size // 2,))
corrupt_tails = torch.randint(0, self.entity_count, (batch_size - batch_size // 2,))
```

**CRITICAL BUG:** The corrupted scores are computed over the WRONG indices. The function returns `torch.cat([corrupt_head_scores, corrupt_tail_scores])` (batch_size total elements) but the positive scores passed to `loss()` are also batch_size. The shapes should match — they do — but the negative samples don't enforce the "closed world assumption" (i.e., corrupted triples might accidentally be valid triples). This is a known limitation acknowledged in the original TransE paper. Standard fidelity, no filtering.

**Post-update normalization:**
```python
with torch.no_grad():
    model.entity_embeddings.weight.data = nn.functional.normalize(
        model.entity_embeddings.weight.data, p=2, dim=1
    )
```
After each batch, entity embeddings are renormalized to unit sphere. Relation embeddings are NOT normalized. This is correct per the original paper.

**Modifications from vanilla TransE:**
- Uses Adam instead of SGD (paper says SGD, code uses Adam — paper is wrong about this)
- Batch negative sampling is per-batch not per-triple (slight efficiency modification)
- No "filtered" evaluation protocol (most modern KGE papers use filtered metrics)

**Triple extraction from Neo4j:**
```python
# Entity IDs are formatted as "{NodeType}_{identifier}"
source_id = f"{record['source_type']}_{record['source_name']}"
```
This means `Email_<message_id>`, `Person_<email>`, `Event_<event_id>`, etc. The mixed node types in a single embedding space is a design choice — it allows cross-type link prediction.

---

## d. EVENT EXTRACTION

### Method: Purely Rule-Based + spaCy NER (NO LLM for event extraction)

**IMPORTANT DISTINCTION:** The paper conflates two separate systems:
1. `analysis/semantic_extractor.py` → uses GPT-3.5-turbo (LLM)  
2. `event_extraction/extract_events.py` → uses regex + spaCy (NO LLM)

The event extraction is entirely rule-based. There is no LLM call in `extract_events.py`.

### Event Detection (`_is_likely_event_email()`):

Gates the pipeline. An email is considered event-bearing if ANY of:
- Subject contains event keywords: `["meeting", "call", "conference", "webinar", "seminar", "workshop", "appointment", "sync", ...]` (26 keywords)
- Subject matches: `invitation|calendar|invite|meeting|appointment|schedule`
- Subject/body matches date/time regex patterns
- Body contains event phrases: `["would like to invite you", "please join", "zoom link", "when:", "where:", ...]`
- Body contains ≥2 event keywords
- Body contains ≥2 date/time patterns

### Event Type Classification (`_identify_event_type()`):

Simple keyword-per-category counting:
```python
event_types = {
    "meeting": ["meeting", "sync", "sync-up", "catchup", "catch-up", "check-in", "standup", "stand-up"],
    "interview": ["interview", "candidate", "hiring", "job", "recruitment", "career", "position"],
    "conference": ["conference", "seminar", "webinar", "symposium", "summit", "convention"],
    "social": ["party", "celebration", "happy hour", "social", "gathering", "birthday", "anniversary"],
    "meal": ["lunch", "dinner", "breakfast", "coffee", "meal", "restaurant", "cafe"],
    "presentation": ["presentation", "demo", "demonstration", "showcase", "review"],
    "workshop": ["workshop", "training", "course", "session", "class"],
    "call": ["call", "phone", "dial-in", "conference call"]
}
```
Winner is max keyword count. Note: "interview" type gets triggered by "job" emails — this is why the README example shows LinkedIn job alert emails being classified as "interview" type. This is the most obvious weakness in the system.

### Date/Time Extraction (`_extract_datetime_info()`):

- Looks for `when:` pattern first
- Applies regex patterns for month names, weekdays, "today/tomorrow/next"
- Uses `dateparser` library to parse extracted date strings
- Finds time ranges: `\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*(?:-|to|until)` → duration calculation

### Location Extraction (`_extract_location()`):

Pattern priority:
1. Explicit `location:|place:|venue:|where:` pattern
2. `room N` or `conference room N` pattern
3. Street address regex
4. Virtual platform detection (zoom, teams, etc.) → returns `"Zoom (Virtual)"`

### Virtual Meeting (`_extract_virtual_meeting_info()`):

Extracts Zoom URL, meeting ID, password/passcode via regex. Handles Zoom, Teams, Google Meet, Skype, Webex.

### Attendee Extraction (`_extract_attendees()`):

1. From email headers (to, cc → 'attendee', from → 'organizer')
2. spaCy PERSON entities from "attendees:|participants:|invitees:" block in body
3. Email regex in attendee block

### Event Validity (`_is_valid_event()`):

Event must have subject/type AND at least one of: date, location, virtual meeting link. This is the quality gate — pure "I think this is an event" keywords aren't enough.

---

## e. SEMANTIC EXTRACTION

### LLM Prompt (verbatim from `analysis/semantic_extractor.py`):

```
Extract the following from this email:
- Email type (job application, statement, booking, receipt, etc.)
- Organizations (companies, institutions, etc.)
- People (names of individuals)
- Products (items, services mentioned)
- Locations (cities, countries, addresses)
- Monetary amounts (prices, costs, fees)
- Dates (deadlines, appointments, scheduled events)
- Actions (applied, paid, booked, etc.)

Return the result as structured JSON with this exact format:
{
  "type": "string",
  "organizations": ["string"],
  "people": ["string"],
  "products": ["string"],
  "locations": ["string"],
  "amounts": ["string"],
  "dates": ["string"],
  "actions": ["string"]
}

Email:
"""<subject + body>"""
```

**System message:** `"You are a precise entity extraction assistant. Extract structured data from emails and format it exactly as requested."`

**Model:** `gpt-3.5-turbo` (hardcoded default; constructor accepts `model` param)
**Temperature:** `0.1`
**Max tokens:** `800`
**Retry:** 3 attempts with 2-second delay

**Response parsing:** Regex `{.*}` with `re.DOTALL` on the response, then `json.loads()`. No schema validation after parsing — any JSON keys are accepted.

**spaCy fallback (`_extract_with_spacy()`):**
- ORG → organizations, PERSON → people, GPE/LOC → locations, DATE/TIME → dates, MONEY → amounts
- VERB lemmas (first 5) → actions
- Keyword-based email type guessing (job, receipt, booking, statement)
- products: always `[]` (spaCy has no product detection)

**No confidence scores.** No validation that extracted entities actually appear in the email. The LLM can hallucinate names and organizations — there's no post-processing check.

---

## f. AGENT INTERFACE

### `agent/agent.py — EmailGraphAgent.process_query()`

**This is NOT an LLM agent.** Despite being called "Agentic AI system," it is simple if/elif keyword routing:

```python
if any(term in query for term in ['overview', 'summary', 'statistics']):
    return self._handle_overview_query()
elif any(term in query for term in ['search', 'find emails', 'containing']):
    return self._handle_search_query(query)
elif any(term in query for term in ['person', 'contact', 'about']) and '@' in query:
    return self._handle_person_query(query)
elif any(term in query for term in ['connection', 'between', 'relate']):
    return self._handle_connection_query(query)
elif any(term in query for term in ['thread', 'conversation']):
    return self._handle_thread_query(query)
elif any(term in query for term in ['time', 'pattern', 'volume']):
    return self._handle_time_query(query)
elif any(term in query for term in ['domain', 'company']):
    return self._handle_domain_query(query)
else:
    return {"type": "error", "message": "I'm not sure how to process this query..."}
```

### Supported Query Intents

| Intent | Trigger Words | What It Does |
|---|---|---|
| Overview | overview, summary, statistics | Counts + top senders/receivers/domains |
| Search | search, find emails, containing | Full-text CONTAINS on subject + body |
| Person | person, contact, about + `@` | Person node details + top partners |
| Connection | connection, between, relate | Path traversal between 2 emails |
| Thread | thread, conversation | Thread details by ID or subject regex |
| Time | time, pattern, volume | Email volume grouped by day/week/month/year |
| Domain | domain, company | Domain stats (sent, received, people) |

### Cypher Queries Used

Most complex: `find_connections_between()` uses variable-length path:
```cypher
MATCH path = (p1:Person {email: $email1})-[*1..3]-(p2:Person {email: $email2})
RETURN path LIMIT 10
```
This is unbounded relationship-type traversal — will traverse through Email nodes, creating potentially misleading "connections."

**No natural language → Cypher translation.** No LLM involved in the agent. The paper implies an "LLM-powered interface" but the code doesn't have one.

### `event_query.py — EventQuery`

Separate, more powerful CLI query interface for the event layer. Commands:
- `recent [--limit N]` — recent events by date
- `details <event_id>` — full event info
- `search <keyword>` — CONTAINS search on subject
- `person <email>` — events for a person
- `dates <start> <end>` — date range filter
- `location <name>` — events at location
- `stats` — aggregate event statistics
- `similar <event_id>` — cosine similarity in embedding space
- `recommend <email>` — `find_events_for_person()` using TransE embeddings

The `similar` and `recommend` commands are where TransE embeddings actually get used. They compute cosine similarity (NOT the TransE scoring function — there's a mismatch here, see Novel Contributions).

---

## g. NOVEL CONTRIBUTIONS (Honest Assessment)

### What Is Genuinely New

**1. Domain-specific KG schema for email events**
The schema design — Email as hub, with Person, Thread, Domain, Label, Event, and Semantic layers — is well thought out for the email domain. The combination of structural email metadata with LLM-extracted semantic layer is interesting. This is the most defensible "design contribution."

**2. End-to-end pipeline from MBOX to KGE**
No prior published open-source system does the full MBOX → Neo4j → TransE pipeline. The Enron email corpus has been studied, but not with this specific KGE approach. The pipeline works and produces real results.

**3. Application of TransE to email link prediction**
MRR=0.564, Hits@10=0.65 on the email communication graph is a legitimate empirical result. The AUC=0.87 on triple classification is strong. These results show TransE captures meaningful structure in email communication graphs.

**4. Dual extraction architecture**
Using rule-based event extraction (fast, no API cost) + LLM semantic extraction (richer, more expensive) as separate, composable modules is a practical system design choice worth documenting.

### What Is Standard / Not Novel

- TransE itself (2013 paper)
- spaCy NER for entity extraction
- Neo4j for graph storage
- Regex-based date/time extraction
- The agent (it's if/elif, not novel)
- The semantic extraction prompt (standard zero-shot NER prompting)

### Where the Strongest Publishable Claim Lives

**"The KGE approach for email event understanding outperforms keyword search."**

Specifically: showing that TransE-based similarity (`similar <event_id>`) retrieves semantically related events that keyword search misses. The paper mentions this: "a search for 'budget review' returned relevant events labeled as 'financial planning' and 'quarterly review' based on their semantic similarity." **This is the core publishable claim** — but it's currently only supported by a single anecdote, not a rigorous evaluation.

The second publishable claim: **link prediction for participant discovery** (finding missing attendees for events). The paper's MRR=0.564 / Hits@10=0.65 results, if reproduced on a larger dataset and compared to baselines, would be solid.

### Critical Weaknesses

**1. Link prediction evaluation is too small.**
Only 100 test triples evaluated for link prediction. For a dataset with 59,085 relationships, 20% test split would give ~11,800 triples — the code tests only 100. The `max_entities=500` cap further limits the evaluation. The numbers are real but not statistically meaningful at this scale.

**2. No baseline comparison.**
No comparison to: keyword search, TF-IDF similarity, BM25, or even random. The paper claims superiority but shows no evidence against any alternative.

**3. The agent has no AI.**
The paper says "LLM-powered interface" but `process_query()` is `if/elif` keyword routing. There is no LLM call in the agent. This claim in the paper is false.

**4. Paper claims "parallel entity extraction" and "embedding compression."**
Neither is implemented in the code. The optimization section (Section 4.3) lists these as implemented but they are not.

**5. "Interview" classification of LinkedIn job alerts.**
The event type classification triggers "interview" on the word "job" in job listing emails. The README example shows this problem directly — LinkedIn job digest emails classified as "interview." The recall/precision numbers (83.7%/76.4%) from manual annotation of 200 emails may not reflect this systematic error.

**6. No confidence/quality scores from LLM extraction.**
The semantic extractor accepts any LLM output without validation. If the model hallucinates an organization name, it gets stored in the graph as fact.

**7. Git merge conflict in requirements.txt.**
The requirements.txt has unresolved `<<<<<<< Updated upstream` / `>>>>>>> Stashed changes` markers. The repo is not in a clean state.

**8. Cosine similarity vs. TransE scoring.**
The `query_similar_entities()` and `find_events_for_person()` functions use **cosine similarity** between embeddings, not the TransE scoring function `||h + r - t||`. These are different metrics. Using cosine similarity on TransE embeddings is a known alternative but should be justified.

---

## h. CODE QUALITY

### Structure: Good
Clear module separation. Configuration via env vars + argparse. tqdm progress bars throughout. Logging consistently used. Error handling with try/except at appropriate levels.

### Test Coverage: Zero
There are **no test files** in the repository. No unit tests, no integration tests, no pytest configuration. For a research codebase, the lack of tests makes it impossible to verify that code changes don't break the pipeline.

### Error Handling: Adequate but Inconsistent
- Good: `load_email()` catches individual email errors and continues
- Good: `extract_events_from_email()` limits body to 50,000 chars, handles None types
- Bad: `_create_semantic_nodes()` uses CREATE (not MERGE) for SemanticData — re-running creates duplicate nodes
- Bad: Amount/Date nodes use CREATE not MERGE — will create duplicates on re-run
- Bad: `get_corrupted_scores()` has dead code (`corrupt_head_batch`, `corrupt_tail_batch` are built but never used)

### Schema/Consistency Issues
```python
# In loader.py — BUG: SemanticData node
"CREATE (s:SemanticData {...")  # Should be MERGE if email might be reprocessed
"CREATE (a:Amount {value: $value})"  # No uniqueness, creates duplicates
"CREATE (d:Date {value: $value})"   # No uniqueness, creates duplicates
```

### Production-Readiness: Not Production-Ready
- Hardcoded password `12345678` in `event_pipeline.py` and `evaluate_embeddings.py` fallback
- No API rate limit handling beyond simple `time.sleep(1)` between batches
- No resumable processing (restart re-processes from scratch)
- No data validation on input JSON
- The `--safe-mode` flag exists because of Windows Unicode bugs — would need fixing for cross-platform

### Dead Code
In `TransE.get_corrupted_scores()`:
```python
# These variables are built but never used:
corrupt_head_batch = torch.cat([corrupt_heads, relations[head_indices], tails[head_indices]], dim=0)
corrupt_tail_batch = torch.cat([heads[tail_indices], relations[tail_indices], corrupt_tails], dim=0)
```

---

## i. WHAT'S MISSING (To Strengthen the Paper)

### Evaluation Gaps (Critical)

**1. Full link prediction evaluation**
Run `evaluate_embeddings.py` with `--max-entities` set to ALL entities (not 500), evaluated on the full 20% test set (~11,800 triples). Current 100-triple evaluation is inadequate for a paper claim.

**2. Baseline comparison for event retrieval**
Compare `similar <event_id>` (TransE cosine) against:
- BM25/TF-IDF on event subjects
- Embedding similarity from a pre-trained sentence transformer (e.g., all-MiniLM)
- Random baseline
This is the essential ablation for the core claim.

**3. Filtered vs. raw evaluation protocol**
Implement the standard filtered link prediction protocol: when computing ranks, remove known true triples from the candidate set. The current evaluation may underestimate performance.

**4. Event extraction against a standard benchmark**
The 200-email manual annotation is internal and not reproducible. If possible, evaluate on a public benchmark (e.g., TimeBank, ACE 2005) or release the annotation dataset.

**5. Embedding dimension ablation**
Test dim = [50, 100, 200] to show the effect of embedding dimension on task performance.

### Missing Features (For the Next Version)

**1. Real LLM agent**
Replace the `if/elif` keyword router with an actual LLM (GPT-4, Gemini, or local model) that can translate natural language to Cypher. The paper claims this exists — make it real.

**2. Incremental graph updates**
The current system requires full rebuild. A real system needs to handle "add new emails" without reloading everything.

**3. Event deduplication**
No deduplication logic for events — the same meeting invite sent to 10 people would create 10 event nodes (one per email). A thread-based deduplication step is needed.

**4. Temporal embeddings**
The paper suggests this as future work. Adding date as a structured property with time-aware scoring (e.g., TTransE) would be a real contribution.

**5. Filtering for TransE negative sampling**
Implement closed-world assumption filtering so that randomly sampled "negative" triples are actually negative.

---

## Key Files Summary Table

| File | Lines | Role | Quality Notes |
|---|---|---|---|
| `event_query.py` | 772 | Primary query CLI for events | Clean, well-structured |
| `evaluate_embeddings.py` | 663 | TransE evaluation suite | Solid but 100-triple LP test |
| `event_extraction/extract_events.py` | 608 | Rule-based event extraction | Overfires on job emails |
| `embeddings/graph_embeddings.py` | 530 | TransE model + training | Dead code in get_corrupted_scores |
| `graph_db/loader.py` | 498 | Neo4j data loading | CREATE vs MERGE bug |
| `event_pipeline.py` | 440 | Event pipeline orchestrator | Unicode workarounds needed |
| `main.py` | 428 | Main pipeline orchestrator | Clean |
| `analysis/semantic_analysis.py` | 344 | Semantic data analysis | Good |
| `analysis/queries.py` | 340 | Pre-built Cypher queries | Good |
| `agent/agent.py` | 325 | "Agent" (actually keyword router) | Misleadingly named |
| `analysis/semantic_extractor.py` | 323 | LLM-based semantic extraction | No output validation |
| `event_extraction/event_to_graph.py` | 301 | Event → Neo4j | Good |
| `analysis/metrics.py` | 265 | Email metrics (not deeply read) | Not analyzed |
| `agent/interactive.py` | 225 | REPL for agent | Fine |
| `agent/actions.py` | 211 | Agent action implementations | Fine |
| `email_parser/email_extractor.py` | 159 | Email parsing utilities | Solid |
| `email_parser/mbox_parser.py` | 151 | MBOX file parsing | Solid |
| `graph_db/schema.py` | 108 | Neo4j schema setup | Good |

---

*Report by Commander Riker | Email-Link Introspection | 2026-02-17*
