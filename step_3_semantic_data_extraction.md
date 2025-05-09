# Step 3: Semantic Data Extraction (Entity & Action Recognition)

## Objective
Extract structured semantic information from each email, including entities (organizations, people, products, locations), actions (applied, paid, booked, etc.), monetary amounts, dates, and email type. This enables rich graph construction and advanced querying in later steps.

---

## Tools & Technologies
- **GenAI LLMs:** OpenAI GPT-3.5/4, Gemini, Llama2/3 (via API)
- **Prompt Engineering:** Custom prompts for structured extraction
- **spaCy:** For fallback Named Entity Recognition (NER)
- **Python:** For orchestration and data handling

---

## Extraction Workflow

### 1. Design Extraction Prompts
- Draft prompts instructing the LLM to extract:
  - Email type (e.g., job application, statement, booking, receipt)
  - Organizations, people, products, locations
  - Monetary amounts, dates, actions
- Example prompt:
  ```
  Extract the following from this email:
  - Email type (job application, statement, booking, etc.)
  - Organizations
  - People
  - Products
  - Locations
  - Monetary amounts
  - Dates
  - Actions
  Return the result as structured JSON.
  Email:
  """
  {email_body}
  """
  ```
- Test and refine the prompt for clarity and completeness.

### 2. Implement Extraction Script
- For each email in your parsed dataset:
  - Send the email content (subject + body) to the LLM with the designed prompt.
  - Parse the LLM's JSON output.
  - If the LLM output is invalid or incomplete, optionally fall back to spaCy NER for basic entity extraction.
- Store the extracted semantic data alongside the original email record.

### 3. Batch Processing & Rate Limiting
- Implement batching to avoid API rate limits and reduce costs.
- Log all requests and responses for traceability and debugging.
- Handle API errors and retries gracefully.

### 4. Output Format
- Store the enriched email data as JSON, with a structure like:
  ```json
  {
    "message_id": "...",
    "subject": "...",
    "body": "...",
    "entities": {
      "type": "job application",
      "organizations": ["Walmart"],
      "people": ["John Doe"],
      "products": [],
      "locations": ["San Francisco"],
      "amounts": ["$5000"],
      "dates": ["2024-06-01"],
      "actions": ["applied"]
    }
  }
  ```
- Save the enriched dataset for use in graph construction.

---

## Validation & Quality Assurance
- **Manual Review:** Spot-check a sample of extracted records for accuracy and completeness.
- **Automated Checks:**
  - Ensure all required fields are present.
  - Validate JSON structure and types.
  - Flag and log any extraction failures or anomalies.
- **Iterate:** Refine prompts and extraction logic based on observed issues.

---

## Next Actions
- [ ] Finalize and test extraction prompts for each use case.
- [ ] Implement the extraction script (LLM API calls, parsing, fallback to spaCy if needed).
- [ ] Run extraction on the cleaned dataset.
- [ ] Validate and review extracted semantic data.
- [ ] Document the enriched data schema and sample records.
- [ ] Prepare the dataset for graph construction (Step 4).

---

**Document created for Step 3 of the Semantic Email Knowledge Graph PoC.** 