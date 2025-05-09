# Step 2: Data Collection & Preprocessing

## Objective
Prepare a clean, structured dataset of emails for semantic extraction and graph construction. This step ensures that all necessary information is available in a consistent format for downstream processing.

## Tools & Technologies
- **Python mailbox**: For parsing MBOX files (Gmail Takeout or synthetic data)
- **LlamaIndex Mbox Reader**: For rapid ingestion and parsing ([LlamaIndex Email Data Extraction Example](https://docs.llamaindex.ai/en/stable/examples/usecases/email_data_extraction/))
- **Pandas**: For data manipulation and storage
- **CSV/JSON**: For storing parsed email data

## Actions
1. **Collect Email Data**
   - Gather a small, diverse set of emails (10–50 per use case) covering all target queries.
   - Use real Gmail Takeout data or generate synthetic emails for each scenario.

2. **Parse Emails**
   - Use Python's `mailbox` or LlamaIndex Mbox Reader to extract:
     - Subject
     - Body
     - Sender
     - Recipient(s)
     - Date
     - (Optional) Attachments, labels, thread info

3. **Store Parsed Data**
   - Save parsed emails in a structured format (CSV, JSON, or Pandas DataFrame).
   - Ensure each record contains all required fields for semantic extraction.

4. **Validate Data Quality**
   - Check for missing or malformed fields.
   - Ensure consistent formatting (e.g., date formats, email addresses).
   - Remove duplicates and irrelevant emails.

## Next Actions
- [ ] Write or adapt a script to parse and save email data from MBOX or sample files.
- [ ] Validate that all required fields are present and correctly formatted.
- [ ] Create a summary (count, types, date range) of the collected dataset.
- [ ] Store the cleaned dataset in the project directory for use in the next step.

---

**Document created for Step 2 of the Semantic Email Knowledge Graph PoC.** 