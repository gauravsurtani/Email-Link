# Email-Link Examples

This directory contains example scripts demonstrating different features of the Email-Link system.

## Semantic Data Extraction

### Example: Extracting Entities and Actions from an Email

```bash
python examples/semantic_extraction_example.py
```

This example:
1. Creates a sample email (job application)
2. Extracts semantic data (entities, actions, dates, etc.) using LLMs
3. Displays the extracted information
4. Saves the enriched email to a JSON file

### Requirements
- OpenAI API key in your `.env` file
- Installed dependencies (`pip install -r requirements.txt`)

## Running the Complete Pipeline

To run the full pipeline with semantic extraction:

```bash
# Process a single MBOX file with semantic extraction
python main.py --mbox input/your-mbox-file.mbox --extract-semantic

# Process all MBOX files in the input directory
python main.py --process-all --extract-semantic

# Test semantic extraction with a small subset (5 emails)
python main.py --process-all --extract-semantic --semantic-test-mode

# Extract semantic data separately
python extract_semantic_data.py --input output/parsed_emails.json --output output/semantic_data

# Analyze the extracted semantic data
python analyze_semantic_data.py --data output/semantic_data --output output/analysis --visualize

# Load previously extracted semantic data into Neo4j
python main.py --load-semantic-only --semantic-data-dir output/semantic_data
```

## Output

Extracted semantic data is saved to:
- `examples/output/`: For example scripts
- `output/semantic_data/`: For the main application

The data includes:
- Email type classification
- Organizations mentioned
- People mentioned
- Products/services mentioned
- Locations mentioned
- Monetary amounts
- Dates and times
- Actions performed 