# Email Event Extraction Guide

This guide explains how to use the email event extraction system to identify events from your Gmail Takeout data, add them to a knowledge graph, and generate embeddings for advanced queries.

## Setup

1. **Install Dependencies**

   Run the setup script to install all required dependencies:
   ```
   python setup.py
   ```
   
   This will:
   - Install Python packages from requirements.txt
   - Download the spaCy language model
   - Create necessary directories
   - Generate a default .env file

2. **Configure Neo4j**

   Make sure Neo4j is installed and running. The system is configured to connect to:
   - URI: bolt://localhost:7687
   - Username: neo4j
   - Password: 12345678
   
   If your Neo4j configuration is different, edit the `.env` file.

3. **Prepare Email Data**

   Place your Gmail Takeout MBOX files in the `input` directory.

## Running the System

### Step 1: Process Email Data

First, process all your MBOX files to create the email graph:

```
python main.py --process-all
```

This will:
- Parse all MBOX files in the `input` directory
- Extract email metadata, content, and relationships
- Create a graph database in Neo4j

### Step 2: Extract Events

Run the event extraction pipeline to identify events in your emails:

```
python event_pipeline.py
```

This will:
- Analyze emails to identify events (meetings, calls, conferences, etc.)
- Extract event details (dates, times, locations, participants)
- Add events to the graph database
- Generate knowledge graph embeddings

### Step 3: Query Events

Use the event query tool to explore and analyze events:

```
# Show recent events
python event_query.py recent

# Search for events
python event_query.py search "budget meeting"

# Show events for a person
python event_query.py person user@example.com

# Show events in a date range
python event_query.py dates 2023-01-01 2023-12-31

# Show event statistics
python event_query.py stats
```

### Step 4: Use Embeddings for Advanced Queries

The system uses knowledge graph embeddings (TransE) for semantic queries:

```
# Find similar events
python event_query.py similar EVENT_ID

# Get event recommendations for a person
python event_query.py recommend user@example.com
```

## Command Reference

### `main.py` - Email Processing

- `python main.py --process-all`: Process all MBOX files in the input directory
- `python main.py --mbox "input/file.mbox"`: Process a specific MBOX file

### `event_pipeline.py` - Event Extraction

- `python event_pipeline.py`: Run complete pipeline (extract events and generate embeddings)
- `python event_pipeline.py --skip-extraction`: Skip event extraction, only generate embeddings
- `python event_pipeline.py --skip-embeddings`: Extract events but skip embedding generation
- `python event_pipeline.py --embedding-dim 200 --epochs 100`: Configure embedding parameters

### `event_query.py` - Querying Events

- `python event_query.py recent [--limit N]`: Show recent events
- `python event_query.py details EVENT_ID`: Show details for a specific event
- `python event_query.py search QUERY [--limit N]`: Search for events
- `python event_query.py person EMAIL [--limit N]`: Show events for a person
- `python event_query.py dates START END [--limit N]`: Show events in a date range
- `python event_query.py location NAME [--limit N]`: Show events at a location
- `python event_query.py stats`: Show event statistics
- `python event_query.py similar EVENT_ID [--limit N]`: Find similar events using embeddings
- `python event_query.py recommend EMAIL [--limit N]`: Recommend events for a person

## Troubleshooting

**Import Error: No module named 'X'**

If you encounter an import error, install the missing dependency:
```
pip install X
```

Or run the setup script again:
```
python setup.py
```

**Neo4j Connection Errors**

- Make sure Neo4j is running and accessible
- Check your credentials in the `.env` file
- Neo4j Desktop: Verify the database is started and the correct port is used

**Embedding Generation Errors**

- The embedding generation requires significant memory for large datasets
- For large email collections, reduce batch size in `graph_embeddings.py`
- If GPU is available, it will be used automatically for faster training 