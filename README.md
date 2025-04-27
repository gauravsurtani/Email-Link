# Email Graph Analysis System

## Overview
This project transforms your Gmail data into a powerful graph database, enabling deep analysis of communication patterns and relationships. By converting Gmail Takeout (.mbox files) into a Neo4j graph, users can visualize connections, discover communication patterns, and gain insights from their email history. The system also extracts events from emails, allowing you to discover meetings, interviews, and other calendar items.

## Features
- 📧 Efficiently parses Gmail Takeout MBOX files
- 🔄 Constructs a comprehensive graph database of email communications
- 🔍 Enables complex queries to analyze communication patterns 
- 📊 Provides metrics for communication frequency, response times, and network centrality
- 🤖 Includes an AI agent interface for natural language exploration of the email graph
- 📅 Extracts events and appointments from emails for calendar analysis
- 🧠 Generates knowledge graph embeddings for event similarity and recommendations

## Requirements
- Python 3.8+
- Neo4j 4.4+ (Desktop or Server)
- Gmail Takeout MBOX file
- Additional packages listed in requirements.txt

## Project Structure
```
EmailLink/
├── requirements.txt        # Project dependencies
├── config.py               # Configuration handling
├── email_parser/           # Email parsing module
│   ├── __init__.py
│   ├── mbox_parser.py      # MBOX file parsing
│   └── email_extractor.py  # Email data extraction
├── graph_db/               # Graph database module
│   ├── __init__.py
│   ├── schema.py           # Database schema
│   └── loader.py           # Data loading into Neo4j
├── analysis/               # Email analysis module
│   ├── __init__.py
│   ├── queries.py          # Common graph queries
│   └── metrics.py          # Compute email metrics
├── agent/                  # AI agent module
│   ├── __init__.py
│   ├── agent.py            # Main agent implementation
│   └── actions.py          # Agent actions
├── event_extraction/       # Event extraction module
│   ├── __init__.py
│   ├── extract_events.py   # Event extraction logic
│   └── event_to_graph.py   # Event graph building
├── embeddings/             # Knowledge graph embeddings
│   └── graph_embeddings.py # Graph embedding generation
├── event_pipeline.py       # Event extraction pipeline
├── event_query.py          # Event querying tools
└── main.py                 # Main execution script
```

## Setup

### 1. Install Neo4j
- Download and install [Neo4j Desktop](https://neo4j.com/download/)
- Create a new database (click "+ Add" → "Local DBMS")
- Set a password and start the database
- Note the connection URI (typically `bolt://localhost:7687`)

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root with your Neo4j credentials:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OUTPUT_DIR=./output
```

## Getting Started

### 1. Export Your Gmail Data
- Go to [Google Takeout](https://takeout.google.com/)
- Select only "Mail" and choose the MBOX format
- Download the export when complete

### 2. Process Your Email Data
```bash
python main.py --mbox path/to/your/takeout-file.mbox
```

### 3. Extract Events from Emails
```bash
python event_pipeline.py --safe-mode
```

### 4. Explore Your Email Graph and Events
The system will:
- Parse your MBOX file
- Extract communication metadata and events
- Build a graph database
- Generate knowledge graph embeddings
- Enable complex queries for analysis

## Email Processing Options
```bash
python main.py --mbox path/to/your/takeout-file.mbox
```

Additional options:
- `--skip-parsing`: Skip MBOX parsing and use existing parsed data
- `--output-json FILE`: Save parsed emails to a specific JSON file
- `--neo4j-uri URI`: Specify Neo4j connection URI (overrides .env file)
- `--neo4j-user USER`: Specify Neo4j username (overrides .env file)
- `--neo4j-password PASSWORD`: Specify Neo4j password (overrides .env file)

## Event Extraction and Analysis
Run the event extraction pipeline:
```bash
python event_pipeline.py --safe-mode
```

Options:
- `--skip-extraction`: Skip event extraction and use existing events
- `--skip-embeddings`: Skip embedding generation
- `--embedding-dim 100`: Set embedding dimensions (default: 100)
- `--epochs 50`: Set training epochs (default: 50)
- `--safe-mode`: Use safe mode for encoding issues (recommended on Windows)
- `--output-dir DIR`: Specify output directory

## Querying Events

### Recent Events
View most recent events (like your job listings and interview invitations):
```bash
python event_query.py recent --limit 10
```

Example output:
```
Recent Events (5 found):
+-------------+---------------------------------------------------------------+-----------+---------------+
| ID          | Subject                                                       | Type      | Date/Time     |
+=============+===============================================================+===========+===============+
| 20240924... | Louis Vuitton, Coffee & Ketchup Ice-cream🍨                   | interview | 2026-01-01 23 |
+-------------+---------------------------------------------------------------+-----------+---------------+
| q3yWE83A... | We think You and Monterey Bay Aquarium could be a great match | interview | 2025-12-31 28 |
+-------------+---------------------------------------------------------------+-----------+---------------+
| 44McDmDo... | Apply for these new Intern jobs in San Jose, CA today         | interview | 2025-12-30 28 |
+-------------+---------------------------------------------------------------+-----------+---------------+
| oXffQfnk... | Gaurav, don't miss these new Intern jobs                      | interview | 2025-12-29 28 |
+-------------+---------------------------------------------------------------+-----------+---------------+
| U0G-PAv1... | Gaurav, don't miss these new Intern jobs                      | interview | 2025-12-28 28 |
+-------------+---------------------------------------------------------------+-----------+---------------+
```

### Event Details
Get detailed information about a specific event:
```bash
python event_query.py details 20240924...
```

### Search Events
Search for events with keywords:
```bash
python event_query.py search "Intern jobs"
python event_query.py search "San Jose"
```

### Find Interview Events
Find all interview events:
```bash
python event_query.py search "interview" --limit 20
```

Or search for specific interview opportunities:
```bash
python event_query.py search "interview jobs" --limit 25
```

### Person's Events
Find events involving a specific person:
```bash
python event_query.py person gaurav@example.com
```

### Date Range Events
View events within a date range:
```bash
python event_query.py dates 2025-12-01 2026-01-31
```

### Location-based Events
Find events at a specific location:
```bash
python event_query.py location "San Jose"
```

### Event Statistics
Get statistics about your event data:
```bash
python event_query.py stats
```

### Similar Events (Using Embeddings)
Find similar job opportunities using knowledge graph embeddings:
```bash
python event_query.py similar 44McDmDo... --limit 5
```

### Event Recommendations (Using Embeddings)
Get personalized job recommendations:
```bash
python event_query.py recommend gaurav@example.com --limit 8
```

## Example Email Queries
After loading your data, try these Neo4j Cypher queries:

```cypher
// Find your top 10 correspondents
MATCH (you:Person)-[r:SENT|RECEIVED]-(contact:Person)
RETURN contact.email, count(r) AS communications
ORDER BY communications DESC
LIMIT 10

// See email activity over time
MATCH (e:Email)
RETURN substring(e.date, 0, 7) AS month, count(*) AS emails
ORDER BY month

// Identify communication clusters
MATCH path = (p1:Person)-[:SENT|RECEIVED*2]-(p2:Person)
WHERE p1 <> p2
RETURN path
LIMIT 100

// Find emails containing events
MATCH (e:Email)-[:CONTAINS_EVENT]->(event:Event)
RETURN e.subject, event.subject, event.event_type, event.event_date
ORDER BY event.event_date DESC
LIMIT 20
```

## Troubleshooting

### Email Processing Issues
- **Memory Issues**: For large MBOX files, increase Java heap size in Neo4j config
- **Import Errors**: Ensure your MBOX file is in standard format from Gmail Takeout
- **Connection Issues**: Verify Neo4j is running and credentials are correct

### Event Extraction Issues
- **Unicode Encoding Issues (Windows)**: If you encounter encoding errors with emojis, use:
  ```bash
  python event_pipeline.py --safe-mode
  ```
- **Missing Embeddings**: If embedding queries don't work, ensure you've run the pipeline without `--skip-embeddings`
- **Long URLs in Locations**: If you see long URLs in your location fields, consider modifying the location extraction logic

## License
This project is available under the MIT License.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.