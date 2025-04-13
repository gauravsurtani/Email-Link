# Email Graph Analysis System

This project extracts data from Gmail Takeout (.mbox files) and builds a graph database to enable powerful analysis of email communication patterns. The system uses a modular approach to process emails, build a Neo4j graph database, and provide query capabilities.

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
└── main.py                 # Main execution script
```

## Setup

1. Install Neo4j:
   - Download and install Neo4j Desktop from https://neo4j.com/download/
   - Create a new database and note your connection details

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Neo4j credentials:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Usage

Run the pipeline to process an MBOX file and load it into Neo4j:

```bash
python main.py --mbox path/to/your/takeout-file.mbox
```

Additional options:
- `--skip-parsing`: Skip MBOX parsing and use existing parsed data
- `--output-json FILE`: Save parsed emails to a specific JSON file
- `--neo4j-uri URI`: Specify Neo4j connection URI (defaults to .env file)
- `--neo4j-user USER`: Specify Neo4j username (defaults to .env file)
- `--neo4j-password PASSWORD`: Specify Neo4j password (defaults to .env file)

## Features

- Parses Gmail Takeout MBOX files
- Builds a comprehensive graph database of email connections
- Enables complex graph queries for communication analysis
- Provides an agentic interface for exploring the email graph 