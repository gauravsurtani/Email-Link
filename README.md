# Email Graph Analysis System

## Overview
This project transforms your Gmail data into a powerful graph database, enabling deep analysis of communication patterns and relationships. By converting Gmail Takeout (.mbox files) into a Neo4j graph, users can visualize connections, discover communication patterns, and gain insights from their email history.

## Features
- 📧 Efficiently parses Gmail Takeout MBOX files
- 🔄 Constructs a comprehensive graph database of email communications
- 🔍 Enables complex queries to analyze communication patterns 
- 📊 Provides metrics for communication frequency, response times, and network centrality
- 🤖 Includes an AI agent interface for natural language exploration of the email graph

## Requirements
- Python 3.8+
- Neo4j 4.4+ (Desktop or Server)
- Gmail Takeout MBOX file

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

### 3. Explore Your Email Graph
The system will:
- Parse your MBOX file
- Extract communication metadata
- Build a graph database
- Enable complex queries for analysis

## Usage Options
```bash
python main.py --mbox path/to/your/takeout-file.mbox
```

Additional options:
- `--skip-parsing`: Skip MBOX parsing and use existing parsed data
- `--output-json FILE`: Save parsed emails to a specific JSON file
- `--neo4j-uri URI`: Specify Neo4j connection URI (overrides .env file)
- `--neo4j-user USER`: Specify Neo4j username (overrides .env file)
- `--neo4j-password PASSWORD`: Specify Neo4j password (overrides .env file)

## Example Queries
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
```

## Troubleshooting
- **Memory Issues**: For large MBOX files, increase Java heap size in Neo4j config
- **Import Errors**: Ensure your MBOX file is in standard format from Gmail Takeout
- **Connection Issues**: Verify Neo4j is running and credentials are correct

## License
This project is available under the MIT License.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.