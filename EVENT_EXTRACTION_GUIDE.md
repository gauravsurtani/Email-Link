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

## Working with Events

### New User-Friendly Event Selection

The system has been enhanced with features to make working with events easier:

1. **Using Event Numbers Instead of IDs**
   
   Events are now displayed with a simple number (#) column that you can use instead of copying long event IDs:
   
   ```
   # Recent Events (3 found):
   +----+--------------------+----------+-------------------+-------------+
   |  # | Subject            | Type     | Date/Time         | Location    |
   +====+====================+==========+===================+=============+
   |  1 | Team Meeting       | meeting  | 2023-05-15 10:00  | Room 101    |
   +----+--------------------+----------+-------------------+-------------+
   |  2 | Project Update     | update   | 2023-05-10 14:30  | Zoom        |
   +----+--------------------+----------+-------------------+-------------+
   |  3 | Coffee Chat        | meeting  | 2023-05-08 09:00  | Cafeteria   |
   +----+--------------------+----------+-------------------+-------------+
   
   Tip: Use event number from the '#' column with the 'select' command instead of copying event IDs.
   ```

2. **Selecting Events by Number**
   
   Instead of copying the event ID, you can now use:
   
   ```
   python event_query.py select 1
   ```
   
   This will select the first event from the most recently displayed list and show its details.

3. **Finding Events by Subject**
   
   If you remember part of the event name/subject but not the ID:
   
   ```
   python event_query.py subject "Meeting"
   ```
   
   This will find all events with "Meeting" in the subject line.

4. **Working with a Selected Event**
   
   Once you've selected an event, you can recall it at any time:
   
   ```
   python event_query.py show-selected
   ```
   
   This will show the details of the currently selected event again.

These improvements make it much easier to work with events without needing to know or copy complex event IDs.

## Technical Deep Dive: Knowledge Graph Embeddings for Event Analysis

### Understanding Knowledge Graph Embeddings

The email event extraction system employs state-of-the-art knowledge graph embedding techniques to provide intelligent event recommendations and find semantically similar events. Here's how it works:

#### The Knowledge Graph Foundation

The system begins by constructing a rich knowledge graph in Neo4j with:

1. **Core Entities**:
   - **Emails**: Messages in your inbox
   - **Persons**: Email senders and recipients
   - **Events**: Extracted events (meetings, interviews, etc.)
   - **Locations**: Places where events occur

2. **Relationships**:
   - Person SENT Email
   - Person RECEIVED Email
   - Person ATTENDS Event
   - Person ORGANIZES Event
   - Email CONTAINS_EVENT Event
   - Event LOCATED_AT Location

This graph representation captures the complex network of interactions between people and events in your email data.

#### TransE: Translating Embeddings

The system uses the **TransE** (Translating Embeddings) algorithm to learn vector representations of all entities and relationships in the knowledge graph:

1. **Mathematical Foundation**:
   - Each entity (person, email, event) is represented as a vector in a high-dimensional space (typically 100 dimensions)
   - Each relationship type is also represented as a vector in the same space
   - TransE models relationships as translations in this vector space

2. **The Key Insight**:
   If a triple (head, relation, tail) exists in the graph (e.g., "Person_john ATTENDS Event_meeting"), then:
   ```
   vector(head) + vector(relation) ≈ vector(tail)
   ```
   Or more concretely:
   ```
   vector(Person_john) + vector(ATTENDS) ≈ vector(Event_meeting)
   ```

3. **Learning Process**:
   - The system extracts all triples from the Neo4j database
   - It trains the TransE model iteratively, optimizing these vector representations
   - Training minimizes the distance between `h + r` and `t` for valid triples
   - The process uses negative sampling (corrupted triples) to ensure discrimination
   - PyTorch is used to implement and train the model efficiently

#### From Graph to Embeddings to Embeddings File

The embedding generation process occurs during the `event_pipeline.py` execution:

1. The `GraphEmbeddings` class extracts triples from Neo4j
2. It prepares a PyTorch dataset with these triples
3. The TransE model is trained using gradient descent
4. Learned embeddings are saved to CSV files:
   - `entity_embeddings.csv`: Contains vectors for all entities
   - `relation_embeddings.csv`: Contains vectors for all relationships

### How Similarity Search Works

When you run `python event_query.py similar EVENT_ID`, the system:

1. **Loads the pre-trained embeddings** from the CSV files
2. **Retrieves the embedding vector** for the specified event
3. **Calculates cosine similarity** between the target event vector and all other event vectors:
   ```python
   similarities = np.dot(all_embeddings, entity_embedding) / (
       np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(entity_embedding)
   )
   ```
4. **Ranks events by similarity** and returns the top-k most similar events
5. For each similar event, it retrieves the full details from Neo4j

The power of this approach is that it finds events that are semantically similar, not just textually similar. Two events might be related even if they have different descriptions, because the embedding captures their context in the graph (who attends them, where they take place, what kind of emails mention them).

### How Event Recommendations Work

When you run `python event_query.py recommend EMAIL`, the system:

1. **Identifies the person's embedding vector** using their email address
2. **Gathers embeddings for all events** in the knowledge graph
3. **Computes similarity scores** between the person's vector and each event vector
4. **Ranks events** by their relevance to the person
5. **Returns detailed information** about the most relevant events

This process can discover events that might be relevant to you even if you've never directly interacted with them. The system might recommend an event because:
- People similar to you have attended similar events
- The event is related to topics you've shown interest in
- The event has semantic connections to your communication patterns

### Technical Advantages of the Embedding Approach

1. **Captures Latent Semantics**: The embeddings capture hidden patterns in your email data that wouldn't be visible from simple querying.

2. **Handles Sparsity**: Even with limited data, the embedding space can generalize relationships and make meaningful recommendations.

3. **Scales Efficiently**: Once embeddings are generated, similarity calculations are extremely fast, even with thousands of events.

4. **Continuous Improvement**: As more email data is processed, the embeddings become more refined and accurate.

5. **Cold Start Handling**: New events and persons can be mapped into the existing embedding space using their known relationships.

The embedding-based approach allows the system to go beyond simple text matching, enabling it to discover semantic connections and make intelligent recommendations that might not be obvious from looking at the raw data. 