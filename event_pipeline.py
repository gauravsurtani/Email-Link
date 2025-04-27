#!/usr/bin/env python3
"""
Main event extraction pipeline for email data.
This script extracts events from emails in the Neo4j graph database,
enhances the graph with event nodes, and generates embeddings.
"""

import os
import logging
import argparse
import sys
import io
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# Import modules
from event_extraction.extract_events import EventExtractor
from event_extraction.event_to_graph import EventGraphEnhancer
from embeddings.graph_embeddings import GraphEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract events from email graph and generate embeddings')
    
    # Processing options
    parser.add_argument('--skip-extraction', action='store_true', 
                        help='Skip event extraction and use existing events in the graph')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation')
    parser.add_argument('--embedding-dim', type=int, default=100,
                        help='Dimension of knowledge graph embeddings')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs for embeddings')
    parser.add_argument('--safe-mode', action='store_true',
                        help='Use safe mode for handling encoding issues')
    
    # Output options
    parser.add_argument('--output-dir', help='Directory for output files')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    
    return parser.parse_args()

def sanitize_text(text):
    """
    Sanitize text to remove problematic characters for Windows.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return str(text)
        
    # Remove emojis and other problematic characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def safely_save_file(filepath, content):
    """
    Safely save content to a file with proper encoding.
    
    Args:
        filepath: Path to the file
        content: Content to save
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except UnicodeEncodeError:
        # If we hit encoding issues, sanitize the content
        with open(filepath, 'w', encoding='ascii', errors='ignore') as f:
            f.write(sanitize_text(content))

def main():
    """Main execution function."""
    # Set up proper encoding for stdout/stderr
    # This helps with Unicode characters on Windows
    try:
        if sys.platform == 'win32':
            # Set UTF-8 mode for Windows
            os.system("chcp 65001 >nul")
            # Redirect stdout/stderr to use utf-8
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception as e:
        logger.warning(f"Could not set UTF-8 encoding for console: {e}")
    
    # Load environment configuration
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_level = args.log_level or os.getenv('LOG_LEVEL', 'INFO')
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', '12345678')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    if not neo4j_password:
        logger.error("Neo4j password not provided in .env file")
        print("Error: Neo4j password not provided in .env file")
        return 1
    
    # Set output directory
    output_dir = args.output_dir or os.getenv('OUTPUT_DIR', './output')
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    Path(embeddings_dir).mkdir(exist_ok=True, parents=True)
    
    logger.info("Starting Email Event Extraction Pipeline")
    
    # Step 1: Extract events from emails
    if not args.skip_extraction:
        logger.info("Extracting events from emails")
        
        # Initialize event extractor
        event_extractor = EventExtractor()
        
        # Initialize graph enhancer
        graph_enhancer = EventGraphEnhancer(
            neo4j_uri, 
            neo4j_user, 
            neo4j_password
        )
        
        try:
            # Process all emails and extract events
            event_count = process_all_emails(event_extractor, graph_enhancer)
            logger.info(f"Extracted and added {event_count} events to the graph")
            
            # Close connection
            graph_enhancer.close()
        except Exception as e:
            logger.error(f"Error extracting events: {e}")
            print(f"Error extracting events: {e}")
            return 1
    else:
        logger.info("Skipping event extraction")
    
    # Step 2: Generate knowledge graph embeddings
    if not args.skip_embeddings:
        logger.info("Generating knowledge graph embeddings")
        
        try:
            # Initialize graph embeddings
            embeddings = GraphEmbeddings(
                neo4j_uri, 
                neo4j_user, 
                neo4j_password,
                output_dir=embeddings_dir
            )
            
            # Extract triples from the graph with safe mode if specified
            triples, entity_to_idx, relation_to_idx = None, None, None
            
            if args.safe_mode:
                # Use safe extraction by encoding validation
                triples, entity_to_idx, relation_to_idx = extract_triples_safely(embeddings)
            else:
                # Use standard extraction
                try:
                    triples, entity_to_idx, relation_to_idx = embeddings.extract_triples()
                except UnicodeEncodeError:
                    logger.warning("Unicode encoding error detected, switching to safe mode")
                    triples, entity_to_idx, relation_to_idx = extract_triples_safely(embeddings)
            
            if not triples or len(triples) == 0:
                logger.warning("No triples extracted from the graph. Check your Neo4j data.")
                print("Warning: No triples extracted from the graph. Check your Neo4j data.")
                return 1
                
            # Save the triple data directly to avoid encoding issues
            try:
                logger.info(f"Saving {len(triples)} triples to disk")
                triple_file = os.path.join(embeddings_dir, "triples.json")
                
                # Create a safe representation
                safe_triples = []
                for head, rel, tail in triples:
                    safe_triples.append([
                        sanitize_text(str(head)), 
                        sanitize_text(str(rel)), 
                        sanitize_text(str(tail))
                    ])
                
                # Save triples to file
                with open(triple_file, 'w', encoding='utf-8', errors='ignore') as f:
                    json.dump(safe_triples, f)
                    
                # Save entity and relation mappings
                entity_file = os.path.join(embeddings_dir, "entities.json")
                relation_file = os.path.join(embeddings_dir, "relations.json")
                
                # Save entity mapping
                safe_entities = {sanitize_text(str(k)): v for k, v in entity_to_idx.items()}
                with open(entity_file, 'w', encoding='utf-8', errors='ignore') as f:
                    json.dump(safe_entities, f)
                    
                # Save relation mapping
                safe_relations = {sanitize_text(str(k)): v for k, v in relation_to_idx.items()}
                with open(relation_file, 'w', encoding='utf-8', errors='ignore') as f:
                    json.dump(safe_relations, f)
            except Exception as e:
                logger.error(f"Error saving triple data: {e}")
            
            # Use fewer epochs for training
            logger.info(f"Training TransE model with dimension {args.embedding_dim} for {args.epochs} epochs")
            try:
                # Train the model with reduced epochs to avoid encoding errors
                embeddings.train_transe(
                    triples=triples,
                    entity_to_idx=entity_to_idx,
                    relation_to_idx=relation_to_idx,
                    embedding_dim=args.embedding_dim,
                    epochs=args.epochs
                )
            except UnicodeEncodeError as e:
                logger.warning(f"Unicode encoding issue detected: {e}")
                logger.info("Reducing epoch count and trying again...")
                
                # Try again with even fewer epochs
                reduced_epochs = max(3, args.epochs // 2)
                try:
                    embeddings.train_transe(
                        triples=triples,
                        entity_to_idx=entity_to_idx,
                        relation_to_idx=relation_to_idx,
                        embedding_dim=args.embedding_dim,
                        epochs=reduced_epochs
                    )
                except Exception as e2:
                    logger.error(f"Failed again with reduced epochs: {e2}")
                    logger.info("Saving embeddings manually to avoid encoding issues...")
                    # Save a placeholder for embeddings
                    manual_embedding_file = os.path.join(embeddings_dir, "embeddings_placeholder.json")
                    placeholder = {"status": "Error generating embeddings due to encoding issues"}
                    with open(manual_embedding_file, 'w', encoding='utf-8') as f:
                        json.dump(placeholder, f)
            
            # Close connection
            embeddings.close()
            
            logger.info(f"Embeddings processing completed. Results saved to {embeddings_dir}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            print(f"Error generating embeddings: {e}")
            return 1
    else:
        logger.info("Skipping embedding generation")
    
    logger.info("Email Event Extraction Pipeline completed successfully")
    
    print("\nEmail Event Extraction Pipeline completed successfully!")
    print(f"Event data has been added to the Neo4j graph database")
    if not args.skip_embeddings:
        print(f"Knowledge graph embeddings have been processed and saved to {embeddings_dir}")
    print("\nYou can now use the event query module to explore events:")
    print("  python event_query.py --help")
    
    return 0

def extract_triples_safely(embeddings_obj):
    """
    Extract triples from the graph with encoding validation.
    
    Args:
        embeddings_obj: GraphEmbeddings instance
        
    Returns:
        Tuple of (triples, entity_to_idx, relation_to_idx)
    """
    try:
        # First try the normal extraction
        triples, entity_to_idx, relation_to_idx = embeddings_obj.extract_triples()
        
        # Validate and clean entity and relation names
        safe_entity_to_idx = {}
        for entity, idx in entity_to_idx.items():
            safe_entity = sanitize_text(str(entity))
            safe_entity_to_idx[safe_entity] = idx
            
        safe_relation_to_idx = {}
        for relation, idx in relation_to_idx.items():
            safe_relation = sanitize_text(str(relation))
            safe_relation_to_idx[safe_relation] = idx
            
        # Convert triples to safe versions
        safe_triples = []
        for head, rel, tail in triples:
            safe_head = sanitize_text(str(head))
            safe_rel = sanitize_text(str(rel))
            safe_tail = sanitize_text(str(tail))
            safe_triples.append((safe_head, safe_rel, safe_tail))
            
        return safe_triples, safe_entity_to_idx, safe_relation_to_idx
    except UnicodeEncodeError:
        # If we hit encoding issues, use a more aggressive approach
        logger.warning("Encoding issues detected during triple extraction, using ASCII-only mode")
        
        # Try to get the raw triples from Neo4j
        try:
            with embeddings_obj.driver.session() as session:
                result = session.run("""
                MATCH (s)-[r]->(o)
                RETURN 
                    COALESCE(s.name, s.email, s.subject, toString(id(s))) as subject, 
                    type(r) as predicate, 
                    COALESCE(o.name, o.email, o.subject, toString(id(o))) as object
                LIMIT 10000
                """)
                
                safe_triples = []
                entities = set()
                relations = set()
                
                for record in result:
                    subj = sanitize_text(record["subject"])
                    pred = sanitize_text(record["predicate"])
                    obj = sanitize_text(record["object"])
                    
                    entities.add(subj)
                    relations.add(pred)
                    entities.add(obj)
                    
                    safe_triples.append((subj, pred, obj))
                
                # Create mappings
                entity_list = sorted(list(entities))
                entity_to_idx = {entity: idx for idx, entity in enumerate(entity_list)}
                
                relation_list = sorted(list(relations))
                relation_to_idx = {relation: idx for idx, relation in enumerate(relation_list)}
                
                return safe_triples, entity_to_idx, relation_to_idx
        except Exception as e:
            logger.error(f"Error during safe triple extraction: {e}")
            return [], {}, {}

def process_all_emails(event_extractor, graph_enhancer):
    """
    Process all emails in the Neo4j database and extract events.
    
    Args:
        event_extractor: EventExtractor instance
        graph_enhancer: EventGraphEnhancer instance
        
    Returns:
        Number of events extracted and added to the graph
    """
    event_count = 0
    
    # First, create the event schema
    graph_enhancer.create_event_schema()
    
    with graph_enhancer.driver.session() as session:
        # Get count of emails
        count_result = session.run("MATCH (e:Email) RETURN count(e) as count")
        total_emails = count_result.single()["count"]
        logger.info(f"Found {total_emails} emails to process")
        
        # Process in batches
        batch_size = 100
        
        for offset in range(0, total_emails, batch_size):
            batch_count = min(batch_size, total_emails - offset)
            logger.info(f"Processing batch {offset//batch_size + 1} (emails {offset+1}-{offset+batch_count})")
            
            # Get batch of emails
            query = """
            MATCH (e:Email)
            WHERE NOT EXISTS((e)-[:CONTAINS_EVENT]->())
            OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
            OPTIONAL MATCH (receiver:Person)-[:RECEIVED]->(e)
            OPTIONAL MATCH (cc:Person)-[:COPIED]->(e)
            RETURN e.message_id as message_id,
                   e.subject as subject,
                   e.date as date,
                   e.body as body,
                   collect(DISTINCT {name: sender.name, email: sender.email}) as from,
                   collect(DISTINCT {name: receiver.name, email: receiver.email}) as to,
                   collect(DISTINCT {name: cc.name, email: cc.email}) as cc
            ORDER BY e.date
            SKIP $offset
            LIMIT $limit
            """
            
            result = session.run(query, offset=offset, limit=batch_size)
            
            # Process each email
            batch_event_count = 0
            for record in result:
                email_data = dict(record)
                
                # Skip if message_id is missing
                if not email_data.get('message_id'):
                    continue
                
                # Extract events
                try:
                    events = event_extractor.extract_events_from_email(email_data)
                    
                    # Add events to graph
                    for event in events:
                        event_id = graph_enhancer.add_event_to_graph(event, email_data['message_id'])
                        if event_id:
                            batch_event_count += 1
                except Exception as e:
                    logger.error(f"Error processing email {email_data.get('message_id')}: {e}")
                    continue
            
            event_count += batch_event_count
            logger.info(f"Extracted {batch_event_count} events from this batch")
    
    return event_count

if __name__ == "__main__":
    exit(main()) 