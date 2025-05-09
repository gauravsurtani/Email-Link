#!/usr/bin/env python3
"""
Main execution script for Email Graph Analysis System.
This script orchestrates the complete workflow:
1. Parse Gmail Takeout MBOX file
2. Extract semantic data from emails (optional)
3. Load data into Neo4j graph database
4. Initialize the agent for analysis

Can process a single MBOX file or all MBOX files in the input directory.
"""

import os
import json
import logging
import glob
import argparse
from pathlib import Path

# Import configuration
from config import get_config

# Import modules
from email_parser.mbox_parser import parse_mbox_to_json
from graph_db.loader import EmailGraphLoader
from agent.agent import EmailGraphAgent
from analysis.semantic_extractor import batch_process_emails

def setup_logging(level):
    """Configure logging for the application."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def process_single_mbox(config):
    """Process a single MBOX file based on configuration."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing MBOX file: {config['mbox']}")

    # Validate required parameters
    if not config.get('mbox'):
        logger.error("Missing required MBOX file path")
        print("Error: Missing required MBOX file path. Use --mbox to specify.")
        return 1
        
    if not config.get('neo4j_password'):
        logger.error("Missing Neo4j password")
        print("Error: Neo4j password not provided in .env file or command line.")
        return 1
    
    # Step 1: Parse MBOX file (if not skipped)
    if not config.get('skip_parsing'):
        logger.info(f"Parsing MBOX file: {config['mbox']}")
        output_json = config.get('output_json') or os.path.join(config['output_dir'], 'parsed_emails.json')
        
        try:
            emails = parse_mbox_to_json(config['mbox'], output_json)
            logger.info(f"Parsed {len(emails)} emails to {output_json}")
        except Exception as e:
            logger.error(f"Error parsing MBOX file: {e}")
            print(f"Error parsing MBOX file: {e}")
            return 1
    else:
        # If parsing is skipped, ensure the JSON file exists
        output_json = config.get('output_json') or os.path.join(config['output_dir'], 'parsed_emails.json')
        if not os.path.exists(output_json):
            logger.error(f"Skipping parsing but output JSON file does not exist: {output_json}")
            print(f"Error: Cannot skip parsing - output JSON file not found: {output_json}")
            return 1
        logger.info(f"Skipping parsing, using existing file: {output_json}")
    
    # Step 2 (Optional): Extract semantic data
    if config.get('extract_semantic'):
        logger.info("Extracting semantic data from emails")
        semantic_output_dir = os.path.join(config['output_dir'], 'semantic_data')
        
        try:
            # Load the parsed emails
            with open(output_json, 'r', encoding='utf-8') as f:
                emails = json.load(f)
            
            # Determine batch size for semantic extraction
            batch_size = config.get('semantic_batch_size', 10)
            
            # Process a subset if in test mode
            if config.get('semantic_test_mode'):
                logger.info("Running semantic extraction in test mode with a small subset")
                emails = emails[:5]
            
            # Extract semantic data
            enriched_emails = batch_process_emails(
                emails=emails,
                output_dir=semantic_output_dir,
                batch_size=batch_size
            )
            
            logger.info(f"Extracted semantic data from {len(enriched_emails)} emails")
            print(f"Extracted semantic data from {len(enriched_emails)} emails")
            print(f"Results saved to {semantic_output_dir}")
            
        except Exception as e:
            logger.error(f"Error extracting semantic data: {e}")
            print(f"Error extracting semantic data: {e}")
            if not config.get('continue_on_error'):
                return 1
    
    # Step 3: Load into Neo4j
    logger.info(f"Loading emails into Neo4j at {config['neo4j_uri']}")
    
    try:
        loader = EmailGraphLoader(
            config['neo4j_uri'], 
            config['neo4j_user'], 
            config['neo4j_password']
        )
        
        # Create database schema
        loader.setup_database()
        
        # Load emails
        success_count = loader.load_emails_from_json(output_json)
        logger.info(f"Successfully loaded {success_count} emails into graph database")
        
        # If loading semantic data separately
        if config.get('load_semantic_data'):
            semantic_dir = config.get('semantic_data_dir') or os.path.join(config['output_dir'], 'semantic_data')
            
            if os.path.exists(semantic_dir):
                logger.info(f"Loading semantic data from {semantic_dir}")
                semantic_count = loader.load_semantic_data(semantic_dir)
                logger.info(f"Successfully loaded semantic data for {semantic_count} emails")
                print(f"Successfully loaded semantic data for {semantic_count} emails")
            else:
                logger.warning(f"Semantic data directory not found: {semantic_dir}")
                print(f"Warning: Semantic data directory not found: {semantic_dir}")
        
        # Close Neo4j connection
        loader.close()
    except Exception as e:
        logger.error(f"Error loading data into Neo4j: {e}")
        print(f"Error loading data into Neo4j: {e}")
        return 1
    
    logger.info("Email graph database successfully created!")
    print("\nEmail graph database successfully created!")
    print(f"Loaded {success_count} emails into Neo4j")
    print("\nYou can now use the Email Graph Agent to analyze your data.")
    print("Example: python -m agent.interactive")
    
    return 0

def process_all_mbox_files(config):
    """Process all MBOX files in the input directory."""
    logger = logging.getLogger(__name__)
    
    # Create input directory if it doesn't exist
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    # Find all .mbox files in the input directory
    mbox_files = glob.glob(str(input_dir / "*.mbox"))
    
    if not mbox_files:
        logger.error("No .mbox files found in the input directory.")
        print("Error: No .mbox files found in the 'input' directory.")
        print("Please place your Gmail Takeout .mbox files in the 'input' directory.")
        return 1
    
    # Process each .mbox file
    logger.info(f"Found {len(mbox_files)} .mbox files to process")
    print(f"Found {len(mbox_files)} .mbox files to process:")
    
    total_emails_loaded = 0
    all_parsed_emails = []
    
    for i, mbox_file in enumerate(mbox_files, 1):
        file_name = os.path.basename(mbox_file)
        print(f"\n[{i}/{len(mbox_files)}] Processing: {file_name}")
        logger.info(f"Processing file {i}/{len(mbox_files)}: {mbox_file}")
        
        # Create new config for this file
        file_config = config.copy()
        file_config['mbox'] = mbox_file
        file_config['output_json'] = os.path.join(config['output_dir'], f"{file_name}.json")
        
        try:
            # Step 1: Parse MBOX file
            logger.info(f"Parsing MBOX file: {mbox_file}")
            emails = parse_mbox_to_json(mbox_file, file_config['output_json'])
            logger.info(f"Parsed {len(emails)} emails to {file_config['output_json']}")
            all_parsed_emails.extend(emails)
            
            # Step 2: Load into Neo4j
            logger.info(f"Loading emails into Neo4j at {file_config['neo4j_uri']}")
            
            loader = EmailGraphLoader(
                file_config['neo4j_uri'], 
                file_config['neo4j_user'], 
                file_config['neo4j_password']
            )
            
            # Create database schema (only needed for first file)
            if i == 1:
                loader.setup_database()
            
            # Load emails
            success_count = loader.load_emails_from_json(file_config['output_json'])
            total_emails_loaded += success_count
            
            # Close Neo4j connection
            loader.close()
            
            logger.info(f"Successfully loaded {success_count} emails from {file_name}")
            print(f"✓ Completed processing: {file_name} - Loaded {success_count} emails")
            
        except Exception as e:
            logger.error(f"Error processing {mbox_file}: {e}")
            print(f"Error: Failed to process {file_name}")
            print(f"Error details: {e}")
            if not config.get('continue_on_error'):
                return 1
    
    # Save all parsed emails to a combined file
    combined_output = os.path.join(config['output_dir'], 'all_parsed_emails.json')
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_parsed_emails, f, indent=2)
    logger.info(f"Saved all {len(all_parsed_emails)} emails to {combined_output}")
    
    # Step 2 (Optional): Extract semantic data from all emails
    if config.get('extract_semantic') and all_parsed_emails:
        logger.info("Extracting semantic data from all emails")
        semantic_output_dir = os.path.join(config['output_dir'], 'semantic_data')
        
        try:
            # Determine batch size for semantic extraction
            batch_size = config.get('semantic_batch_size', 10)
            
            # Process a subset if in test mode
            if config.get('semantic_test_mode'):
                logger.info("Running semantic extraction in test mode with a small subset")
                all_parsed_emails = all_parsed_emails[:5]
            
            # Extract semantic data
            enriched_emails = batch_process_emails(
                emails=all_parsed_emails,
                output_dir=semantic_output_dir,
                batch_size=batch_size
            )
            
            logger.info(f"Extracted semantic data from {len(enriched_emails)} emails")
            print(f"Extracted semantic data from {len(enriched_emails)} emails")
            print(f"Results saved to {semantic_output_dir}")
            
        except Exception as e:
            logger.error(f"Error extracting semantic data: {e}")
            print(f"Error extracting semantic data: {e}")
            if not config.get('continue_on_error'):
                return 1
    
    # If loading semantic data separately
    if config.get('load_semantic_data'):
        logger.info("Loading semantic data into Neo4j")
        semantic_dir = config.get('semantic_data_dir') or os.path.join(config['output_dir'], 'semantic_data')
        
        if os.path.exists(semantic_dir):
            try:
                loader = EmailGraphLoader(
                    config['neo4j_uri'], 
                    config['neo4j_user'], 
                    config['neo4j_password']
                )
                
                logger.info(f"Loading semantic data from {semantic_dir}")
                semantic_count = loader.load_semantic_data(semantic_dir)
                
                logger.info(f"Successfully loaded semantic data for {semantic_count} emails")
                print(f"Successfully loaded semantic data for {semantic_count} emails")
                
                # Close Neo4j connection
                loader.close()
            except Exception as e:
                logger.error(f"Error loading semantic data into Neo4j: {e}")
                print(f"Error loading semantic data into Neo4j: {e}")
                if not config.get('continue_on_error'):
                    return 1
        else:
            logger.warning(f"Semantic data directory not found: {semantic_dir}")
            print(f"Warning: Semantic data directory not found: {semantic_dir}")
    
    if total_emails_loaded > 0:
        logger.info(f"Completed loading a total of {total_emails_loaded} emails from all files")
        print(f"\nAll MBOX files have been processed!")
        print(f"Total emails loaded into Neo4j: {total_emails_loaded}")
        print("\nYou can now explore your email data using the interactive agent:")
        print("  python -m agent.interactive")
        return 0
    else:
        logger.error("Failed to load any emails from the MBOX files")
        print("Error: No emails were successfully loaded into the database.")
        return 1

def load_semantic_data_only(config):
    """Load only semantic data into the Neo4j database."""
    logger = logging.getLogger(__name__)
    
    # Validate required parameters
    if not config.get('semantic_data_dir'):
        logger.error("Missing semantic data directory")
        print("Error: Semantic data directory not provided.")
        return 1
    
    if not config.get('neo4j_password'):
        logger.error("Missing Neo4j password")
        print("Error: Neo4j password not provided in .env file or command line.")
        return 1
    
    semantic_dir = config.get('semantic_data_dir')
    if not os.path.exists(semantic_dir):
        logger.error(f"Semantic data directory not found: {semantic_dir}")
        print(f"Error: Semantic data directory not found: {semantic_dir}")
        return 1
    
    try:
        logger.info(f"Connecting to Neo4j at {config['neo4j_uri']}")
        loader = EmailGraphLoader(
            config['neo4j_uri'],
            config['neo4j_user'],
            config['neo4j_password']
        )
        
        logger.info(f"Loading semantic data from {semantic_dir}")
        semantic_count = loader.load_semantic_data(semantic_dir)
        
        logger.info(f"Successfully loaded semantic data for {semantic_count} emails")
        print(f"Successfully loaded semantic data for {semantic_count} emails")
        
        # Close Neo4j connection
        loader.close()
        
        return 0
    except Exception as e:
        logger.error(f"Error loading semantic data: {e}")
        print(f"Error loading semantic data: {e}")
        return 1

def parse_command_line_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Gmail Takeout MBOX file into Neo4j graph database')
    
    # Input file argument group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--mbox', help='Path to a single MBOX file to process')
    input_group.add_argument('--process-all', action='store_true', 
                    help='Process all MBOX files in the input directory')
    input_group.add_argument('--load-semantic-only', action='store_true',
                    help='Load only semantic data into an existing graph database')
    
    # Processing options
    parser.add_argument('--skip-parsing', action='store_true', 
                    help='Skip MBOX parsing and use existing parsed data')
    parser.add_argument('--output-json', 
                    help='Output file for parsed email JSON data')
    
    # Semantic extraction options
    parser.add_argument('--extract-semantic', action='store_true',
                    help='Extract semantic data from emails using LLMs')
    parser.add_argument('--semantic-test-mode', action='store_true',
                    help='Run semantic extraction on a small subset (5 emails)')
    parser.add_argument('--semantic-batch-size', type=int, default=10,
                    help='Batch size for semantic extraction')
    parser.add_argument('--load-semantic-data', action='store_true',
                    help='Load semantic data into the graph database')
    parser.add_argument('--semantic-data-dir', 
                    help='Directory containing semantic data files')
    
    # Neo4j connection options
    parser.add_argument('--neo4j-uri', help='Neo4j URI')
    parser.add_argument('--neo4j-user', help='Neo4j username')
    parser.add_argument('--neo4j-password', help='Neo4j password')
    
    # Output options
    parser.add_argument('--output-dir', help='Directory for output files')
    parser.add_argument('--temp-dir', help='Directory for temporary files')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                    help='Logging level')
    parser.add_argument('--continue-on-error', action='store_true',
                    help='Continue processing even if errors occur')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Load environment configuration
    config = get_config()
    
    # Parse command line arguments
    args = parse_command_line_args()
    
    # Update config with command line arguments
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None:
            config_key = key.replace('-', '_')
            config[config_key] = value
    
    # Create output and temp directories
    Path(config['output_dir']).mkdir(exist_ok=True, parents=True)
    Path(config['temp_dir']).mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    setup_logging(config['log_level'])
    logger = logging.getLogger(__name__)
    logger.info("Starting Email Graph Analysis System")
    
    # Process based on command line arguments
    if config.get('load_semantic_only'):
        return load_semantic_data_only(config)
    elif config.get('process_all'):
        return process_all_mbox_files(config)
    else:
        return process_single_mbox(config)

if __name__ == "__main__":
    exit(main()) 