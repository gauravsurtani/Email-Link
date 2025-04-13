"""
Configuration module for the Email Graph Analysis System.
Handles loading configuration from environment variables and defaults.
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    'neo4j_uri': 'bolt://localhost:7687',
    'neo4j_user': 'neo4j',
    'neo4j_password': None,
    'output_dir': './output',
    'temp_dir': './temp',
    'log_level': 'INFO',
}

def get_config():
    """
    Load configuration from environment variables with defaults.
    Returns a dictionary with configuration values.
    """
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables if present
    config['neo4j_uri'] = os.getenv('NEO4J_URI', config['neo4j_uri'])
    config['neo4j_user'] = os.getenv('NEO4J_USER', config['neo4j_user'])
    config['neo4j_password'] = os.getenv('NEO4J_PASSWORD', config['neo4j_password'])
    config['output_dir'] = os.getenv('OUTPUT_DIR', config['output_dir'])
    config['temp_dir'] = os.getenv('TEMP_DIR', config['temp_dir'])
    config['log_level'] = os.getenv('LOG_LEVEL', config['log_level'])
    
    return config

def parse_args():
    """Parse command line arguments and update configuration."""
    parser = argparse.ArgumentParser(description='Process Gmail Takeout MBOX file into Neo4j graph database')
    
    # Input file argument
    parser.add_argument('--mbox', required=True, help='Path to the MBOX file')
    
    # Processing options
    parser.add_argument('--skip-parsing', action='store_true', 
                        help='Skip MBOX parsing and use existing parsed data')
    parser.add_argument('--output-json', 
                        help='Output file for parsed email JSON data')
    
    # Neo4j connection options
    parser.add_argument('--neo4j-uri', help='Neo4j URI')
    parser.add_argument('--neo4j-user', help='Neo4j username')
    parser.add_argument('--neo4j-password', help='Neo4j password')
    
    # Output options
    parser.add_argument('--output-dir', help='Directory for output files')
    parser.add_argument('--temp-dir', help='Directory for temporary files')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        help='Logging level')
    
    return parser.parse_args()

def get_combined_config():
    """
    Combine environment variable config and command line arguments.
    Command line arguments take precedence over environment variables.
    """
    config = get_config()
    args = parse_args()
    
    # Update config with command line arguments if provided
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None:
            config_key = key.replace('-', '_')
            config[config_key] = value
    
    # Create directories if they don't exist
    Path(config['output_dir']).mkdir(exist_ok=True, parents=True)
    Path(config['temp_dir']).mkdir(exist_ok=True, parents=True)
    
    return config 