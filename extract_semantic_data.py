#!/usr/bin/env python3
"""
Extract semantic data from parsed emails.

This script loads parsed emails from a JSON file, extracts semantic data using LLMs,
and saves the enriched email data for further processing.

Usage:
    python extract_semantic_data.py --input output/parsed_emails.json --output output/semantic_data --batch-size 10
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("semantic_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the semantic extractor
from analysis.semantic_extractor import batch_process_emails

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract semantic data from emails")
    parser.add_argument("--input", "-i", type=str, default="output/parsed_emails.json",
                        help="Path to input JSON file containing parsed emails")
    parser.add_argument("--output", "-o", type=str, default="output/semantic_data",
                        help="Output directory for processed emails")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Number of emails to process in each batch")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Run with a small test set (first 5 emails)")
    
    args = parser.parse_args()
    
    try:
        # Check if input file exists
        input_file = Path(args.input)
        if not input_file.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        # Load input data
        with open(input_file, 'r', encoding='utf-8') as f:
            emails = json.load(f)
        
        if not isinstance(emails, list):
            logger.error("Input file must contain a list of email objects")
            return 1
        
        # Use a small test set if requested
        if args.test:
            emails = emails[:5]
            logger.info(f"Running in test mode with {len(emails)} emails")
        else:
            logger.info(f"Loaded {len(emails)} emails from {args.input}")
        
        # Process emails
        enriched_emails = batch_process_emails(
            emails=emails,
            output_dir=args.output,
            batch_size=args.batch_size
        )
        
        logger.info(f"Successfully processed {len(enriched_emails)} emails")
        logger.info(f"Results saved to {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Error in semantic extraction process: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 