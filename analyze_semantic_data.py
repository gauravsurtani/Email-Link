#!/usr/bin/env python3
"""
Analyze semantic data extracted from emails.

This script analyzes the semantic data extracted from emails and generates
reports and visualizations to help understand the data.

Usage:
    python analyze_semantic_data.py --data output/semantic_data --output output/analysis
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("semantic_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the semantic analyzer
from analysis.semantic_analysis import analyze_semantic_data

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze semantic data from emails")
    parser.add_argument("--data", "-d", type=str, default="output/semantic_data",
                        help="Directory containing semantic data files")
    parser.add_argument("--output", "-o", type=str, default="output/analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Display visualization plots (requires display)")
    
    args = parser.parse_args()
    
    try:
        # Check if data directory exists
        data_dir = Path(args.data)
        if not data_dir.exists():
            logger.error(f"Data directory not found: {args.data}")
            return 1
        
        # Run analysis
        analyze_semantic_data(args.data, args.output)
        
        # Success message
        print(f"\nAnalysis complete! Results saved to {args.output}")
        
        # If visualize flag is set, import the analyzer and show some plots
        if args.visualize:
            try:
                import matplotlib.pyplot as plt
                from analysis.semantic_analysis import SemanticAnalyzer
                
                analyzer = SemanticAnalyzer(args.data)
                
                # Email types
                analyzer.visualize_email_types()
                
                # Top organizations
                analyzer.visualize_entity_counts('organizations')
                
                # Top people
                analyzer.visualize_entity_counts('people')
                
                # Top locations
                analyzer.visualize_entity_counts('locations')
                
                # Top actions
                analyzer.visualize_entity_counts('actions')
            except Exception as e:
                logger.error(f"Error in visualization: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in semantic analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 