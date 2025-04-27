#!/usr/bin/env python3
"""
Setup script for Email Event Extraction system.
This script helps set up the environment for running the event extraction system.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Set up the environment for the email event extraction system."""
    print("Setting up Email Event Extraction System...")
    
    # Create necessary directories
    print("Creating necessary directories...")
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/embeddings").mkdir(exist_ok=True)
    
    # Install dependencies
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError:
        print("Error installing dependencies. Please install manually using:")
        print("  pip install -r requirements.txt")
        return 1
    
    # Download spaCy model
    print("Downloading spaCy language model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    except subprocess.CalledProcessError:
        print("Error downloading spaCy model. Please download manually using:")
        print("  python -m spacy download en_core_web_md")
    
    # Check if .env file exists, create if not
    if not Path(".env").exists():
        print("Creating default .env file...")
        with open(".env", "w") as f:
            f.write("""NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=12345678
OUTPUT_DIR=./output
TEMP_DIR=./temp
LOG_LEVEL=INFO
""")
        print("Created default .env file. Please edit it with your Neo4j credentials.")
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Make sure Neo4j is running")
    print("2. Place your MBOX files in the 'input' directory")
    print("3. Process the emails with: python main.py --process-all")
    print("4. Extract events with: python event_pipeline.py")
    print("5. Query events with: python event_query.py --help")
    
    return 0

if __name__ == "__main__":
    exit(main()) 