#!/usr/bin/env python3
"""
Script to process all MBOX files in the input directory.
This will iterate through all .mbox files and process each one.
"""

import os
import subprocess
import glob
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Process all MBOX files in the input directory."""
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
    
    for i, mbox_file in enumerate(mbox_files, 1):
        file_name = os.path.basename(mbox_file)
        print(f"\n[{i}/{len(mbox_files)}] Processing: {file_name}")
        logger.info(f"Processing file {i}/{len(mbox_files)}: {mbox_file}")
        
        # Call main.py for each file
        try:
            cmd = ["python", "main.py", "--mbox", mbox_file]
            result = subprocess.run(cmd, check=True)
            
            if result.returncode != 0:
                logger.warning(f"Process returned non-zero exit code: {result.returncode}")
                print(f"Warning: Process completed with exit code {result.returncode}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing {mbox_file}: {e}")
            print(f"Error: Failed to process {file_name}")
            print(f"Subprocess error: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"Unexpected error: {e}")
            continue
            
        print(f"✓ Completed processing: {file_name}")
    
    print("\nAll MBOX files have been processed!")
    print("You can now explore your email data using the interactive agent:")
    print("  python -m agent.interactive")
    
    return 0

if __name__ == "__main__":
    exit(main()) 