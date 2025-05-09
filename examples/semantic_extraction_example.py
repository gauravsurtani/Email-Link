#!/usr/bin/env python3
"""
Example script for extracting semantic data from a sample email.

This example demonstrates the semantic extraction process:
1. Create a sample email
2. Extract semantic data using LLM
3. Display the extracted entities and relationships
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path if running as script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the semantic extractor
from analysis.semantic_extractor import SemanticExtractor

# Create a sample email
sample_email = {
    "message_id": "sample123@example.com",
    "subject": "Job Application for Software Engineering Position at TechCorp",
    "body": """Dear Hiring Manager,

I am writing to apply for the Software Engineering position at TechCorp that I saw advertised on LinkedIn. With my 5 years of experience in Python development and machine learning, I believe I would be a great fit for your team.

I have attached my resume and portfolio for your review. My salary expectation is in the range of $120,000 to $140,000 per year.

I am available for an interview starting next Monday (June 10th, 2024). Please let me know if you need any additional information.

Best regards,
John Smith
john.smith@example.com
(555) 123-4567
"""
}

def main():
    """Run the semantic extraction example."""
    print("Semantic Data Extraction Example")
    print("================================\n")
    
    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the semantic extractor
    extractor = SemanticExtractor()
    
    print(f"Sample Email:\n")
    print(f"Subject: {sample_email['subject']}")
    print(f"Body:\n{sample_email['body']}\n")
    
    print("Extracting semantic data...")
    semantic_data = extractor.extract_from_email(sample_email)
    
    print("\nExtracted Semantic Data:")
    print("=======================")
    print(f"Email Type: {semantic_data['type']}")
    print(f"\nOrganizations: {', '.join(semantic_data['organizations'])}")
    print(f"People: {', '.join(semantic_data['people'])}")
    print(f"Products: {', '.join(semantic_data['products'])}")
    print(f"Locations: {', '.join(semantic_data['locations'])}")
    print(f"Amounts: {', '.join(semantic_data['amounts'])}")
    print(f"Dates: {', '.join(semantic_data['dates'])}")
    print(f"Actions: {', '.join(semantic_data['actions'])}")
    
    # Save the enriched email
    enriched_email = sample_email.copy()
    enriched_email['entities'] = semantic_data
    
    output_file = output_dir / "sample_enriched_email.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_email, f, indent=2)
    
    print(f"\nEnriched email saved to {output_file}")
    
    print("\nHow This Works:")
    print("==============")
    print("1. The email is sent to an LLM (OpenAI GPT model) with a carefully crafted prompt")
    print("2. The LLM extracts entities, actions, dates, amounts, and email type")
    print("3. The extracted data is structured as JSON for use in applications")
    print("4. If the LLM extraction fails, we fall back to spaCy for basic NER")
    print("5. The enriched email can be used to build a semantic knowledge graph")
    
if __name__ == "__main__":
    main() 