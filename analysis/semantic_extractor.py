"""
Semantic Data Extraction module - Extract structured semantic data from emails.

This module provides functions to extract entities, actions, and other semantic information
from emails using LLMs (OpenAI) with fallback to spaCy for basic NER.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import spacy
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model for fallback NER
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load spaCy model: {e}")
    nlp = None


class SemanticExtractor:
    """
    Extract semantic data from emails using LLMs with spaCy fallback.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the semantic extractor.
        
        Args:
            api_key: OpenAI API key (if None, load from environment)
            model: LLM model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found. LLM extraction will be unavailable.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.prompt_template = """
Extract the following from this email:
- Email type (job application, statement, booking, receipt, etc.)
- Organizations (companies, institutions, etc.)
- People (names of individuals)
- Products (items, services mentioned)
- Locations (cities, countries, addresses)
- Monetary amounts (prices, costs, fees)
- Dates (deadlines, appointments, scheduled events)
- Actions (applied, paid, booked, etc.)

Return the result as structured JSON with this exact format:
{
  "type": "string",
  "organizations": ["string"],
  "people": ["string"],
  "products": ["string"],
  "locations": ["string"],
  "amounts": ["string"],
  "dates": ["string"],
  "actions": ["string"]
}

Email:
"""
        self.retry_limit = 3
        self.retry_delay = 2  # seconds

    def extract_from_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract semantic data from a single email.
        
        Args:
            email_data: Dictionary containing email metadata and content
            
        Returns:
            Dictionary with extracted semantic data
        """
        # Prepare email content
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        
        email_content = f"Subject: {subject}\n\n{body}"
        
        # Try LLM extraction first
        if self.client:
            for attempt in range(self.retry_limit):
                try:
                    extracted_data = self._extract_with_llm(email_content)
                    if extracted_data:
                        return extracted_data
                except Exception as e:
                    logger.warning(f"LLM extraction failed (attempt {attempt+1}/{self.retry_limit}): {e}")
                    time.sleep(self.retry_delay)
        
        # Fall back to spaCy if LLM fails or isn't available
        logger.info("Falling back to spaCy NER")
        return self._extract_with_spacy(email_content)
    
    def _extract_with_llm(self, email_content: str) -> Dict[str, Any]:
        """
        Extract semantic data using LLM.
        
        Args:
            email_content: Email content (subject + body)
            
        Returns:
            Dictionary with extracted semantic data
        """
        prompt = f"{self.prompt_template}\n\"\"\"\n{email_content}\n\"\"\""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise entity extraction assistant. Extract structured data from emails and format it exactly as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=800
            )
            
            content = response.choices[0].message.content
            
            # Find JSON in the response
            import re
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from LLM response: {e}")
            else:
                logger.warning("No JSON found in LLM response")
                
            return None
        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            raise
    
    def _extract_with_spacy(self, email_content: str) -> Dict[str, Any]:
        """
        Extract basic entities using spaCy as fallback.
        
        Args:
            email_content: Email content (subject + body)
            
        Returns:
            Dictionary with extracted semantic data (more limited than LLM)
        """
        if not nlp:
            logger.warning("spaCy model not available for fallback")
            return {
                "type": "unknown",
                "organizations": [],
                "people": [],
                "products": [],
                "locations": [],
                "amounts": [],
                "dates": [],
                "actions": []
            }
        
        doc = nlp(email_content)
        
        # Extract entities
        organizations = [ent.text for ent in doc.ents if ent.label_ in ["ORG"]]
        people = [ent.text for ent in doc.ents if ent.label_ in ["PERSON"]]
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        dates = [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
        amounts = [ent.text for ent in doc.ents if ent.label_ in ["MONEY"]]
        
        # Extract verbs as potential actions
        actions = [token.lemma_ for token in doc if token.pos_ == "VERB"][:5]  # Limit to first 5 verbs
        
        # Try to guess email type based on keywords
        email_type = "unknown"
        type_keywords = {
            "job application": ["job", "application", "resume", "cv", "position", "hiring", "interview"],
            "receipt": ["receipt", "invoice", "payment", "order", "purchase", "transaction"],
            "booking": ["booking", "reservation", "hotel", "flight", "ticket"],
            "statement": ["statement", "bill", "account", "balance", "due", "payment"]
        }
        
        email_lower = email_content.lower()
        for t_type, keywords in type_keywords.items():
            if any(keyword in email_lower for keyword in keywords):
                email_type = t_type
                break
        
        return {
            "type": email_type,
            "organizations": organizations,
            "people": people,
            "products": [],  # spaCy doesn't have a good way to identify products
            "locations": locations,
            "amounts": amounts,
            "dates": dates,
            "actions": actions
        }


def batch_process_emails(emails: List[Dict[str, Any]], 
                         output_dir: Union[str, Path],
                         batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Process a batch of emails for semantic data extraction.
    
    Args:
        emails: List of email data dictionaries
        output_dir: Directory to save the processed results
        batch_size: Number of emails to process in each batch
        
    Returns:
        List of email dictionaries with added semantic data
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = SemanticExtractor()
    
    # Process emails in batches
    enriched_emails = []
    
    for i in tqdm(range(0, len(emails), batch_size), desc="Processing email batches"):
        batch = emails[i:i+batch_size]
        
        for email in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
            try:
                # Extract semantic data
                semantic_data = extractor.extract_from_email(email)
                
                # Add to email
                enriched_email = email.copy()
                enriched_email['entities'] = semantic_data
                enriched_emails.append(enriched_email)
                
                # Save individual email
                if 'message_id' in email:
                    email_id = email['message_id']
                    # Clean up message ID for filename
                    clean_id = "".join(c if c.isalnum() else "_" for c in email_id)
                    output_file = output_dir / f"email_{clean_id}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(enriched_email, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error processing email: {e}")
                continue
        
        # Save batch progress
        batch_output = output_dir / f"processed_batch_{i//batch_size + 1}.json"
        with open(batch_output, 'w', encoding='utf-8') as f:
            json.dump(enriched_emails, f, indent=2, ensure_ascii=False)
        
        # Wait between batches to avoid API rate limits
        if i + batch_size < len(emails):
            time.sleep(1)
    
    # Save complete dataset
    complete_output = output_dir / "processed_emails_complete.json"
    with open(complete_output, 'w', encoding='utf-8') as f:
        json.dump(enriched_emails, f, indent=2, ensure_ascii=False)
    
    return enriched_emails


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract semantic data from emails")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Path to input JSON file containing parsed emails")
    parser.add_argument("--output", "-o", type=str, default="output/semantic_data",
                        help="Output directory for processed emails")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Number of emails to process in each batch")
    
    args = parser.parse_args()
    
    try:
        # Load input data
        with open(args.input, 'r', encoding='utf-8') as f:
            emails = json.load(f)
        
        if not isinstance(emails, list):
            logger.error("Input file must contain a list of email objects")
            sys.exit(1)
        
        logger.info(f"Loaded {len(emails)} emails from {args.input}")
        
        # Process emails
        enriched_emails = batch_process_emails(
            emails=emails,
            output_dir=args.output,
            batch_size=args.batch_size
        )
        
        logger.info(f"Successfully processed {len(enriched_emails)} emails")
        logger.info(f"Results saved to {args.output}")
        
    except Exception as e:
        logger.error(f"Error in semantic extraction process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 