"""
MBOX parser module for extracting email data from Gmail Takeout files.
"""

import os
import json
import mailbox
import email.utils
import logging
from datetime import datetime
from tqdm import tqdm
from .email_extractor import (
    decode_str, 
    extract_email_addresses, 
    get_email_body, 
    extract_attachments,
    extract_references
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_date(date_str):
    """
    Parse date string to ISO format.
    
    Args:
        date_str: Email date string
        
    Returns:
        ISO formatted date string or original string if parsing fails
    """
    if not date_str:
        return None
        
    try:
        # Parse the date string to a datetime object
        dt_tuple = email.utils.parsedate_tz(date_str)
        if dt_tuple:
            # Convert to timestamp, adjusting for timezone
            timestamp = email.utils.mktime_tz(dt_tuple)
            dt = datetime.fromtimestamp(timestamp)
            return dt.isoformat()
    except Exception as e:
        logger.warning(f"Error parsing date '{date_str}': {e}")
    
    return date_str

def parse_mbox_to_json(mbox_file, output_file=None, limit=None):
    """
    Parse an MBOX file and extract email data into JSON-like format.
    
    Args:
        mbox_file: Path to the MBOX file
        output_file: Optional path to save JSON output
        limit: Optional maximum number of emails to process
        
    Returns:
        List of dictionaries with email data
    """
    logger.info(f"Opening MBOX file: {mbox_file}")
    mbox = mailbox.mbox(mbox_file)
    emails = []
    
    total_messages = len(mbox)
    limit = limit or total_messages
    logger.info(f"Processing {limit} of {total_messages} emails")
    
    for i, message in enumerate(tqdm(mbox, total=min(total_messages, limit))):
        if i >= limit:
            break
            
        try:
            # Extract basic email attributes
            msg_id = message.get('Message-ID', '')
            if msg_id:
                # Clean up message ID (remove < > if present)
                msg_id = msg_id.strip().strip('<>') 
            
            subject = decode_str(message.get('Subject', ''))
            from_header = decode_str(message.get('From', ''))
            to_header = decode_str(message.get('To', ''))
            cc_header = decode_str(message.get('Cc', ''))
            bcc_header = decode_str(message.get('Bcc', ''))
            date_str = message.get('Date', '')
            
            # Extract Gmail-specific headers
            thread_id = message.get('X-GM-THRID', '')
            labels = message.get('X-Gmail-Labels', '')
            
            # Parse and normalize email addresses
            from_list = extract_email_addresses(from_header)
            to_list = extract_email_addresses(to_header)
            cc_list = extract_email_addresses(cc_header)
            bcc_list = extract_email_addresses(bcc_header)
            
            # Get email body
            body = get_email_body(message)
            
            # Get attachments
            attachments = extract_attachments(message)
            
            # Get references and in-reply-to
            refs = extract_references(message)
            
            # Create structured email object
            email_obj = {
                'message_id': msg_id,
                'thread_id': thread_id,
                'subject': subject,
                'date': parse_date(date_str),
                'raw_date': date_str,
                'labels': labels.split(',') if labels else [],
                'from': from_list,
                'to': to_list,
                'cc': cc_list,
                'bcc': bcc_list,
                'body': body,
                'attachments': attachments,
                'has_attachments': len(attachments) > 0,
                'references': refs['references'],
                'in_reply_to': refs['in_reply_to']
            }
            
            emails.append(email_obj)
        except Exception as e:
            logger.error(f"Error processing email at index {i}: {e}")
    
    logger.info(f"Completed processing {len(emails)} emails")
    
    # Save to JSON file if output file is specified
    if output_file:
        logger.info(f"Saving emails to {output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(emails, f, ensure_ascii=False, indent=2)
    
    return emails 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse MBOX file and extract email data to JSON.")
    parser.add_argument('--input', required=True, help='Path to the input MBOX file')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    parser.add_argument('--limit', type=int, default=None, help='Optional: maximum number of emails to process')
    args = parser.parse_args()

    parse_mbox_to_json(args.input, args.output, args.limit) 