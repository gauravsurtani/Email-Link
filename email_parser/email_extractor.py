"""
Email extractor module containing functions for parsing email components.
"""

import re
import email
from email.header import decode_header
from email.utils import parseaddr, getaddresses

def decode_str(encoded_str):
    """
    Decode encoded email header strings.
    
    Args:
        encoded_str: The encoded header string
        
    Returns:
        Decoded string
    """
    if encoded_str is None:
        return ""
        
    decoded_parts = decode_header(encoded_str)
    return ''.join([
        part.decode(encoding or 'utf-8', errors='replace') if isinstance(part, bytes) else part
        for part, encoding in decoded_parts
    ])

def extract_email_addresses(header_value):
    """
    Extract email addresses from header fields like From, To, Cc.
    
    Args:
        header_value: The header string containing email addresses
        
    Returns:
        List of dictionaries with 'name' and 'email' keys
    """
    if not header_value:
        return []
    
    # Parse addresses from the header
    addresses = getaddresses([header_value])
    return [{'name': name, 'email': addr.lower()} for name, addr in addresses if addr]

def get_email_body(message):
    """
    Extract the text content from an email message.
    Tries to get plain text first, then falls back to HTML.
    
    Args:
        message: The email.message.Message object
        
    Returns:
        String containing the email body text
    """
    text_content = ""
    html_content = ""
    
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            # Skip attachments
            if "attachment" in content_disposition:
                continue
                
            if content_type == "text/plain":
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    text_content = part.get_payload(decode=True).decode(charset, errors='replace')
                except Exception:
                    try:
                        text_content = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    except Exception:
                        pass
                    
            if content_type == "text/html":
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    html_content = part.get_payload(decode=True).decode(charset, errors='replace')
                except Exception:
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    except Exception:
                        pass
    else:
        # Not multipart - get the content directly
        try:
            charset = message.get_content_charset() or 'utf-8'
            content = message.get_payload(decode=True).decode(charset, errors='replace')
            
            if message.get_content_type() == "text/plain":
                text_content = content
            elif message.get_content_type() == "text/html":
                html_content = content
        except Exception:
            try:
                content = message.get_payload(decode=True).decode('utf-8', errors='replace')
                if message.get_content_type() == "text/plain":
                    text_content = content
                elif message.get_content_type() == "text/html":
                    html_content = content
            except Exception:
                pass
    
    # Prefer text content over HTML
    return text_content if text_content else html_content
    
def extract_attachments(message):
    """
    Extract attachment information from an email message.
    
    Args:
        message: The email.message.Message object
        
    Returns:
        List of dictionaries with attachment metadata
    """
    attachments = []
    
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == 'multipart':
                continue
                
            filename = part.get_filename()
            if filename:
                content_type = part.get_content_type()
                size = len(part.get_payload(decode=True))
                attachments.append({
                    'filename': decode_str(filename),
                    'content_type': content_type,
                    'size': size
                })
                
    return attachments

def extract_references(message):
    """
    Extract message references and in-reply-to headers.
    
    Args:
        message: The email.message.Message object
        
    Returns:
        Dictionary with 'references' and 'in_reply_to' keys
    """
    references = message.get('References', '')
    in_reply_to = message.get('In-Reply-To', '')
    
    # Split references into individual message IDs
    reference_ids = re.findall(r'<([^<>]+)>', references) if references else []
    reply_to_id = re.findall(r'<([^<>]+)>', in_reply_to) if in_reply_to else []
    
    return {
        'references': reference_ids,
        'in_reply_to': reply_to_id[0] if reply_to_id else None
    } 