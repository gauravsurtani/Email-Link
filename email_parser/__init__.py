"""
Email parser module for extracting data from MBOX files.
"""

from .mbox_parser import parse_mbox_to_json
from .email_extractor import (
    decode_str, 
    extract_email_addresses, 
    get_email_body, 
    extract_attachments
)

__all__ = [
    'parse_mbox_to_json',
    'decode_str',
    'extract_email_addresses',
    'get_email_body',
    'extract_attachments'
] 