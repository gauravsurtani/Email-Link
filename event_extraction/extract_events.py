"""
Module for extracting event information from email content.
Uses NLP and rule-based techniques to identify event-related information.
"""

import re
import spacy
import dateparser
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_md")
except:
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.warning("Using smaller spaCy model en_core_web_sm. For better performance, install: python -m spacy download en_core_web_md")
    except:
        logger.warning("Could not load spaCy model. Please install it using: python -m spacy download en_core_web_sm")
        # Create a minimal processing function if spaCy is not available
        class MinimalNLP:
            def __call__(self, text):
                return type('obj', (object,), {'ents': []})
        nlp = MinimalNLP()

class EventExtractor:
    """Class for extracting events from email content."""
    
    def __init__(self):
        """Initialize the event extractor."""
        # Common event-related keywords
        self.event_keywords = [
            "meeting", "call", "conference", "webinar", "seminar", "workshop",
            "appointment", "sync", "gathering", "celebration", "ceremony",
            "party", "lunch", "dinner", "breakfast", "coffee", "catchup",
            "presentation", "interview", "discussion", "review", "planning",
            "session", "event", "meetup", "standup", "update", "sync-up"
        ]
        
        # Location-related keywords
        self.location_keywords = [
            "room", "office", "building", "floor", "suite", "street", "avenue",
            "conference room", "zoom", "meet", "teams", "hangout", "skype",
            "location", "place", "venue", "center", "centre", "restaurant",
            "cafe", "hotel", "lobby", "plaza", "area", "park", "hall"
        ]
        
        # Time-related patterns
        self.time_patterns = [
            r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b',
            r'\b\d{1,2}\s*(?:am|pm|AM|PM)\b',
            r'\b(?:1[0-2]|0?[1-9])(?::[0-5][0-9])?\s*(?:am|pm|AM|PM)\b',
            r'\b(?:from|between)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?\s*(?:to|and|until|-)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?\b'
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}?\b',
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*\d{4}?\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b',
            r'\btoday\b',
            r'\btomorrow\b',
            r'\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\bthis\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        # Virtual meeting patterns
        self.virtual_meeting_patterns = [
            r'https?://(?:www\.)?(?:zoom|teams|meet|hangout|skype)\.(?:us|com|google)/[^\s]+',
            r'zoom\s+id:?\s*\d{9,11}',
            r'meeting\s+id:?\s*\d{3}[\s-]?\d{3}[\s-]?\d{3,4}',
            r'(?:zoom|teams|meet|hangout|skype)[^\n.]*(?:link|url)',
            r'password:?\s*\d{6}',
            r'passcode:?\s*\d{6}'
        ]
        
    def extract_events_from_email(self, email_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract event information from an email.
        
        Args:
            email_data: Dictionary containing email data with 'subject' and 'body'
            
        Returns:
            List of dictionaries containing event information
        """
        events = []
        
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        date = email_data.get('date', '')
        
        # Validate inputs to prevent issues
        if not isinstance(subject, str):
            subject = str(subject) if subject is not None else ""
            
        if not isinstance(body, str):
            body = str(body) if body is not None else ""
            
        # Limit body size to prevent memory issues with large emails
        if len(body) > 50000:
            body = body[:50000] + "... [truncated]"
        
        # First, check if this email is likely to contain event information
        if not self._is_likely_event_email(subject, body):
            return events
            
        # Extract event details
        event_info = {
            'email_id': email_data.get('message_id', ''),
            'event_type': self._identify_event_type(subject, body),
            'subject': subject,
            'email_date': date,
            'from': self._format_sender(email_data.get('from', [])),
            'to': self._format_recipients(email_data.get('to', [])),
            'cc': self._format_recipients(email_data.get('cc', [])),
        }
        
        # Extract the potential event date and time
        event_info.update(self._extract_datetime_info(subject, body, date))
        
        # Extract location information
        event_info['location'] = self._extract_location(body)
        
        # Extract virtual meeting links
        event_info['virtual_meeting'] = self._extract_virtual_meeting_info(body)
        
        # Extract attendees (not just recipients)
        event_info['attendees'] = self._extract_attendees(body, email_data)
        
        # Extract calendar action (accept/decline/tentative)
        event_info['calendar_action'] = self._extract_calendar_action(body)
        
        # Check if we have enough information to consider this an event
        if self._is_valid_event(event_info):
            events.append(event_info)
            
        return events
    
    def _is_likely_event_email(self, subject: str, body: str) -> bool:
        """
        Determine if an email is likely to contain event information.
        
        Args:
            subject: Email subject
            body: Email body text
            
        Returns:
            Boolean indicating likelihood
        """
        # Check for event keywords in subject
        subject_lower = subject.lower()
        if any(keyword in subject_lower for keyword in self.event_keywords):
            return True
            
        # Check for invitation or calendar keywords in subject
        if re.search(r'invitation|calendar|invite|meeting|appointment|schedule', subject_lower):
            return True
            
        # Check for date and time patterns in subject
        for pattern in self.date_patterns + self.time_patterns:
            if re.search(pattern, subject, re.IGNORECASE):
                return True
                
        # Check body for event signals
        body_lower = body.lower()
        
        # Check for phrases that strongly indicate an event
        event_phrases = [
            "would like to invite you",
            "please join",
            "invitation to",
            "you are invited",
            "calendar invite",
            "has been scheduled",
            "meeting details",
            "conference details",
            "zoom link",
            "teams link",
            "meeting link",
            "when:",
            "where:",
            "location:",
            "agenda:",
            "attendees:"
        ]
        
        if any(phrase in body_lower for phrase in event_phrases):
            return True
            
        # Count event keywords in body
        event_keyword_count = sum(1 for keyword in self.event_keywords if keyword in body_lower)
        if event_keyword_count >= 2:
            return True
            
        # Check for date/time patterns in the body
        date_time_patterns = 0
        for pattern in self.date_patterns + self.time_patterns:
            if re.search(pattern, body, re.IGNORECASE):
                date_time_patterns += 1
                
        if date_time_patterns >= 2:
            return True
            
        # Not enough evidence this is an event-related email
        return False
    
    def _identify_event_type(self, subject: str, body: str) -> str:
        """
        Identify the type of event based on content.
        
        Args:
            subject: Email subject
            body: Email body text
            
        Returns:
            Event type string
        """
        text = (subject + " " + body).lower()
        
        # Define event types and their related keywords
        event_types = {
            "meeting": ["meeting", "sync", "sync-up", "catchup", "catch-up", "check-in", "standup", "stand-up"],
            "interview": ["interview", "candidate", "hiring", "job", "recruitment", "career", "position"],
            "conference": ["conference", "seminar", "webinar", "symposium", "summit", "convention"],
            "social": ["party", "celebration", "happy hour", "social", "gathering", "birthday", "anniversary"],
            "meal": ["lunch", "dinner", "breakfast", "coffee", "meal", "restaurant", "cafe"],
            "presentation": ["presentation", "demo", "demonstration", "showcase", "review"],
            "workshop": ["workshop", "training", "course", "session", "class"],
            "call": ["call", "phone", "dial-in", "conference call"]
        }
        
        # Count occurrences of each event type's keywords
        type_scores = {}
        for event_type, keywords in event_types.items():
            type_scores[event_type] = sum(1 for keyword in keywords if keyword in text)
            
        # Get the type with highest score
        if type_scores:
            max_type = max(type_scores.items(), key=lambda x: x[1])
            if max_type[1] > 0:
                return max_type[0]
                
        # If no specific type is detected, use a generic type
        if any(keyword in text for keyword in self.event_keywords):
            return "event"
            
        return "unknown"
    
    def _extract_datetime_info(self, subject: str, body: str, email_date: str) -> Dict[str, str]:
        """
        Extract date and time information from email content.
        
        Args:
            subject: Email subject
            body: Email body text
            email_date: Email send date
            
        Returns:
            Dictionary with date and time information
        """
        result = {
            'event_date': None,
            'event_time': None,
            'duration': None,
            'end_time': None
        }
        
        # Combine subject and body for searching
        text = subject + "\n" + body
        
        # First, try to find explicit date-time patterns
        date_found = False
        
        # Look for "when:" pattern which often has date and time info
        when_match = re.search(r'when:([^\n]+)', text, re.IGNORECASE)
        if when_match:
            when_text = when_match.group(1).strip()
            # Parse the extracted date-time text
            try:
                parsed_date = dateparser.parse(when_text)
                if parsed_date:
                    result['event_date'] = parsed_date.strftime('%Y-%m-%d')
                    result['event_time'] = parsed_date.strftime('%H:%M:%S')
                    date_found = True
            except Exception as e:
                logger.debug(f"Error parsing date from 'when:' pattern: {e}")
        
        # If not found, try to extract from more general patterns
        if not date_found:
            # Extract potential date references
            date_matches = []
            for pattern in self.date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    date_str = match.group(0)
                    
                    # Handle relative dates (today, tomorrow, etc.)
                    try:
                        if re.match(r'\b(today|tomorrow|next|this)\b', date_str, re.IGNORECASE):
                            reference_date = dateparser.parse(email_date) if email_date else datetime.now()
                            parsed_date = dateparser.parse(date_str, settings={'RELATIVE_BASE': reference_date})
                        else:
                            parsed_date = dateparser.parse(date_str)
                            
                        if parsed_date:
                            date_matches.append((parsed_date, match.start()))
                    except Exception as e:
                        logger.debug(f"Error parsing date '{date_str}': {e}")
            
            # Extract potential time references
            time_matches = []
            for pattern in self.time_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    time_str = match.group(0)
                    time_matches.append((time_str, match.start()))
            
            # Look for time ranges
            time_range_match = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)\s*(?:-|to|until)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)', text, re.IGNORECASE)
            if time_range_match:
                start_time = time_range_match.group(1)
                end_time = time_range_match.group(2)
                result['event_time'] = start_time
                result['end_time'] = end_time
                
                # Try to parse duration
                try:
                    start_parsed = dateparser.parse(start_time)
                    end_parsed = dateparser.parse(end_time)
                    if start_parsed and end_parsed:
                        duration_minutes = (end_parsed - start_parsed).seconds // 60
                        result['duration'] = f"{duration_minutes} minutes"
                except Exception:
                    pass
            
            # If we found date and time separately, use the closest time to the date
            if date_matches and time_matches and not result['event_time']:
                # Sort by position in text
                date_matches.sort(key=lambda x: x[1])
                parsed_date = date_matches[0][0]
                result['event_date'] = parsed_date.strftime('%Y-%m-%d')
                
                # Find nearest time to the date mention
                date_pos = date_matches[0][1]
                nearest_time = min(time_matches, key=lambda x: abs(x[1] - date_pos))
                result['event_time'] = nearest_time[0]
            elif date_matches:
                # Only date found
                parsed_date = date_matches[0][0]
                result['event_date'] = parsed_date.strftime('%Y-%m-%d')
            
        return result
    
    def _extract_location(self, body: str) -> Optional[str]:
        """
        Extract location information from email body.
        
        Args:
            body: Email body text
            
        Returns:
            Location string or None
        """
        # Look for explicit location marker
        location_match = re.search(r'(?:location|place|venue|where):\s*([^\n]+)', body, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            # Validate location - it should be reasonably sized
            if len(location) > 500:
                location = location[:500] + "... [truncated]"
            return location
            
        # Look for room numbers
        room_match = re.search(r'(?:room|conference room)\s+([A-Za-z0-9\-]+)', body, re.IGNORECASE)
        if room_match:
            return f"Room {room_match.group(1)}"
            
        # Look for addresses (simple pattern)
        address_match = re.search(r'\d+\s+[A-Za-z]+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|place|pl)\b[^@\n]{5,100}', body, re.IGNORECASE)
        if address_match:
            return address_match.group(0).strip()
            
        # Check for virtual meeting platforms
        for platform in ["zoom", "teams", "google meet", "skype", "webex"]:
            if re.search(r'\b' + re.escape(platform) + r'\b', body, re.IGNORECASE):
                return platform.title() + " (Virtual)"
                
        return None
    
    def _extract_virtual_meeting_info(self, body: str) -> Dict[str, str]:
        """
        Extract virtual meeting information from email body.
        
        Args:
            body: Email body text
            
        Returns:
            Dictionary with virtual meeting details
        """
        info = {
            'platform': None,
            'link': None,
            'meeting_id': None,
            'password': None
        }
        
        # Detect virtual meeting platform
        platforms = {
            'zoom': r'\bzoom\b',
            'teams': r'\b(?:microsoft\s+)?teams\b',
            'meet': r'\bgoogle\s+meet\b',
            'skype': r'\bskype\b',
            'webex': r'\bwebex\b'
        }
        
        for platform, pattern in platforms.items():
            if re.search(pattern, body, re.IGNORECASE):
                info['platform'] = platform
                break
                
        # Extract meeting link
        link_patterns = [
            r'(https?://(?:www\.)?zoom\.us/[j]/[^\s<>"\']+)',
            r'(https?://(?:teams|meet)\.(?:microsoft|google)\.com/[^\s<>"\']+)',
            r'(https?://[^\s<>"\']+(?:zoom|teams|meet|skype|webex)[^\s<>"\']*)'
        ]
        
        for pattern in link_patterns:
            link_match = re.search(pattern, body)
            if link_match:
                link = link_match.group(1)
                # Limit link length
                if len(link) > 500:
                    link = link[:500]
                info['link'] = link
                break
                
        # Extract meeting ID
        id_match = re.search(r'(?:meeting|conference)\s+ID:?\s*(\d[\d\s\-]{8,})', body, re.IGNORECASE)
        if id_match:
            # Clean up the ID - remove spaces and dashes
            meeting_id = re.sub(r'[\s\-]', '', id_match.group(1))
            info['meeting_id'] = meeting_id
            
        # Extract password/passcode
        pw_match = re.search(r'(?:password|passcode):\s*([^\s<>"\']+)', body, re.IGNORECASE)
        if pw_match:
            info['password'] = pw_match.group(1)
            
        return info
    
    def _extract_attendees(self, body: str, email_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract list of attendees from email content.
        
        Args:
            body: Email body text
            email_data: Full email data
            
        Returns:
            List of attendee info dictionaries
        """
        attendees = []
        
        # First, add recipients as attendees
        for person in email_data.get('to', []):
            if person.get('email'):
                attendees.append({
                    'name': person.get('name', ''),
                    'email': person.get('email', '').lower(),
                    'role': 'attendee'
                })
                
        # Add cc recipients
        for person in email_data.get('cc', []):
            if person.get('email'):
                attendees.append({
                    'name': person.get('name', ''),
                    'email': person.get('email', '').lower(),
                    'role': 'cc'
                })
                
        # Add sender
        for person in email_data.get('from', []):
            if person.get('email'):
                attendees.append({
                    'name': person.get('name', ''),
                    'email': person.get('email', '').lower(),
                    'role': 'organizer'
                })
                
        # Look for an attendee list in the body
        attendee_block_match = re.search(r'(?:attendees|participants|invitees|guests)[:;]\s*([^#]+)', body, re.IGNORECASE)
        if attendee_block_match:
            attendee_block = attendee_block_match.group(1)
            if len(attendee_block) > 1000:  # Limit processing for large blocks
                attendee_block = attendee_block[:1000]
            
            try:
                # Use NLP to extract people
                doc = nlp(attendee_block)
                
                # Extract person entities
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        person_name = ent.text.strip()
                        
                        # Check if this person is already in the list
                        if not any(attendee['name'].lower() == person_name.lower() for attendee in attendees if attendee.get('name')):
                            attendees.append({
                                'name': person_name,
                                'email': '',
                                'role': 'attendee'
                            })
            except Exception as e:
                logger.debug(f"Error processing attendees with NLP: {e}")
            
            # Also look for email patterns in the attendee block
            email_matches = re.finditer(r'[\w\.-]+@[\w\.-]+', attendee_block)
            for match in email_matches:
                email = match.group(0).lower()
                
                # Check if this email is already in the list
                if not any(attendee['email'] == email for attendee in attendees if attendee.get('email')):
                    # Try to find a name near this email
                    context = attendee_block[max(0, match.start() - 30):match.start()]
                    name_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', context)
                    name = name_match.group(1) if name_match else ''
                    
                    attendees.append({
                        'name': name,
                        'email': email,
                        'role': 'attendee'
                    })
                    
        return attendees
    
    def _extract_calendar_action(self, body: str) -> Optional[str]:
        """
        Extract calendar action (accept/decline/tentative) if present.
        
        Args:
            body: Email body text
            
        Returns:
            Action string or None
        """
        action_patterns = {
            'accepted': r'\b(?:accepted|confirmed|attending|will attend|going)\b',
            'declined': r'\b(?:declined|rejected|not attending|cannot attend|won\'t attend|will not attend)\b',
            'tentative': r'\b(?:tentative|maybe|possibly|might attend|not sure)\b'
        }
        
        for action, pattern in action_patterns.items():
            if re.search(pattern, body, re.IGNORECASE):
                return action
                
        return None
    
    def _format_sender(self, from_list: List[Dict[str, str]]) -> Dict[str, str]:
        """Format sender information."""
        if from_list and len(from_list) > 0:
            return {
                'name': from_list[0].get('name', ''),
                'email': from_list[0].get('email', '').lower() if from_list[0].get('email') else ''
            }
        return {'name': '', 'email': ''}
    
    def _format_recipients(self, recipients_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format recipient list."""
        return [
            {'name': r.get('name', ''), 'email': r.get('email', '').lower()}
            for r in recipients_list
            if r.get('email')
        ]
    
    def _is_valid_event(self, event_info: Dict[str, Any]) -> bool:
        """
        Determine if we have enough information to consider this a valid event.
        
        Args:
            event_info: Dictionary of event information
            
        Returns:
            Boolean indicating if this is a valid event
        """
        # At minimum, we need:
        # 1. A subject or event type
        # 2. At least one of: event_date, location, virtual meeting link
        
        if not event_info.get('subject') and not event_info.get('event_type'):
            return False
            
        # Check for date or location information
        has_date = bool(event_info.get('event_date'))
        has_location = bool(event_info.get('location'))
        has_virtual = bool(event_info.get('virtual_meeting', {}).get('link'))
        
        return has_date or has_location or has_virtual 