"""
Module for adding extracted events to the Neo4j knowledge graph.
"""

import logging
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EventGraphEnhancer:
    """Class for enhancing the email graph with event information."""
    
    def __init__(self, uri, username, password):
        """
        Initialize with Neo4j connection details.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def create_event_schema(self):
        """Create Neo4j schema for event nodes and relationships."""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                session.run(constraint)
                
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.event_type)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.event_date)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Location) ON (e.name)"
            ]
            
            for index in indexes:
                session.run(index)
                
            logger.info("Event schema created in Neo4j")
            
    def add_event_to_graph(self, event_data: Dict[str, Any], email_id: str) -> Optional[str]:
        """
        Add an event to the Neo4j graph.
        
        Args:
            event_data: Dictionary containing event information
            email_id: ID of the email that contained this event
            
        Returns:
            Event ID or None if failed
        """
        if not event_data:
            return None
            
        try:
            with self.driver.session() as session:
                result = session.execute_write(self._create_event_transaction, event_data, email_id)
                return result
        except Exception as e:
            logger.error(f"Error adding event to graph: {e}")
            return None
            
    def _trim_string(self, text, max_length=1000):
        """
        Trim a string to specified maximum length to prevent Neo4j index issues.
        
        Args:
            text: Text to trim
            max_length: Maximum allowed length
            
        Returns:
            Trimmed string
        """
        if not text:
            return text
            
        if len(text) <= max_length:
            return text
            
        return text[:max_length] + "... [truncated]"
            
    def _create_event_transaction(self, tx, event_data: Dict[str, Any], email_id: str) -> str:
        """
        Create event nodes and relationships in a transaction.
        
        Args:
            tx: Neo4j transaction
            event_data: Dictionary containing event information
            email_id: ID of the email that contained this event
            
        Returns:
            Event ID
        """
        # Generate an event ID (using email ID and a timestamp suffix)
        import time
        event_id = f"{email_id}_{int(time.time() * 1000)}"
        
        # Create Event node
        event_query = """
        CREATE (e:Event {
            event_id: $event_id,
            event_type: $event_type,
            subject: $subject,
            event_date: $event_date,
            event_time: $event_time,
            duration: $duration,
            end_time: $end_time
        })
        RETURN e.event_id as event_id
        """
        
        # Trim subject to prevent index issues
        subject = self._trim_string(event_data.get('subject', ''), 1000)
        
        result = tx.run(event_query, 
                    event_id=event_id,
                    event_type=event_data.get('event_type', 'unknown'),
                    subject=subject,
                    event_date=event_data.get('event_date'),
                    event_time=event_data.get('event_time'),
                    duration=event_data.get('duration'),
                    end_time=event_data.get('end_time')
                )
                
        # Connect Event to the Email
        email_query = """
        MATCH (email:Email {message_id: $email_id})
        MATCH (event:Event {event_id: $event_id})
        MERGE (email)-[:CONTAINS_EVENT]->(event)
        """
        
        tx.run(email_query, email_id=email_id, event_id=event_id)
        
        # Create Location node if location is specified
        if event_data.get('location'):
            # Trim location name to prevent index issues
            location_name = self._trim_string(event_data.get('location'), 500)
            
            location_query = """
            MERGE (l:Location {name: $location})
            WITH l
            MATCH (e:Event {event_id: $event_id})
            MERGE (e)-[:LOCATED_AT]->(l)
            """
            
            tx.run(location_query, 
                location=location_name,
                event_id=event_id
            )
            
        # Add virtual meeting info if available
        virtual_meeting = event_data.get('virtual_meeting', {})
        if virtual_meeting and virtual_meeting.get('platform'):
            # Trim link to prevent issues
            link = self._trim_string(virtual_meeting.get('link', ''), 500)
            
            virtual_query = """
            MATCH (e:Event {event_id: $event_id})
            SET e.virtual_platform = $platform,
                e.virtual_link = $link,
                e.virtual_meeting_id = $meeting_id,
                e.virtual_password = $password
            """
            
            tx.run(virtual_query,
                event_id=event_id,
                platform=virtual_meeting.get('platform'),
                link=link,
                meeting_id=virtual_meeting.get('meeting_id'),
                password=virtual_meeting.get('password')
            )
            
        # Create Attendee relationships
        attendees = event_data.get('attendees', [])
        for attendee in attendees:
            # Find or create the Person node
            if not attendee.get('email'):
                continue
                
            attendee_query = """
            MATCH (p:Person {email: $email})
            MATCH (e:Event {event_id: $event_id})
            MERGE (p)-[r:ATTENDS]->(e)
            SET r.role = $role
            """
            
            tx.run(attendee_query,
                email=attendee.get('email'),
                event_id=event_id,
                role=attendee.get('role', 'attendee')
            )
            
        # If the creator/organizer is clear, create that relationship
        organizer = None
        for attendee in attendees:
            if attendee.get('role') == 'organizer':
                organizer = attendee
                break
                
        if organizer and organizer.get('email'):
            organizer_query = """
            MATCH (p:Person {email: $email})
            MATCH (e:Event {event_id: $event_id})
            MERGE (p)-[:ORGANIZES]->(e)
            """
            
            tx.run(organizer_query,
                email=organizer.get('email'),
                event_id=event_id
            )
            
        return event_id
        
    def process_all_emails(self, event_extractor):
        """
        Process all emails in the database to extract and add events.
        
        Args:
            event_extractor: EventExtractor instance
            
        Returns:
            Number of events extracted
        """
        logger.info("Processing all emails in the database for events")
        
        # First, create the event schema
        self.create_event_schema()
        
        # Batch process emails
        with self.driver.session() as session:
            # Get count of emails
            count_result = session.run("MATCH (e:Email) RETURN count(e) as count")
            total_emails = count_result.single()["count"]
            logger.info(f"Found {total_emails} emails to process")
            
            # Process in batches
            batch_size = 100
            event_count = 0
            
            for offset in range(0, total_emails, batch_size):
                logger.info(f"Processing batch {offset//batch_size + 1} (emails {offset+1}-{min(offset+batch_size, total_emails)})")
                
                # Get batch of emails
                query = """
                MATCH (e:Email)
                OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
                OPTIONAL MATCH (receiver:Person)-[:RECEIVED]->(e)
                OPTIONAL MATCH (cc:Person)-[:COPIED]->(e)
                RETURN e.message_id as message_id,
                       e.subject as subject,
                       e.date as date,
                       e.body as body,
                       collect(DISTINCT {name: sender.name, email: sender.email}) as from,
                       collect(DISTINCT {name: receiver.name, email: receiver.email}) as to,
                       collect(DISTINCT {name: cc.name, email: cc.email}) as cc
                ORDER BY e.date
                SKIP $offset
                LIMIT $limit
                """
                
                result = session.run(query, offset=offset, limit=batch_size)
                
                # Process each email
                batch_event_count = 0
                for record in result:
                    email_data = dict(record)
                    
                    # Extract events
                    try:
                        events = event_extractor.extract_events_from_email(email_data)
                        
                        # Add events to graph
                        for event in events:
                            event_id = self.add_event_to_graph(event, email_data['message_id'])
                            if event_id:
                                batch_event_count += 1
                    except Exception as e:
                        logger.error(f"Error processing email {email_data.get('message_id')}: {e}")
                        continue
                
                event_count += batch_event_count
                logger.info(f"Extracted {batch_event_count} events from this batch")
                
        logger.info(f"Total events extracted and added to graph: {event_count}")
        return event_count 