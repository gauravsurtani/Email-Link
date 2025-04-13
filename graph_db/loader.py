"""
Loader module for inserting email data into Neo4j graph database.
"""

import re
import json
import logging
from tqdm import tqdm
from neo4j import GraphDatabase
from .schema import setup_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailGraphLoader:
    """Class for loading email data into Neo4j graph database."""
    
    def __init__(self, uri, username, password):
        """
        Initialize the loader with Neo4j connection details.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def setup_database(self, clear=False):
        """
        Set up the Neo4j database schema.
        
        Args:
            clear: Boolean to clear database before setup (default: False)
        """
        setup_database(self.driver, clear)
        
    def extract_domain(self, email_address):
        """
        Extract domain from email address.
        
        Args:
            email_address: Email address string
            
        Returns:
            Domain string or None if not found
        """
        if not email_address:
            return None
            
        match = re.search(r'@([^@]+)$', email_address)
        if match:
            return match.group(1).lower()
        return None
        
    def load_email(self, email_data):
        """
        Load a single email into the graph database.
        
        Args:
            email_data: Dictionary containing email data
            
        Returns:
            Boolean indicating success
        """
        if not email_data.get('message_id'):
            logger.warning("Skipping email with no message ID")
            return False
            
        try:
            with self.driver.session() as session:
                session.execute_write(self._create_email_nodes, email_data)
            return True
        except Exception as e:
            logger.error(f"Error loading email {email_data.get('message_id')}: {e}")
            return False
    
    def _create_email_nodes(self, tx, email):
        """
        Transaction function to create email nodes and relationships.
        
        Args:
            tx: Neo4j transaction
            email: Dictionary containing email data
        """
        # Create Email node
        query = """
        MERGE (e:Email {message_id: $message_id}) 
        SET e.subject = $subject,
            e.date = $date,
            e.body = $body,
            e.has_attachments = $has_attachments
        RETURN e
        """
        
        tx.run(query, 
               message_id=email['message_id'],
               subject=email['subject'],
               date=email['date'],
               body=email['body'],
               has_attachments=email['has_attachments'])
        
        # Create Thread node and relationship
        if email['thread_id']:
            query = """
            MERGE (t:Thread {thread_id: $thread_id})
            WITH t
            MATCH (e:Email {message_id: $message_id})
            MERGE (e)-[:BELONGS_TO]->(t)
            """
            tx.run(query, thread_id=email['thread_id'], message_id=email['message_id'])
        
        # Create Person nodes (from) and relationships
        for person in email['from']:
            self._create_person_relationships(tx, person, email['message_id'], "SENT")
        
        # Create Person nodes (to) and relationships
        for person in email['to']:
            self._create_person_relationships(tx, person, email['message_id'], "RECEIVED")
            
        # Create Person nodes (cc) and relationships
        for person in email['cc']:
            self._create_person_relationships(tx, person, email['message_id'], "COPIED")
            
        # Create Person nodes (bcc) and relationships
        for person in email.get('bcc', []):
            self._create_person_relationships(tx, person, email['message_id'], "BCC")
            
        # Create Label nodes and relationships
        for label in email['labels']:
            if label and label.strip():
                query = """
                MERGE (l:Label {name: $label})
                WITH l
                MATCH (e:Email {message_id: $message_id})
                MERGE (e)-[:HAS_LABEL]->(l)
                """
                tx.run(query, label=label.strip(), message_id=email['message_id'])
                
        # Create Attachment nodes and relationships
        for attachment in email['attachments']:
            query = """
            CREATE (a:Attachment {
                filename: $filename,
                content_type: $content_type,
                size: $size
            })
            WITH a
            MATCH (e:Email {message_id: $message_id})
            MERGE (e)-[:HAS_ATTACHMENT]->(a)
            """
            tx.run(query, 
                   filename=attachment['filename'],
                   content_type=attachment['content_type'],
                   size=attachment['size'],
                   message_id=email['message_id'])
                   
        # Create Reply relationships
        if email.get('in_reply_to'):
            query = """
            MATCH (original:Email {message_id: $in_reply_to})
            MATCH (reply:Email {message_id: $message_id})
            MERGE (reply)-[:REPLY_TO]->(original)
            """
            tx.run(query, 
                   in_reply_to=email['in_reply_to'],
                   message_id=email['message_id'])
    
    def _create_person_relationships(self, tx, person, message_id, rel_type):
        """
        Create Person node and its relationships.
        
        Args:
            tx: Neo4j transaction
            person: Dictionary with name and email
            message_id: Email message ID
            rel_type: Relationship type
        """
        if not person.get('email') or '@' not in person['email']:
            return
            
        # Create Person node
        query = """
        MERGE (p:Person {email: $email})
        SET p.name = $name
        WITH p
        MATCH (e:Email {message_id: $message_id})
        MERGE (p)-[:%s]->(e)
        """ % rel_type
        
        tx.run(query, 
               email=person['email'], 
               name=person['name'] if person.get('name') else person['email'],
               message_id=message_id)
               
        # Create Domain relationship
        domain = self.extract_domain(person['email'])
        if domain:
            query = """
            MERGE (d:Domain {name: $domain})
            WITH d
            MATCH (p:Person {email: $email})
            MERGE (p)-[:BELONGS_TO]->(d)
            """
            tx.run(query, domain=domain, email=person['email'])
    
    def load_emails_from_json(self, json_file, batch_size=100):
        """
        Load all emails from a JSON file into the graph database.
        
        Args:
            json_file: Path to JSON file containing email data
            batch_size: Number of emails to process in each batch
            
        Returns:
            Number of successfully loaded emails
        """
        logger.info(f"Loading emails from {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                emails = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return 0
            
        return self.load_emails(emails, batch_size)
    
    def load_emails(self, emails, batch_size=100):
        """
        Load a list of email dictionaries into the graph database.
        
        Args:
            emails: List of email dictionaries
            batch_size: Number of emails to process in each batch
            
        Returns:
            Number of successfully loaded emails
        """
        total = len(emails)
        logger.info(f"Loading {total} emails into graph database")
        
        success_count = 0
        
        for i in tqdm(range(0, total, batch_size)):
            batch = emails[i:i+batch_size]
            
            for email in batch:
                if self.load_email(email):
                    success_count += 1
        
        logger.info(f"Successfully loaded {success_count}/{total} emails")
        return success_count 