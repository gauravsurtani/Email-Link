"""
Queries module with common Neo4j Cypher queries for email graph analysis.
"""

import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailGraphQueries:
    """Class for executing common queries against the email graph database."""
    
    def __init__(self, driver):
        """
        Initialize with a Neo4j driver.
        
        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
        
    def close(self):
        """Close the Neo4j driver."""
        pass  # Driver is managed externally
        
    def get_email_count(self):
        """
        Get total number of emails in the database.
        
        Returns:
            Integer count of emails
        """
        with self.driver.session() as session:
            result = session.run("MATCH (e:Email) RETURN count(e) as count")
            return result.single()["count"]
            
    def get_person_count(self):
        """
        Get total number of people in the database.
        
        Returns:
            Integer count of people
        """
        with self.driver.session() as session:
            result = session.run("MATCH (p:Person) RETURN count(p) as count")
            return result.single()["count"]
            
    def get_top_senders(self, limit=10):
        """
        Get list of top email senders.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with sender info and counts
        """
        query = """
        MATCH (p:Person)-[:SENT]->(e:Email)
        RETURN p.email as email, p.name as name, count(e) as sent_count
        ORDER BY sent_count DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]
            
    def get_top_receivers(self, limit=10):
        """
        Get list of top email receivers.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with receiver info and counts
        """
        query = """
        MATCH (p:Person)-[:RECEIVED]->(e:Email)
        RETURN p.email as email, p.name as name, count(e) as received_count
        ORDER BY received_count DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]
            
    def get_communication_between(self, email1, email2):
        """
        Get communication patterns between two email addresses.
        
        Args:
            email1: First email address
            email2: Second email address
            
        Returns:
            Dictionary with communication statistics
        """
        query = """
        // Emails from email1 to email2
        MATCH (p1:Person {email: $email1})-[:SENT]->(e1:Email)<-[:RECEIVED]-(p2:Person {email: $email2})
        WITH count(e1) as sent_count
        
        // Emails from email2 to email1
        MATCH (p2:Person {email: $email2})-[:SENT]->(e2:Email)<-[:RECEIVED]-(p1:Person {email: $email1})
        
        RETURN sent_count, count(e2) as received_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, email1=email1, email2=email2)
            record = result.single()
            if record:
                return {
                    "from": email1,
                    "to": email2,
                    "sent": record["sent_count"],
                    "received": record["received_count"],
                    "total": record["sent_count"] + record["received_count"]
                }
            return {
                "from": email1,
                "to": email2,
                "sent": 0,
                "received": 0,
                "total": 0
            }
            
    def get_email_threads_by_subject(self, subject_pattern, limit=10):
        """
        Get email threads matching a subject pattern.
        
        Args:
            subject_pattern: Subject text to match (regex pattern)
            limit: Maximum number of results
            
        Returns:
            List of thread information
        """
        query = """
        MATCH (e:Email)-[:BELONGS_TO]->(t:Thread)
        WHERE e.subject =~ $subject_pattern
        WITH t, collect(e) as emails
        RETURN t.thread_id as thread_id, 
               size(emails) as email_count,
               [e in emails | e.subject][0] as subject
        ORDER BY email_count DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, subject_pattern=subject_pattern, limit=limit)
            return [dict(record) for record in result]
            
    def get_emails_by_thread(self, thread_id, limit=100):
        """
        Get emails in a specific thread.
        
        Args:
            thread_id: Thread ID
            limit: Maximum number of results
            
        Returns:
            List of emails in the thread
        """
        query = """
        MATCH (e:Email)-[:BELONGS_TO]->(t:Thread {thread_id: $thread_id})
        OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
        OPTIONAL MATCH (receiver:Person)-[:RECEIVED]->(e)
        RETURN e.message_id as message_id,
               e.subject as subject,
               e.date as date,
               sender.email as sender,
               collect(distinct receiver.email) as receivers
        ORDER BY e.date
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, thread_id=thread_id, limit=limit)
            return [dict(record) for record in result]
            
    def get_communication_over_time(self, interval='month', limit=12):
        """
        Get email communication volume over time.
        
        Args:
            interval: Time grouping ('day', 'week', 'month', 'year')
            limit: Maximum number of intervals to return
            
        Returns:
            List of time periods with email counts
        """
        time_format = {
            'day': '%Y-%m-%d',
            'week': '%Y-%W',
            'month': '%Y-%m',
            'year': '%Y'
        }.get(interval.lower(), '%Y-%m')
        
        query = f"""
        MATCH (e:Email)
        WHERE e.date IS NOT NULL
        WITH datetime(e.date) as date, e
        RETURN date.{interval} as period, count(e) as email_count
        ORDER BY period DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]
            
    def get_common_domains(self, limit=10):
        """
        Get most common email domains.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of domains with counts
        """
        query = """
        MATCH (p:Person)-[:BELONGS_TO]->(d:Domain)
        RETURN d.name as domain, count(p) as person_count
        ORDER BY person_count DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]
            
    def find_connections_between(self, email1, email2, max_depth=3):
        """
        Find connection paths between two people.
        
        Args:
            email1: First email address
            email2: Second email address
            max_depth: Maximum path length
            
        Returns:
            List of paths between the two people
        """
        query = f"""
        MATCH path = (p1:Person {{email: $email1}})-[*1..{max_depth}]-(p2:Person {{email: $email2}})
        RETURN path
        LIMIT 10
        """
        
        with self.driver.session() as session:
            result = session.run(query, email1=email1, email2=email2)
            # Converting paths to a simpler representation
            paths = []
            for record in result:
                path = record["path"]
                path_data = []
                for rel in path.relationships:
                    path_data.append({
                        "start": rel.start_node['email'] if 'email' in rel.start_node else rel.start_node['message_id'],
                        "type": rel.type,
                        "end": rel.end_node['email'] if 'email' in rel.end_node else rel.end_node['message_id'],
                    })
                paths.append(path_data)
            return paths
            
    def search_emails(self, query_text, limit=50):
        """
        Search emails containing specific text.
        
        Args:
            query_text: Text to search for in subject or body
            limit: Maximum number of results
            
        Returns:
            List of matching emails
        """
        query = """
        MATCH (e:Email)
        WHERE e.subject CONTAINS $query_text OR e.body CONTAINS $query_text
        OPTIONAL MATCH (sender:Person)-[:SENT]->(e)
        RETURN e.message_id as message_id,
               e.subject as subject,
               e.date as date,
               sender.email as sender,
               LEFT(e.body, 200) as excerpt
        ORDER BY e.date DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, query_text=query_text, limit=limit)
            return [dict(record) for record in result]
            
    def get_person_details(self, email):
        """
        Get detailed information about a person.
        
        Args:
            email: Email address of the person
            
        Returns:
            Dictionary with person details
        """
        query = """
        MATCH (p:Person {email: $email})
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(d:Domain)
        OPTIONAL MATCH (p)-[:SENT]->(sent:Email)
        OPTIONAL MATCH (p)-[:RECEIVED]->(received:Email)
        
        WITH p, d, 
             count(distinct sent) as sent_count, 
             count(distinct received) as received_count
        
        // Find top 5 communication partners
        OPTIONAL MATCH (p)-[:SENT]->(:Email)<-[:RECEIVED]-(partner:Person)
        WITH p, d, sent_count, received_count, partner, count(*) as partner_count
        ORDER BY partner_count DESC
        LIMIT 5
        
        RETURN p.email as email,
               p.name as name,
               d.name as domain,
               sent_count,
               received_count,
               collect({email: partner.email, name: partner.name, count: partner_count}) as top_partners
        """
        
        with self.driver.session() as session:
            result = session.run(query, email=email)
            record = result.single()
            return dict(record) if record else None 