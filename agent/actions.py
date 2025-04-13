"""
Actions module defining the operations that the email graph agent can perform.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentActions:
    """
    Class encapsulating the actions that the email graph agent can perform.
    This serves as an interface between the agent and the graph database.
    """
    
    def __init__(self, queries_api):
        """
        Initialize with a queries API instance.
        
        Args:
            queries_api: EmailGraphQueries instance
        """
        self.queries = queries_api
        
    def get_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the email graph.
        
        Returns:
            Dictionary with overview statistics
        """
        return {
            "total_emails": self.queries.get_email_count(),
            "total_people": self.queries.get_person_count(),
            "top_senders": self.queries.get_top_senders(5),
            "top_receivers": self.queries.get_top_receivers(5),
            "top_domains": self.queries.get_common_domains(5)
        }
        
    def search_emails(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for emails containing specific text.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching emails
        """
        return self.queries.search_emails(query, limit)
        
    def get_person_info(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific person.
        
        Args:
            email: Email address
            
        Returns:
            Dictionary with person details or None if not found
        """
        return self.queries.get_person_details(email)
        
    def analyze_thread(self, thread_id: str) -> Dict[str, Any]:
        """
        Analyze an email thread.
        
        Args:
            thread_id: Thread ID
            
        Returns:
            Dictionary with thread analysis
        """
        emails = self.queries.get_emails_by_thread(thread_id)
        
        # Extract participants
        participants = set()
        for email in emails:
            if email.get("sender"):
                participants.add(email["sender"])
            if email.get("receivers"):
                for receiver in email["receivers"]:
                    participants.add(receiver)
        
        return {
            "thread_id": thread_id,
            "email_count": len(emails),
            "subject": emails[0]["subject"] if emails else "",
            "participants": list(participants),
            "start_date": emails[0]["date"] if emails else None,
            "end_date": emails[-1]["date"] if emails else None,
            "emails": emails
        }
        
    def find_connections(self, email1: str, email2: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Find connections between two people.
        
        Args:
            email1: First email address
            email2: Second email address
            max_depth: Maximum connection depth
            
        Returns:
            Dictionary with connection information
        """
        person1 = self.queries.get_person_details(email1)
        person2 = self.queries.get_person_details(email2)
        
        if not person1 or not person2:
            return {
                "error": "One or both people not found",
                "email1": email1,
                "email2": email2,
                "found_email1": person1 is not None,
                "found_email2": person2 is not None
            }
            
        comm_stats = self.queries.get_communication_between(email1, email2)
        paths = self.queries.find_connections_between(email1, email2, max_depth)
        
        return {
            "person1": person1,
            "person2": person2,
            "direct_communication": comm_stats,
            "connection_paths": paths,
            "has_direct_connection": comm_stats["total"] > 0,
            "has_indirect_connection": len(paths) > 0
        }
        
    def get_communication_over_time(self, interval: str = 'month', limit: int = 12) -> List[Dict[str, Any]]:
        """
        Get email communication volume over time.
        
        Args:
            interval: Time grouping ('day', 'week', 'month', 'year')
            limit: Maximum number of intervals
            
        Returns:
            List of time periods with email counts
        """
        return self.queries.get_communication_over_time(interval, limit)
        
    def find_similar_threads(self, subject: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find email threads with similar subjects.
        
        Args:
            subject: Subject text to match
            limit: Maximum number of results
            
        Returns:
            List of similar threads
        """
        # Create a case-insensitive regex pattern from the subject
        pattern = f"(?i).*{subject}.*"
        return self.queries.get_email_threads_by_subject(pattern, limit)
        
    def get_domain_analysis(self, domain: str) -> Dict[str, Any]:
        """
        Analyze communications with a specific domain.
        
        Args:
            domain: Email domain
            
        Returns:
            Dictionary with domain analysis
        """
        query = f"""
        MATCH (d:Domain {{name: $domain}})<-[:BELONGS_TO]-(p:Person)
        RETURN count(p) as person_count
        """
        
        query_sent = f"""
        MATCH (d:Domain {{name: $domain}})<-[:BELONGS_TO]-(p:Person)-[:SENT]->(e:Email)
        RETURN count(e) as count
        """
        
        query_received = f"""
        MATCH (d:Domain {{name: $domain}})<-[:BELONGS_TO]-(p:Person)-[:RECEIVED]->(e:Email)
        RETURN count(e) as count
        """
        
        query_top_people = f"""
        MATCH (d:Domain {{name: $domain}})<-[:BELONGS_TO]-(p:Person)
        OPTIONAL MATCH (p)-[:SENT]->(sent:Email)
        OPTIONAL MATCH (p)-[:RECEIVED]->(received:Email)
        WITH p, count(distinct sent) + count(distinct received) as activity
        RETURN p.email as email, p.name as name, activity
        ORDER BY activity DESC
        LIMIT 5
        """
        
        with self.queries.driver.session() as session:
            person_count = session.run(query, domain=domain).single()
            sent_count = session.run(query_sent, domain=domain).single()
            received_count = session.run(query_received, domain=domain).single()
            top_people = [dict(record) for record in session.run(query_top_people, domain=domain)]
            
        return {
            "domain": domain,
            "person_count": person_count["person_count"] if person_count else 0,
            "emails_sent": sent_count["count"] if sent_count else 0,
            "emails_received": received_count["count"] if received_count else 0,
            "top_people": top_people
        } 