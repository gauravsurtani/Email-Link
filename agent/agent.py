"""
Agent module implementing the agentic AI system for email graph exploration.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase

from .actions import AgentActions
from analysis.queries import EmailGraphQueries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailGraphAgent:
    """
    Agentic AI system for exploring and analyzing the email graph database.
    This class serves as the main interface for interacting with the email graph.
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        """
        Initialize the agent with Neo4j connection parameters.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.queries = EmailGraphQueries(self.driver)
        self.actions = AgentActions(self.queries)
        self.context = {}  # Maintains agent state/context
        
    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()
        
    def reset_context(self):
        """Reset the agent's context/state."""
        self.context = {}
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query and execute appropriate actions.
        
        Args:
            query: Natural language query string
            
        Returns:
            Result dictionary with response and any data
        """
        # For now, we'll use a simple keyword-based approach
        # In a real implementation, this would use a more sophisticated NLP system
        query = query.lower()
        
        # Dispatch to appropriate handler based on query content
        if any(term in query for term in ['overview', 'summary', 'statistics']):
            return self._handle_overview_query()
            
        elif any(term in query for term in ['search', 'find emails', 'containing']):
            return self._handle_search_query(query)
            
        elif any(term in query for term in ['person', 'contact', 'about']) and '@' in query:
            return self._handle_person_query(query)
            
        elif any(term in query for term in ['connection', 'between', 'relate']):
            return self._handle_connection_query(query)
            
        elif any(term in query for term in ['thread', 'conversation']):
            return self._handle_thread_query(query)
            
        elif any(term in query for term in ['time', 'pattern', 'volume']):
            return self._handle_time_query(query)
            
        elif any(term in query for term in ['domain', 'company']):
            return self._handle_domain_query(query)
            
        else:
            # Default fallback
            return {
                "type": "error",
                "message": "I'm not sure how to process this query. Try asking about email overviews, searching for emails, looking up person information, or finding connections between people."
            }
            
    def _handle_overview_query(self) -> Dict[str, Any]:
        """Handle overview/summary queries."""
        overview = self.actions.get_overview()
        
        # Store in context
        self.context['last_overview'] = overview
        
        return {
            "type": "overview",
            "data": overview,
            "message": "Here's an overview of your email graph."
        }
        
    def _handle_search_query(self, query: str) -> Dict[str, Any]:
        """Handle email search queries."""
        # Extract search terms (simple approach)
        search_terms = []
        for prefix in ['search for', 'find emails', 'containing', 'about', 'with']:
            if prefix in query:
                parts = query.split(prefix, 1)
                if len(parts) > 1 and parts[1].strip():
                    search_terms.append(parts[1].strip())
        
        # Use the longest extracted term as it's likely the most specific
        search_term = max(search_terms, key=len) if search_terms else ""
        
        if not search_term:
            return {
                "type": "error",
                "message": "I couldn't identify what you want to search for. Please specify search terms."
            }
            
        # Perform search
        results = self.actions.search_emails(search_term)
        
        # Store in context
        self.context['last_search'] = {
            'term': search_term,
            'results': results
        }
        
        return {
            "type": "search",
            "query": search_term,
            "data": results,
            "count": len(results),
            "message": f"Found {len(results)} emails containing '{search_term}'."
        }
        
    def _handle_person_query(self, query: str) -> Dict[str, Any]:
        """Handle person information queries."""
        # Extract email address (simple approach)
        import re
        email_match = re.search(r'[\w\.-]+@[\w\.-]+', query)
        
        if not email_match:
            return {
                "type": "error",
                "message": "I couldn't identify an email address in your query. Please include a valid email."
            }
            
        email = email_match.group(0)
        
        # Get person information
        person_info = self.actions.get_person_info(email)
        
        if not person_info:
            return {
                "type": "error",
                "message": f"I couldn't find information for the email address: {email}"
            }
            
        # Store in context
        self.context['last_person'] = person_info
        
        return {
            "type": "person",
            "email": email,
            "data": person_info,
            "message": f"Here's information about {email}."
        }
        
    def _handle_connection_query(self, query: str) -> Dict[str, Any]:
        """Handle connection queries between people."""
        # Extract email addresses (simple approach)
        import re
        email_matches = re.findall(r'[\w\.-]+@[\w\.-]+', query)
        
        if len(email_matches) < 2:
            return {
                "type": "error",
                "message": "I need two email addresses to find connections. Please include both."
            }
            
        email1 = email_matches[0]
        email2 = email_matches[1]
        
        # Find connections
        max_depth = 3
        if 'depth' in query:
            # Extract depth if specified
            depth_match = re.search(r'depth\s+(\d+)', query)
            if depth_match:
                max_depth = int(depth_match.group(1))
                
        connection_info = self.actions.find_connections(email1, email2, max_depth)
        
        # Store in context
        self.context['last_connection'] = connection_info
        
        if 'error' in connection_info:
            return {
                "type": "error",
                "message": connection_info['error']
            }
            
        return {
            "type": "connection",
            "emails": [email1, email2],
            "data": connection_info,
            "message": f"Here are the connections between {email1} and {email2}."
        }
        
    def _handle_thread_query(self, query: str) -> Dict[str, Any]:
        """Handle thread analysis queries."""
        # Extract thread ID or subject
        import re
        
        # First try to extract thread ID
        thread_match = re.search(r'thread\s+(?:id\s+)?([a-zA-Z0-9]+)', query)
        
        if thread_match:
            thread_id = thread_match.group(1)
            thread_info = self.actions.analyze_thread(thread_id)
            
            # Store in context
            self.context['last_thread'] = thread_info
            
            return {
                "type": "thread",
                "thread_id": thread_id,
                "data": thread_info,
                "message": f"Here's analysis of thread {thread_id}."
            }
            
        # Try to extract subject
        subject_match = re.search(r'subject\s+"([^"]+)"', query)
        if not subject_match:
            subject_match = re.search(r'about\s+"([^"]+)"', query)
            
        if subject_match:
            subject = subject_match.group(1)
            similar_threads = self.actions.find_similar_threads(subject)
            
            # Store in context
            self.context['similar_threads'] = {
                'subject': subject,
                'threads': similar_threads
            }
            
            return {
                "type": "similar_threads",
                "subject": subject,
                "data": similar_threads,
                "count": len(similar_threads),
                "message": f"Found {len(similar_threads)} threads with subject similar to '{subject}'."
            }
            
        return {
            "type": "error",
            "message": "I need a thread ID or subject to analyze threads. Please specify one."
        }
        
    def _handle_time_query(self, query: str) -> Dict[str, Any]:
        """Handle time-based analysis queries."""
        # Extract time interval
        interval = 'month'  # Default
        for term in ['day', 'week', 'month', 'year']:
            if term in query:
                interval = term
                break
                
        # Extract limit
        import re
        limit = 12  # Default
        limit_match = re.search(r'last\s+(\d+)', query)
        if limit_match:
            limit = int(limit_match.group(1))
            
        # Get time-based data
        time_data = self.actions.get_communication_over_time(interval, limit)
        
        # Store in context
        self.context['time_analysis'] = {
            'interval': interval,
            'limit': limit,
            'data': time_data
        }
        
        return {
            "type": "time_analysis",
            "interval": interval,
            "limit": limit,
            "data": time_data,
            "message": f"Here's your email communication volume over the last {limit} {interval}s."
        }
        
    def _handle_domain_query(self, query: str) -> Dict[str, Any]:
        """Handle domain analysis queries."""
        # Extract domain
        import re
        domain_match = re.search(r'domain\s+([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,})', query)
        
        if not domain_match:
            # Try another pattern
            domain_match = re.search(r'@([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,})', query)
            
        if not domain_match:
            return {
                "type": "error",
                "message": "I couldn't identify a domain in your query. Please specify one."
            }
            
        domain = domain_match.group(1)
        
        # Get domain analysis
        domain_info = self.actions.get_domain_analysis(domain)
        
        # Store in context
        self.context['domain_analysis'] = domain_info
        
        return {
            "type": "domain_analysis",
            "domain": domain,
            "data": domain_info,
            "message": f"Here's analysis of communications with the domain {domain}."
        } 