"""
Metrics module for computing email communication metrics from the graph database.
"""

import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_communication_metrics(driver):
    """
    Compute various communication metrics from the email graph.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        Dictionary with various metrics
    """
    metrics = {}
    
    # Basic counts
    metrics.update(compute_basic_counts(driver))
    
    # Time-based metrics
    metrics.update(compute_time_metrics(driver))
    
    # Network metrics
    metrics.update(compute_network_metrics(driver))
    
    # Content metrics
    metrics.update(compute_content_metrics(driver))
    
    return metrics

def compute_basic_counts(driver):
    """
    Compute basic count metrics.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        Dictionary with count metrics
    """
    metrics = {}
    
    queries = {
        "total_emails": "MATCH (e:Email) RETURN count(e) as count",
        "total_threads": "MATCH (t:Thread) RETURN count(t) as count",
        "total_people": "MATCH (p:Person) RETURN count(p) as count",
        "total_domains": "MATCH (d:Domain) RETURN count(d) as count",
        "emails_with_attachments": "MATCH (e:Email)-[:HAS_ATTACHMENT]->() RETURN count(distinct e) as count",
        "total_attachments": "MATCH ()-[:HAS_ATTACHMENT]->(a:Attachment) RETURN count(a) as count",
    }
    
    with driver.session() as session:
        for metric_name, query in queries.items():
            try:
                result = session.run(query)
                metrics[metric_name] = result.single()["count"]
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                metrics[metric_name] = 0
                
    return metrics

def compute_time_metrics(driver):
    """
    Compute time-based metrics.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        Dictionary with time-based metrics
    """
    metrics = {}
    
    queries = {
        "first_email_date": """
            MATCH (e:Email)
            WHERE e.date IS NOT NULL
            RETURN e.date as date
            ORDER BY e.date ASC
            LIMIT 1
        """,
        "last_email_date": """
            MATCH (e:Email)
            WHERE e.date IS NOT NULL
            RETURN e.date as date
            ORDER BY e.date DESC
            LIMIT 1
        """,
        "busiest_month": """
            MATCH (e:Email)
            WHERE e.date IS NOT NULL
            WITH datetime(e.date) as date
            RETURN date.year + '-' + date.month as month, count(*) as count
            ORDER BY count DESC
            LIMIT 1
        """,
        "average_thread_length": """
            MATCH (t:Thread)<-[:BELONGS_TO]-(e:Email)
            WITH t, count(e) as thread_length
            RETURN avg(thread_length) as avg_length
        """
    }
    
    with driver.session() as session:
        for metric_name, query in queries.items():
            try:
                result = session.run(query)
                record = result.single()
                if record:
                    if metric_name == "busiest_month":
                        metrics[metric_name] = {
                            "month": record["month"],
                            "count": record["count"]
                        }
                    elif metric_name == "average_thread_length":
                        metrics[metric_name] = record["avg_length"]
                    else:
                        metrics[metric_name] = record["date"]
                else:
                    metrics[metric_name] = None
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                metrics[metric_name] = None
                
    return metrics

def compute_network_metrics(driver):
    """
    Compute network analysis metrics.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        Dictionary with network metrics
    """
    metrics = {}
    
    queries = {
        "most_active_person": """
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:SENT]->(sent:Email)
            OPTIONAL MATCH (p)-[:RECEIVED]->(received:Email)
            WITH p, count(distinct sent) + count(distinct received) as activity
            RETURN p.email as email, p.name as name, activity
            ORDER BY activity DESC
            LIMIT 1
        """,
        "largest_community": """
            MATCH (p1:Person)-[:SENT]->(:Email)<-[:RECEIVED]-(p2:Person)
            WITH p1, p2, count(*) as comms
            WHERE comms > 5
            WITH p1, collect(distinct p2) as connections
            RETURN p1.email as central_person, size(connections) as community_size
            ORDER BY community_size DESC
            LIMIT 1
        """,
        "top_domains": """
            MATCH (p:Person)-[:BELONGS_TO]->(d:Domain)
            WITH d, count(p) as person_count
            RETURN collect({domain: d.name, count: person_count}) as domains
            ORDER BY person_count DESC
            LIMIT 5
        """
    }
    
    with driver.session() as session:
        for metric_name, query in queries.items():
            try:
                result = session.run(query)
                record = result.single()
                if record:
                    if metric_name == "most_active_person":
                        metrics[metric_name] = {
                            "email": record["email"],
                            "name": record["name"],
                            "activity": record["activity"]
                        }
                    elif metric_name == "largest_community":
                        metrics[metric_name] = {
                            "central_person": record["central_person"],
                            "size": record["community_size"]
                        }
                    elif metric_name == "top_domains":
                        metrics[metric_name] = record["domains"]
                    else:
                        metrics[metric_name] = dict(record)
                else:
                    metrics[metric_name] = None
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                metrics[metric_name] = None
                
    return metrics

def compute_content_metrics(driver):
    """
    Compute content-based metrics.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        Dictionary with content metrics
    """
    metrics = {}
    
    queries = {
        "average_email_length": """
            MATCH (e:Email)
            WHERE e.body IS NOT NULL
            RETURN avg(size(e.body)) as avg_length
        """,
        "top_subject_words": """
            MATCH (e:Email)
            WHERE e.subject IS NOT NULL
            WITH apoc.text.words(toLower(e.subject)) as words
            UNWIND words as word
            WHERE size(word) > 3
            RETURN word, count(*) as count
            ORDER BY count DESC
            LIMIT 10
        """,
        "common_labels": """
            MATCH (e:Email)-[:HAS_LABEL]->(l:Label)
            WITH l, count(e) as email_count
            RETURN collect({label: l.name, count: email_count}) as labels
            ORDER BY email_count DESC
            LIMIT 5
        """
    }
    
    with driver.session() as session:
        for metric_name, query in queries.items():
            try:
                if metric_name == "top_subject_words":
                    # Special handling for word frequency
                    result = session.run(query)
                    words = [(record["word"], record["count"]) for record in result]
                    metrics[metric_name] = words
                else:
                    result = session.run(query)
                    record = result.single()
                    if record:
                        if metric_name == "common_labels":
                            metrics[metric_name] = record["labels"]
                        else:
                            metrics[metric_name] = record["avg_length"]
                    else:
                        metrics[metric_name] = None
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                metrics[metric_name] = None
                
    return metrics 