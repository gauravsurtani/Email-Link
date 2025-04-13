"""
Schema module for defining the Neo4j graph database schema for email data.
"""

import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_constraints(driver):
    """
    Create database constraints for unique identifiers.
    
    Args:
        driver: A Neo4j driver instance
    """
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Email) REQUIRE e.message_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Thread) REQUIRE t.thread_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label) REQUIRE l.name IS UNIQUE"
    ]
    
    logger.info("Creating database constraints")
    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                logger.error(f"Error creating constraint: {e}")
    logger.info("Database constraints created")

def create_indexes(driver):
    """
    Create indexes for better query performance.
    
    Args:
        driver: A Neo4j driver instance
    """
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (e:Email) ON (e.date)",
        "CREATE INDEX IF NOT EXISTS FOR (e:Email) ON (e.subject)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
        "CREATE INDEX IF NOT EXISTS FOR (d:Domain) ON (d.name)"
    ]
    
    logger.info("Creating database indexes")
    with driver.session() as session:
        for index in indexes:
            try:
                session.run(index)
            except Exception as e:
                logger.error(f"Error creating index: {e}")
    logger.info("Database indexes created")

def clear_database(driver, confirm=True):
    """
    Clear all data from the Neo4j database.
    
    Args:
        driver: A Neo4j driver instance
        confirm: Boolean to confirm deletion (default: True)
    
    Returns:
        Boolean indicating success
    """
    if not confirm:
        logger.warning("Database clear operation aborted")
        return False
    
    logger.warning("Clearing all data from the database")
    with driver.session() as session:
        try:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

def setup_database(driver, clear=False):
    """
    Set up the Neo4j database schema.
    
    Args:
        driver: A Neo4j driver instance
        clear: Boolean to clear database before setup (default: False)
    """
    if clear:
        clear_database(driver)
    
    create_constraints(driver)
    create_indexes(driver) 