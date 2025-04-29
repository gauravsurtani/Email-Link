#!/usr/bin/env python3
"""
Event query module for exploring extracted events in the graph.
Provides command-line interface for querying events and using embeddings for recommendations.
"""

import os
import json
import logging
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from tabulate import tabulate
from neo4j import GraphDatabase

# Import modules
from embeddings.graph_embeddings import GraphEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store event IDs mapping between display numbers and internal IDs
_event_id_cache = {}
# Global variable to store the last selected event ID
_selected_event_id = None

class EventQuery:
    """Class for querying events from the Neo4j graph database."""
    
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
        
    def get_event_count(self):
        """
        Get the total number of events in the database.
        
        Returns:
            Integer count of events
        """
        with self.driver.session() as session:
            result = session.run("MATCH (e:Event) RETURN count(e) as count")
            return result.single()["count"]
            
    def get_recent_events(self, limit=10):
        """
        Get a list of recent events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        query = """
        MATCH (e:Event)
        WHERE e.event_date IS NOT NULL
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
        OPTIONAL MATCH (p:Person)-[:ORGANIZES]->(e)
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               l.name as location,
               p.email as organizer
        ORDER BY e.event_date DESC, e.event_time DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]
            
    def get_event_by_id(self, event_id):
        """
        Get detailed information about a specific event.
        
        Args:
            event_id: ID of the event
            
        Returns:
            Dictionary with event details or None if not found
        """
        query = """
        MATCH (e:Event {event_id: $event_id})
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
        OPTIONAL MATCH (p:Person)-[:ORGANIZES]->(e)
        OPTIONAL MATCH (a:Person)-[:ATTENDS]->(e)
        OPTIONAL MATCH (email:Email)-[:CONTAINS_EVENT]->(e)
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               e.duration as duration,
               e.end_time as end_time,
               l.name as location,
               e.virtual_platform as platform,
               e.virtual_link as link,
               e.virtual_meeting_id as meeting_id,
               e.virtual_password as password,
               p.email as organizer,
               collect(DISTINCT a.email) as attendees,
               email.message_id as email_id,
               email.subject as email_subject
        """
        
        with self.driver.session() as session:
            result = session.run(query, event_id=event_id)
            record = result.single()
            return dict(record) if record else None
            
    def search_events(self, query_text, limit=20):
        """
        Search for events with text matching in subject or related emails.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching events
        """
        query = """
        MATCH (e:Event)
        WHERE e.subject CONTAINS $query_text
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               l.name as location
        
        UNION
        
        MATCH (email:Email)-[:CONTAINS_EVENT]->(e:Event)
        WHERE email.subject CONTAINS $query_text OR email.body CONTAINS $query_text
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               l.name as location
        
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, query_text=query_text, limit=limit)
            return [dict(record) for record in result]
            
    def get_events_by_person(self, email, limit=20):
        """
        Get events associated with a specific person.
        
        Args:
            email: Email address of the person
            limit: Maximum number of results
            
        Returns:
            List of events
        """
        query = """
        MATCH (p:Person {email: $email})-[:ATTENDS|ORGANIZES]->(e:Event)
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               l.name as location,
               CASE WHEN exists((p)-[:ORGANIZES]->(e)) THEN 'Organizer' ELSE 'Attendee' END as role
        ORDER BY e.event_date DESC, e.event_time DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, email=email, limit=limit)
            return [dict(record) for record in result]
            
    def get_events_by_date_range(self, start_date, end_date, limit=50):
        """
        Get events within a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of results
            
        Returns:
            List of events
        """
        query = """
        MATCH (e:Event)
        WHERE e.event_date >= $start_date AND e.event_date <= $end_date
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
        OPTIONAL MATCH (p:Person)-[:ORGANIZES]->(e)
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               l.name as location,
               p.email as organizer
        ORDER BY e.event_date, e.event_time
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, start_date=start_date, end_date=end_date, limit=limit)
            return [dict(record) for record in result]
            
    def get_events_by_location(self, location, limit=20):
        """
        Get events at a specific location.
        
        Args:
            location: Location name or partial match
            limit: Maximum number of results
            
        Returns:
            List of events
        """
        query = """
        MATCH (e:Event)-[:LOCATED_AT]->(l:Location)
        WHERE l.name CONTAINS $location
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               l.name as location
        ORDER BY e.event_date DESC, e.event_time DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, location=location, limit=limit)
            return [dict(record) for record in result]
            
    def get_event_by_subject(self, subject, limit=5):
        """
        Get events by subject text.
        
        Args:
            subject: Text to match in event subject
            limit: Maximum number of results
            
        Returns:
            List of event dictionaries
        """
        query = """
        MATCH (e:Event)
        WHERE e.subject CONTAINS $subject
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
        OPTIONAL MATCH (p:Person)-[:ORGANIZES]->(e)
        RETURN e.event_id as id,
               e.subject as subject,
               e.event_type as type,
               e.event_date as date,
               e.event_time as time,
               l.name as location,
               p.email as organizer
        ORDER BY e.event_date DESC, e.event_time DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, subject=subject, limit=limit)
            return [dict(record) for record in result]
            
    def get_event_statistics(self):
        """
        Get statistics about events in the database.
        
        Returns:
            Dictionary with event statistics
        """
        stats = {}
        
        queries = {
            "total_events": "MATCH (e:Event) RETURN count(e) as count",
            "event_types": """
                MATCH (e:Event) 
                RETURN e.event_type as type, count(*) as count 
                ORDER BY count DESC
            """,
            "virtual_vs_physical": """
                MATCH (e:Event)
                RETURN 
                    sum(CASE WHEN e.virtual_platform IS NOT NULL THEN 1 ELSE 0 END) as virtual_count,
                    sum(CASE WHEN exists((e)-[:LOCATED_AT]->()) THEN 1 ELSE 0 END) as physical_count
            """,
            "events_per_month": """
                MATCH (e:Event)
                WHERE e.event_date IS NOT NULL
                WITH substring(e.event_date, 0, 7) as month, count(*) as count
                RETURN month, count
                ORDER BY month
            """,
            "top_organizers": """
                MATCH (p:Person)-[:ORGANIZES]->(e:Event)
                RETURN p.email as email, count(*) as count
                ORDER BY count DESC
                LIMIT 5
            """
        }
        
        with self.driver.session() as session:
            for stat_name, query in queries.items():
                result = session.run(query)
                
                if stat_name == "total_events":
                    stats[stat_name] = result.single()["count"]
                elif stat_name == "virtual_vs_physical":
                    record = result.single()
                    stats[stat_name] = {
                        "virtual": record["virtual_count"],
                        "physical": record["physical_count"]
                    }
                else:
                    stats[stat_name] = [dict(record) for record in result]
                    
        return stats

def format_event_list(events):
    """Format event list for display."""
    if not events:
        return "No events found."
        
    # Create table data
    table_data = []
    # Store the full event IDs for later reference
    global _event_id_cache
    _event_id_cache = {}
    
    for idx, event in enumerate(events, 1):
        # Store the event ID in the cache with its index
        _event_id_cache[idx] = event.get('id', 'N/A')
        
        date_str = event.get('date', 'N/A')
        time_str = event.get('time', '')
        datetime_str = f"{date_str} {time_str}".strip()
        
        row = [
            idx,  # Use index number instead of ID
            event.get('subject', 'N/A'),
            event.get('type', 'N/A'),
            datetime_str,
            event.get('location', 'N/A')
        ]
        
        # Add role if present
        if 'role' in event:
            row.append(event['role'])
            
        table_data.append(row)
    
    # Create headers
    headers = ["#", "Subject", "Type", "Date/Time", "Location"]
    if 'role' in events[0]:
        headers.append("Role")
    
    result = tabulate(table_data, headers=headers, tablefmt="grid")
    result += "\n\nTip: Use event number from the '#' column with the 'select' command instead of copying event IDs."
        
    return result

def format_event_details(event):
    """Format detailed event information for display."""
    if not event:
        return "Event not found."
        
    # Basic info section
    info = f"Event ID: {event.get('id')}\n"
    info += f"Subject: {event.get('subject')}\n"
    info += f"Type: {event.get('type')}\n"
    
    # Date and time
    date_str = event.get('date', 'N/A')
    time_str = event.get('time', '')
    
    info += f"Date: {date_str}\n"
    if time_str:
        info += f"Time: {time_str}\n"
        
    if event.get('duration'):
        info += f"Duration: {event.get('duration')}\n"
    if event.get('end_time'):
        info += f"End Time: {event.get('end_time')}\n"
    
    # Location information
    if event.get('location'):
        info += f"Location: {event.get('location')}\n"
        
    # Virtual meeting details
    if event.get('platform'):
        info += f"\nVirtual Meeting Info:\n"
        info += f"  Platform: {event.get('platform')}\n"
        if event.get('link'):
            info += f"  Link: {event.get('link')}\n"
        if event.get('meeting_id'):
            info += f"  Meeting ID: {event.get('meeting_id')}\n"
        if event.get('password'):
            info += f"  Password: {event.get('password')}\n"
    
    # People
    info += f"\nOrganizer: {event.get('organizer', 'N/A')}\n"
    
    if event.get('attendees'):
        info += f"Attendees:\n"
        for attendee in event.get('attendees'):
            info += f"  - {attendee}\n"
    
    # Email source
    if event.get('email_id'):
        info += f"\nSource Email: {event.get('email_subject')} (ID: {event.get('email_id')})\n"
    
    return info

def format_statistics(stats):
    """Format event statistics for display."""
    if not stats:
        return "No statistics available."
        
    output = "Event Statistics\n"
    output += "===============\n\n"
    
    output += f"Total Events: {stats.get('total_events', 0)}\n\n"
    
    # Event types
    if 'event_types' in stats:
        output += "Event Types:\n"
        for type_stat in stats['event_types']:
            output += f"  {type_stat['type']}: {type_stat['count']}\n"
        output += "\n"
    
    # Virtual vs Physical
    if 'virtual_vs_physical' in stats:
        vp = stats['virtual_vs_physical']
        output += "Virtual vs Physical:\n"
        output += f"  Virtual Meetings: {vp['virtual']}\n"
        output += f"  Physical Locations: {vp['physical']}\n"
        output += "\n"
    
    # Events per month
    if 'events_per_month' in stats:
        output += "Events per Month:\n"
        for month_stat in stats['events_per_month']:
            output += f"  {month_stat['month']}: {month_stat['count']}\n"
        output += "\n"
    
    # Top organizers
    if 'top_organizers' in stats:
        output += "Top Event Organizers:\n"
        for organizer in stats['top_organizers']:
            output += f"  {organizer['email']}: {organizer['count']} events\n"
    
    return output

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Query events from email graph')
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Recent events command
    recent_parser = subparsers.add_parser('recent', help='Show recent events')
    recent_parser.add_argument('--limit', type=int, default=10, help='Number of events to show')
    
    # Details command
    details_parser = subparsers.add_parser('details', help='Show details for a specific event')
    details_parser.add_argument('event_id', help='Event ID to get details for')
    
    # Select command (new)
    select_parser = subparsers.add_parser('select', help='Select an event by number from the last displayed list')
    select_parser.add_argument('event_number', type=int, help='Event number from the list to select')
    
    # Subject command (new)
    subject_parser = subparsers.add_parser('subject', help='Find events by subject text')
    subject_parser.add_argument('text', help='Subject text to search for')
    subject_parser.add_argument('--limit', type=int, default=5, help='Maximum number of results')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for events')
    search_parser.add_argument('query', help='Text to search for')
    search_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
    
    # Person command
    person_parser = subparsers.add_parser('person', help='Show events for a person')
    person_parser.add_argument('email', help='Email address of the person')
    person_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
    
    # Date range command
    date_parser = subparsers.add_parser('dates', help='Show events in a date range')
    date_parser.add_argument('start_date', help='Start date (YYYY-MM-DD)')
    date_parser.add_argument('end_date', help='End date (YYYY-MM-DD)')
    date_parser.add_argument('--limit', type=int, default=50, help='Maximum number of results')
    
    # Location command
    location_parser = subparsers.add_parser('location', help='Show events at a location')
    location_parser.add_argument('name', help='Location name or partial match')
    location_parser.add_argument('--limit', type=int, default=20, help='Maximum number of results')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show event statistics')
    
    # Similar events command (using embeddings)
    similar_parser = subparsers.add_parser('similar', help='Find similar events using embeddings')
    similar_parser.add_argument('event_id', help='Reference event ID')
    similar_parser.add_argument('--limit', type=int, default=5, help='Number of similar events to show')
    
    # Recommended events command (using embeddings)
    recommend_parser = subparsers.add_parser('recommend', help='Recommend events for a person')
    recommend_parser.add_argument('email', help='Email address of the person')
    recommend_parser.add_argument('--limit', type=int, default=5, help='Number of recommendations')
    
    # Show selected event command (new)
    show_selected_parser = subparsers.add_parser('show-selected', help='Show details of the currently selected event')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Load environment configuration
    load_dotenv()
    
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        logger.error("Neo4j password not provided in .env file")
        print("Error: Neo4j password not provided in .env file")
        return 1
    
    # Set output directory
    output_dir = os.getenv('OUTPUT_DIR', './output')
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    
    # Parse command line arguments
    args = parse_args()
    
    if not args.command:
        print("Error: No command specified. Use --help to see available commands.")
        return 1
    
    # Initialize event query
    event_query = EventQuery(neo4j_uri, neo4j_user, neo4j_password)
    
    # Access global variables
    global _event_id_cache, _selected_event_id
    
    try:
        # Process command
        if args.command == 'recent':
            events = event_query.get_recent_events(args.limit)
            print(f"\nRecent Events ({len(events)} found):")
            print(format_event_list(events))
            
        elif args.command == 'details':
            event = event_query.get_event_by_id(args.event_id)
            print("\nEvent Details:")
            print(format_event_details(event))
            
        elif args.command == 'select':
            # Select an event by its number in the displayed list
            if args.event_number not in _event_id_cache:
                print(f"Error: Event number {args.event_number} is not in the current list.")
                print("Run a query first to get a list of events, then use the number from the '#' column.")
                return 1
                
            _selected_event_id = _event_id_cache[args.event_number]
            event = event_query.get_event_by_id(_selected_event_id)
            
            print(f"\nSelected Event #{args.event_number}:")
            print(format_event_details(event))
            print("\nTip: Use 'show-selected' command to view this event again later.")
            
        elif args.command == 'show-selected':
            # Show the currently selected event
            if not _selected_event_id:
                print("No event currently selected. Use the 'select' command first.")
                return 1
                
            event = event_query.get_event_by_id(_selected_event_id)
            print("\nCurrently Selected Event:")
            print(format_event_details(event))
            
        elif args.command == 'subject':
            # Find events by subject text
            events = event_query.get_event_by_subject(args.text, args.limit)
            print(f"\nEvents with subject containing '{args.text}' ({len(events)} found):")
            print(format_event_list(events))
            
        elif args.command == 'search':
            events = event_query.search_events(args.query, args.limit)
            print(f"\nSearch Results for '{args.query}' ({len(events)} found):")
            print(format_event_list(events))
            
        elif args.command == 'person':
            events = event_query.get_events_by_person(args.email, args.limit)
            print(f"\nEvents for {args.email} ({len(events)} found):")
            print(format_event_list(events))
            
        elif args.command == 'dates':
            events = event_query.get_events_by_date_range(args.start_date, args.end_date, args.limit)
            print(f"\nEvents from {args.start_date} to {args.end_date} ({len(events)} found):")
            print(format_event_list(events))
            
        elif args.command == 'location':
            events = event_query.get_events_by_location(args.name, args.limit)
            print(f"\nEvents at location '{args.name}' ({len(events)} found):")
            print(format_event_list(events))
            
        elif args.command == 'stats':
            stats = event_query.get_event_statistics()
            print(format_statistics(stats))
            
        elif args.command == 'similar':
            # Check if embeddings exist
            entity_embeddings = os.path.join(embeddings_dir, 'entity_embeddings.csv')
            if not os.path.exists(entity_embeddings):
                print("Error: No embeddings found. Run event_pipeline.py first to generate embeddings.")
                return 1
                
            # If the argument is a number and we have an event cache, use the cached ID
            event_id = args.event_id
            if event_id.isdigit() and int(event_id) in _event_id_cache:
                event_id = _event_id_cache[int(event_id)]
                
            # Load embeddings
            embeddings = GraphEmbeddings(neo4j_uri, neo4j_user, neo4j_password, output_dir=embeddings_dir)
            if not embeddings.load_embeddings():
                print("Error: Failed to load embeddings.")
                return 1
                
            # Get event details to confirm it exists
            event = event_query.get_event_by_id(event_id)
            if not event:
                print(f"Error: Event with ID {event_id} not found.")
                return 1
                
            # Find similar entities
            entity_id = f"Event_{event_id}"
            similar_entities = embeddings.query_similar_entities(entity_id, args.limit)
            
            # Filter for events only
            similar_events = []
            for entity in similar_entities:
                if entity['entity'].startswith('Event_'):
                    event_id = entity['entity'].split('_', 1)[1]
                    event_details = event_query.get_event_by_id(event_id)
                    if event_details:
                        event_details['similarity'] = entity['similarity']
                        similar_events.append(event_details)
            
            print(f"\nEvents Similar to '{event['subject']}' ({len(similar_events)} found):")
            
            # Update the event cache
            _event_id_cache = {}
            
            # Create table data
            table_data = []
            for idx, event in enumerate(similar_events, 1):
                # Store the event ID in the cache with its index
                _event_id_cache[idx] = event.get('id', 'N/A')
                
                date_str = event.get('date', 'N/A')
                time_str = event.get('time', '')
                datetime_str = f"{date_str} {time_str}".strip()
                
                row = [
                    idx,  # Use index number instead of ID
                    event.get('subject', 'N/A'),
                    event.get('type', 'N/A'),
                    datetime_str,
                    event.get('location', 'N/A'),
                    f"{event.get('similarity', 0):.4f}"
                ]
                
                table_data.append(row)
            
            # Create headers
            headers = ["#", "Subject", "Type", "Date/Time", "Location", "Similarity"]
            
            result = tabulate(table_data, headers=headers, tablefmt="grid")
            result += "\n\nTip: Use event number from the '#' column with the 'select' command instead of copying event IDs."
            print(result)
            
        elif args.command == 'recommend':
            # Check if embeddings exist
            entity_embeddings = os.path.join(embeddings_dir, 'entity_embeddings.csv')
            if not os.path.exists(entity_embeddings):
                print("Error: No embeddings found. Run event_pipeline.py first to generate embeddings.")
                return 1
                
            # Load embeddings
            embeddings = GraphEmbeddings(neo4j_uri, neo4j_user, neo4j_password, output_dir=embeddings_dir)
            if not embeddings.load_embeddings():
                print("Error: Failed to load embeddings.")
                return 1
                
            # Find events for person
            events = embeddings.find_events_for_person(args.email, args.limit)
            
            print(f"\nRecommended Events for {args.email} ({len(events)} found):")
            
            # Update the event cache
            _event_id_cache = {}
            
            # Create table data
            table_data = []
            for idx, event in enumerate(events, 1):
                # Store the event ID in the cache with its index
                _event_id_cache[idx] = event.get('id', 'N/A')
                
                date_str = event.get('date', 'N/A')
                time_str = event.get('time', '')
                datetime_str = f"{date_str} {time_str}".strip()
                
                row = [
                    idx,  # Use index number instead of ID
                    event.get('subject', 'N/A'),
                    event.get('type', 'N/A'),
                    datetime_str,
                    event.get('location', 'N/A') or event.get('platform', 'N/A'),
                    f"{event.get('similarity', 0):.4f}"
                ]
                
                table_data.append(row)
            
            # Create headers
            headers = ["#", "Subject", "Type", "Date/Time", "Location", "Relevance"]
            
            result = tabulate(table_data, headers=headers, tablefmt="grid")
            result += "\n\nTip: Use event number from the '#' column with the 'select' command instead of copying event IDs."
            print(result)
            
        # Close Neo4j connection
        event_query.close()
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 