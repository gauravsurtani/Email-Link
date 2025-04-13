#!/usr/bin/env python3
"""
Interactive mode for the Email Graph Agent.
This allows users to interact with the email graph through natural language queries.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path if running as script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import configuration
from config import get_config
from dotenv import load_dotenv
from agent.agent import EmailGraphAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_formatted_result(result):
    """
    Format and print agent result in a user-friendly way.
    
    Args:
        result: Result dictionary from agent
    """
    print("\n" + "=" * 80)
    
    # Print message
    if 'message' in result:
        print(f"\n{result['message']}\n")
    
    # Print different output based on result type
    result_type = result.get('type')
    
    if result_type == 'overview':
        data = result['data']
        print(f"Total Emails: {data['total_emails']}")
        print(f"Total People: {data['total_people']}")
        
        print("\nTop Senders:")
        for sender in data['top_senders']:
            print(f"  {sender['name']} ({sender['email']}): {sender['sent_count']} emails")
            
        print("\nTop Receivers:")
        for receiver in data['top_receivers']:
            print(f"  {receiver['name']} ({receiver['email']}): {receiver['received_count']} emails")
            
        print("\nTop Domains:")
        for domain in data['top_domains']:
            print(f"  {domain['domain']}: {domain['person_count']} people")
    
    elif result_type == 'search':
        print(f"Query: {result['query']}")
        print(f"Found {result['count']} results\n")
        
        for i, email in enumerate(result['data'], 1):
            print(f"{i}. Subject: {email['subject']}")
            print(f"   From: {email['sender']}")
            print(f"   Date: {email['date']}")
            print(f"   Excerpt: {email['excerpt'][:100]}...")
            if i < len(result['data']):
                print()
    
    elif result_type == 'person':
        person = result['data']
        print(f"Name: {person['name']}")
        print(f"Email: {person['email']}")
        print(f"Domain: {person['domain']}")
        print(f"Sent: {person['sent_count']} emails")
        print(f"Received: {person['received_count']} emails")
        
        print("\nTop Communication Partners:")
        for partner in person['top_partners']:
            if partner['email']:
                print(f"  {partner['name']} ({partner['email']}): {partner['count']} communications")
    
    elif result_type == 'connection':
        conn = result['data']
        print(f"Connection between {conn['person1']['email']} and {conn['person2']['email']}")
        
        direct = conn['direct_communication']
        print(f"\nDirect Communication:")
        print(f"  From {direct['from']} to {direct['to']}: {direct['sent']} emails")
        print(f"  From {direct['to']} to {direct['from']}: {direct['received']} emails")
        print(f"  Total: {direct['total']} emails")
        
        if conn['connection_paths']:
            print(f"\nConnection Paths ({len(conn['connection_paths'])} found):")
            for i, path in enumerate(conn['connection_paths'][:3], 1):
                print(f"  Path {i}:")
                for step in path:
                    print(f"    {step['start']} --[{step['type']}]--> {step['end']}")
        else:
            print("\nNo connection paths found")
    
    elif result_type == 'thread':
        thread = result['data']
        print(f"Thread ID: {thread['thread_id']}")
        print(f"Subject: {thread['subject']}")
        print(f"Emails: {thread['email_count']}")
        print(f"Time span: {thread['start_date']} to {thread['end_date']}")
        print(f"Participants: {', '.join(thread['participants'])}")
        
        print("\nEmails in Thread:")
        for i, email in enumerate(thread['emails'], 1):
            print(f"  {i}. From: {email['sender']}")
            print(f"     Date: {email['date']}")
            print(f"     To: {', '.join(email['receivers'])}")
            if i < len(thread['emails']):
                print()
    
    elif result_type == 'similar_threads':
        print(f"Subject pattern: {result['subject']}")
        print(f"Found {result['count']} similar threads\n")
        
        for i, thread in enumerate(result['data'], 1):
            print(f"{i}. Thread ID: {thread['thread_id']}")
            print(f"   Subject: {thread['subject']}")
            print(f"   Emails: {thread['email_count']}")
            if i < len(result['data']):
                print()
    
    elif result_type == 'time_analysis':
        print(f"Time interval: {result['interval']}")
        print(f"Looking at last {result['limit']} {result['interval']}s\n")
        
        for period in result['data']:
            print(f"  {period['period']}: {period['email_count']} emails")
    
    elif result_type == 'domain_analysis':
        domain = result['data']
        print(f"Domain: {domain['domain']}")
        print(f"People: {domain['person_count']}")
        print(f"Emails sent: {domain['emails_sent']}")
        print(f"Emails received: {domain['emails_received']}")
        
        print("\nTop People in Domain:")
        for person in domain['top_people']:
            print(f"  {person['name']} ({person['email']}): {person['activity']} communications")
    
    elif result_type == 'error':
        print(f"Error: {result['message']}")
    
    else:
        # Default formatting for other result types
        print(json.dumps(result, indent=2))
    
    print("\n" + "=" * 80)

def interactive_mode():
    """Run the Email Graph Agent in interactive mode."""
    print("\nEmail Graph Agent Interactive Mode")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    # Load configuration
    load_dotenv()
    config = get_config()
    
    # Validate Neo4j connection details
    if not all([config.get('neo4j_uri'), config.get('neo4j_user'), config.get('neo4j_password')]):
        print("Error: Missing Neo4j connection details in .env file.")
        print("Please create a .env file with NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD.")
        return 1
    
    try:
        # Initialize agent
        agent = EmailGraphAgent(
            config['neo4j_uri'],
            config['neo4j_user'],
            config['neo4j_password']
        )
        
        print("\nAgent initialized successfully. Connected to Neo4j database.")
        print("Here are some example queries you can try:")
        print("  - Show me an overview of my emails")
        print("  - Search for emails containing 'meeting'")
        print("  - Tell me about person@example.com")
        print("  - Find connections between person1@example.com and person2@example.com")
        print("  - Analyze thread with subject 'Project Update'")
        print("  - Show email volume over time")
        print("  - Analyze communications with domain example.com")
        
        # Interactive loop
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Process query
                result = agent.process_query(query)
                
                # Print formatted result
                print_formatted_result(result)
                
            except KeyboardInterrupt:
                print("\nInterrupted. Use 'quit' to exit properly.")
            except Exception as e:
                print(f"Error processing query: {e}")
        
        # Clean up
        agent.close()
        return 0
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return 1

if __name__ == "__main__":
    exit(interactive_mode()) 