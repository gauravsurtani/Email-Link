"""
Semantic Analysis module - Analyze extracted semantic data from emails.

This module provides functions to analyze and visualize the semantic data
extracted from emails, such as entity distributions, common actions, etc.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Counter, Optional
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """
    Analyze semantic data extracted from emails.
    """
    
    def __init__(self, data_dir: Optional[str] = None, data: Optional[List[Dict]] = None):
        """
        Initialize the semantic analyzer.
        
        Args:
            data_dir: Directory containing the semantic data files
            data: List of email data dictionaries with semantic information
        """
        if data is not None:
            self.emails = data
        elif data_dir is not None:
            self.emails = self._load_data(data_dir)
        else:
            self.emails = []
            logger.warning("No data provided. Use load_data() to load semantic data.")
    
    def _load_data(self, data_dir: str) -> List[Dict]:
        """
        Load semantic data from JSON files.
        
        Args:
            data_dir: Directory containing the semantic data files
            
        Returns:
            List of email dictionaries with semantic data
        """
        data_path = Path(data_dir)
        
        # Look for the complete processed file first
        complete_file = data_path / "processed_emails_complete.json"
        if complete_file.exists():
            with open(complete_file, 'r', encoding='utf-8') as f:
                logger.info(f"Loaded complete dataset from {complete_file}")
                return json.load(f)
        
        # If no complete file, load the latest batch file
        batch_files = list(data_path.glob("processed_batch_*.json"))
        if batch_files:
            # Sort by batch number to find the latest
            latest_batch = sorted(batch_files, key=lambda f: int(f.stem.split('_')[-1]))[-1]
            with open(latest_batch, 'r', encoding='utf-8') as f:
                logger.info(f"Loaded latest batch from {latest_batch}")
                return json.load(f)
        
        # If no batch files, load individual email files
        email_files = list(data_path.glob("email_*.json"))
        if email_files:
            emails = []
            for file in tqdm(email_files, desc="Loading email files"):
                with open(file, 'r', encoding='utf-8') as f:
                    emails.append(json.load(f))
            logger.info(f"Loaded {len(emails)} individual email files")
            return emails
        
        logger.warning(f"No semantic data files found in {data_dir}")
        return []
    
    def load_data(self, data_dir: str) -> None:
        """
        Load semantic data from JSON files.
        
        Args:
            data_dir: Directory containing the semantic data files
        """
        self.emails = self._load_data(data_dir)
        logger.info(f"Loaded {len(self.emails)} emails with semantic data")
    
    def get_email_types(self) -> Dict[str, int]:
        """
        Get the distribution of email types.
        
        Returns:
            Dictionary mapping email types to counts
        """
        types = Counter()
        for email in self.emails:
            if 'entities' in email and 'type' in email['entities']:
                types[email['entities']['type']] += 1
        return dict(types)
    
    def get_entity_counts(self, entity_type: str) -> Dict[str, int]:
        """
        Get counts of a specific entity type across all emails.
        
        Args:
            entity_type: Type of entity to count (organizations, people, etc.)
            
        Returns:
            Dictionary mapping entity names to counts
        """
        counts = Counter()
        for email in self.emails:
            if 'entities' in email and entity_type in email['entities']:
                for entity in email['entities'][entity_type]:
                    counts[entity] += 1
        return dict(counts.most_common(20))  # Return top 20
    
    def get_action_counts(self) -> Dict[str, int]:
        """
        Get counts of actions across all emails.
        
        Returns:
            Dictionary mapping actions to counts
        """
        return self.get_entity_counts('actions')
    
    def get_organization_counts(self) -> Dict[str, int]:
        """
        Get counts of organizations across all emails.
        
        Returns:
            Dictionary mapping organizations to counts
        """
        return self.get_entity_counts('organizations')
    
    def get_people_counts(self) -> Dict[str, int]:
        """
        Get counts of people across all emails.
        
        Returns:
            Dictionary mapping people to counts
        """
        return self.get_entity_counts('people')
    
    def get_location_counts(self) -> Dict[str, int]:
        """
        Get counts of locations across all emails.
        
        Returns:
            Dictionary mapping locations to counts
        """
        return self.get_entity_counts('locations')
    
    def get_emails_by_type(self, email_type: str) -> List[Dict]:
        """
        Get all emails of a specific type.
        
        Args:
            email_type: Type of email to filter by
            
        Returns:
            List of email dictionaries
        """
        return [
            email for email in self.emails
            if 'entities' in email 
            and 'type' in email['entities'] 
            and email['entities']['type'] == email_type
        ]
    
    def get_emails_mentioning_entity(self, entity: str, entity_type: Optional[str] = None) -> List[Dict]:
        """
        Get all emails mentioning a specific entity.
        
        Args:
            entity: Name of the entity to search for
            entity_type: Type of entity (organizations, people, etc.), or None to search all types
            
        Returns:
            List of email dictionaries
        """
        result = []
        entity_lower = entity.lower()
        
        for email in self.emails:
            if 'entities' not in email:
                continue
                
            if entity_type:
                # Search in specific entity type
                if entity_type in email['entities']:
                    entities = [e.lower() for e in email['entities'][entity_type]]
                    if entity_lower in entities:
                        result.append(email)
            else:
                # Search in all entity types
                found = False
                for e_type in ['organizations', 'people', 'products', 'locations']:
                    if e_type in email['entities']:
                        entities = [e.lower() for e in email['entities'][e_type]]
                        if entity_lower in entities:
                            found = True
                            break
                if found:
                    result.append(email)
        
        return result
    
    def visualize_email_types(self, title: str = "Email Type Distribution") -> None:
        """
        Visualize the distribution of email types as a pie chart.
        
        Args:
            title: Title for the chart
        """
        types = self.get_email_types()
        
        # Filter out empty or very small categories
        types = {k: v for k, v in types.items() if v >= 2}
        
        # Create the pie chart
        plt.figure(figsize=(10, 7))
        plt.pie(
            types.values(), 
            labels=types.keys(), 
            autopct='%1.1f%%', 
            startangle=90
        )
        plt.axis('equal')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_entity_counts(self, entity_type: str, title: Optional[str] = None, top_n: int = 10) -> None:
        """
        Visualize counts of a specific entity type as a bar chart.
        
        Args:
            entity_type: Type of entity to visualize (organizations, people, etc.)
            title: Title for the chart
            top_n: Number of top entities to show
        """
        counts = self.get_entity_counts(entity_type)
        
        # Get top N entities
        top_entities = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create the bar chart
        plt.figure(figsize=(12, 6))
        names, values = zip(*top_entities) if top_entities else ([], [])
        plt.bar(names, values)
        plt.xticks(rotation=45, ha='right')
        plt.title(title or f"Top {top_n} {entity_type.capitalize()}")
        plt.tight_layout()
        plt.show()


def analyze_semantic_data(data_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Analyze semantic data from emails and generate reports/visualizations.
    
    Args:
        data_dir: Directory containing the semantic data files
        output_dir: Directory to save analysis results
    """
    analyzer = SemanticAnalyzer(data_dir)
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Basic statistics
    email_types = analyzer.get_email_types()
    organization_counts = analyzer.get_organization_counts()
    people_counts = analyzer.get_people_counts()
    location_counts = analyzer.get_location_counts()
    action_counts = analyzer.get_action_counts()
    
    # Print some statistics
    print(f"Total emails analyzed: {len(analyzer.emails)}")
    print("\nEmail Type Distribution:")
    for email_type, count in email_types.items():
        print(f"  {email_type}: {count}")
    
    print("\nTop 10 Organizations:")
    for org, count in list(organization_counts.items())[:10]:
        print(f"  {org}: {count}")
    
    print("\nTop 10 People:")
    for person, count in list(people_counts.items())[:10]:
        print(f"  {person}: {count}")
    
    print("\nTop 10 Locations:")
    for location, count in list(location_counts.items())[:10]:
        print(f"  {location}: {count}")
    
    print("\nTop 10 Actions:")
    for action, count in list(action_counts.items())[:10]:
        print(f"  {action}: {count}")
    
    # Save analysis results if output directory is specified
    if output_dir:
        analysis_results = {
            "email_count": len(analyzer.emails),
            "email_types": email_types,
            "top_organizations": dict(list(organization_counts.items())[:20]),
            "top_people": dict(list(people_counts.items())[:20]),
            "top_locations": dict(list(location_counts.items())[:20]),
            "top_actions": dict(list(action_counts.items())[:20])
        }
        
        output_file = Path(output_dir) / "semantic_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis results saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze semantic data from emails")
    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Directory containing semantic data files")
    parser.add_argument("--output", "-o", type=str, default="output/analysis",
                        help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    try:
        analyze_semantic_data(args.data, args.output)
    except Exception as e:
        logger.error(f"Error in semantic analysis: {e}")
        import traceback
        traceback.print_exc() 