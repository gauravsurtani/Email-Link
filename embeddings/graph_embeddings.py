"""
Module for generating and using knowledge graph embeddings.
Uses TransE and other models to create vector representations of entities and relations.
"""

import os
import numpy as np
import torch
import pandas as pd
import logging
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Set
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KGTripleDataset(Dataset):
    """Dataset for knowledge graph triples."""
    
    def __init__(self, triples, entity_to_idx, relation_to_idx):
        """
        Initialize the dataset.
        
        Args:
            triples: List of (head, relation, tail) tuples
            entity_to_idx: Dictionary mapping entity IDs to indices
            relation_to_idx: Dictionary mapping relation types to indices
        """
        self.triples = triples
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        
    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        head, rel, tail = self.triples[idx]
        return (
            torch.tensor(self.entity_to_idx[head], dtype=torch.long),
            torch.tensor(self.relation_to_idx[rel], dtype=torch.long),
            torch.tensor(self.entity_to_idx[tail], dtype=torch.long)
        )

class TransE(nn.Module):
    """TransE model for knowledge graph embeddings."""
    
    def __init__(self, entity_count, relation_count, embedding_dim=100, margin=1.0):
        """
        Initialize the TransE model.
        
        Args:
            entity_count: Number of entities in the knowledge graph
            relation_count: Number of relation types
            embedding_dim: Dimension of the embeddings
            margin: Margin for the loss function
        """
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Initialize entity embeddings
        self.entity_embeddings = nn.Embedding(entity_count, embedding_dim)
        self.entity_embeddings.weight.data.uniform_(-6/np.sqrt(embedding_dim), 
                                                   6/np.sqrt(embedding_dim))
        self.entity_embeddings.weight.data = nn.functional.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
        
        # Initialize relation embeddings
        self.relation_embeddings = nn.Embedding(relation_count, embedding_dim)
        self.relation_embeddings.weight.data.uniform_(-6/np.sqrt(embedding_dim), 
                                                     6/np.sqrt(embedding_dim))
        
    def forward(self, heads, relations, tails):
        """
        Forward pass for the TransE model.
        
        Args:
            heads: Indices of head entities
            relations: Indices of relations
            tails: Indices of tail entities
            
        Returns:
            Embedding distances
        """
        head_embeddings = self.entity_embeddings(heads)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tails)
        
        # TransE scoring function: ||h + r - t||
        scores = torch.norm(head_embeddings + relation_embeddings - tail_embeddings, p=1, dim=1)
        return scores
        
    def get_corrupted_scores(self, heads, relations, tails):
        """
        Calculate scores for corrupted triples (used for training).
        
        Args:
            heads: Indices of head entities
            relations: Indices of relations
            tails: Indices of tail entities
            
        Returns:
            Scores for corrupted triples
        """
        batch_size = heads.size(0)
        
        # Create corrupted triples by replacing either heads or tails
        # Half of the batch corrupts heads, half corrupts tails
        corrupt_heads = torch.randint(0, self.entity_count, (batch_size // 2,), device=heads.device)
        corrupt_tails = torch.randint(0, self.entity_count, (batch_size - batch_size // 2,), device=tails.device)
        
        # Original indices
        head_indices = torch.arange(0, batch_size // 2, device=heads.device)
        tail_indices = torch.arange(batch_size // 2, batch_size, device=heads.device)
        
        # Create corrupted batches
        corrupt_head_batch = torch.cat([corrupt_heads, relations[head_indices], tails[head_indices]], dim=0)
        corrupt_tail_batch = torch.cat([heads[tail_indices], relations[tail_indices], corrupt_tails], dim=0)
        
        # Calculate scores
        corrupt_head_scores = self.forward(corrupt_heads, relations[head_indices], tails[head_indices])
        corrupt_tail_scores = self.forward(heads[tail_indices], relations[tail_indices], corrupt_tails)
        
        return torch.cat([corrupt_head_scores, corrupt_tail_scores])
        
    def loss(self, pos_scores, neg_scores):
        """
        TransE margin-based ranking loss.
        
        Args:
            pos_scores: Scores for positive triples
            neg_scores: Scores for negative triples
            
        Returns:
            Loss value
        """
        return torch.mean(torch.relu(self.margin + pos_scores - neg_scores))

class GraphEmbeddings:
    """Class for generating and using knowledge graph embeddings."""
    
    def __init__(self, uri, username, password, output_dir="./embeddings"):
        """
        Initialize the graph embeddings manager.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            output_dir: Directory for saving embeddings
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.output_dir = output_dir
        self.entity_to_idx = {}
        self.relation_to_idx = {}
        self.idx_to_entity = {}
        self.idx_to_relation = {}
        self.model = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def extract_triples(self, include_event_nodes=True):
        """
        Extract knowledge graph triples from Neo4j.
        
        Args:
            include_event_nodes: Whether to include Event nodes
            
        Returns:
            List of triples and mappings for entities and relations
        """
        logger.info("Extracting triples from Neo4j")
        
        # Define the types of nodes and relationships to extract
        node_types = ["Person", "Email", "Domain", "Thread", "Label", "Attachment"]
        if include_event_nodes:
            node_types.extend(["Event", "Location"])
            
        # Extract all relationships
        with self.driver.session() as session:
            # Query to get all triples
            query = """
            MATCH (s)-[r]->(o)
            WHERE any(label IN labels(s) WHERE label IN $node_types)
            AND any(label IN labels(o) WHERE label IN $node_types)
            RETURN 
                id(s) AS source_id,
                labels(s)[0] as source_type,
                CASE
                    WHEN 'Person' IN labels(s) THEN s.email
                    WHEN 'Email' IN labels(s) THEN s.message_id
                    WHEN 'Thread' IN labels(s) THEN s.thread_id
                    WHEN 'Domain' IN labels(s) THEN s.name
                    WHEN 'Label' IN labels(s) THEN s.name
                    WHEN 'Event' IN labels(s) THEN s.event_id
                    WHEN 'Location' IN labels(s) THEN s.name
                    ELSE toString(id(s))
                END AS source_name,
                type(r) AS relation,
                id(o) AS target_id,
                labels(o)[0] as target_type,
                CASE
                    WHEN 'Person' IN labels(o) THEN o.email
                    WHEN 'Email' IN labels(o) THEN o.message_id
                    WHEN 'Thread' IN labels(o) THEN o.thread_id
                    WHEN 'Domain' IN labels(o) THEN o.name
                    WHEN 'Label' IN labels(o) THEN o.name
                    WHEN 'Event' IN labels(o) THEN o.event_id
                    WHEN 'Location' IN labels(o) THEN o.name
                    ELSE toString(id(o))
                END AS target_name
            """
            
            result = session.run(query, node_types=node_types)
            triples = []
            
            # Entity and relation index dictionaries
            entities = set()
            relations = set()
            
            # Process results
            for record in result:
                source_id = f"{record['source_type']}_{record['source_name']}"
                relation = record['relation']
                target_id = f"{record['target_type']}_{record['target_name']}"
                
                triples.append((source_id, relation, target_id))
                entities.add(source_id)
                entities.add(target_id)
                relations.add(relation)
            
            logger.info(f"Extracted {len(triples)} triples with {len(entities)} entities and {len(relations)} relation types")
            
            # Create entity and relation mappings
            self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entities))}
            self.relation_to_idx = {rel: idx for idx, rel in enumerate(sorted(relations))}
            self.idx_to_entity = {idx: entity for entity, idx in self.entity_to_idx.items()}
            self.idx_to_relation = {idx: rel for rel, idx in self.relation_to_idx.items()}
            
            return triples, self.entity_to_idx, self.relation_to_idx
            
    def train_transe(self, triples, entity_to_idx, relation_to_idx, 
                    embedding_dim=100, batch_size=128, epochs=100, lr=0.001):
        """
        Train a TransE model on the knowledge graph.
        
        Args:
            triples: List of (head, relation, tail) tuples
            entity_to_idx: Dictionary mapping entity IDs to indices
            relation_to_idx: Dictionary mapping relation types to indices
            embedding_dim: Dimension of embeddings
            batch_size: Training batch size
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Trained TransE model
        """
        logger.info(f"Training TransE model with {len(triples)} triples, dim={embedding_dim}")
        
        # Create dataset and dataloader
        dataset = KGTripleDataset(triples, entity_to_idx, relation_to_idx)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransE(
            entity_count=len(entity_to_idx),
            relation_count=len(relation_to_idx),
            embedding_dim=embedding_dim
        ).to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (heads, relations, tails) in enumerate(progress_bar):
                heads, relations, tails = heads.to(device), relations.to(device), tails.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                pos_scores = model(heads, relations, tails)
                neg_scores = model.get_corrupted_scores(heads, relations, tails)
                
                # Calculate loss
                loss = model.loss(pos_scores, neg_scores)
                total_loss += loss.item()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Normalize entity embeddings after update
                with torch.no_grad():
                    model.entity_embeddings.weight.data = nn.functional.normalize(
                        model.entity_embeddings.weight.data, p=2, dim=1
                    )
                
                # Update progress bar
                progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        # Save the trained model
        self.model = model
        self.save_embeddings()
        
        return model
    
    def save_embeddings(self):
        """Save entity and relation embeddings to files."""
        if self.model is None:
            logger.error("No trained model available to save embeddings from")
            return
            
        # Save entity embeddings
        entity_embeddings = self.model.entity_embeddings.weight.detach().cpu().numpy()
        entity_df = pd.DataFrame(entity_embeddings)
        entity_df['entity'] = [self.idx_to_entity[i] for i in range(len(self.idx_to_entity))]
        entity_df.to_csv(os.path.join(self.output_dir, 'entity_embeddings.csv'), index=False)
        
        # Save relation embeddings
        relation_embeddings = self.model.relation_embeddings.weight.detach().cpu().numpy()
        relation_df = pd.DataFrame(relation_embeddings)
        relation_df['relation'] = [self.idx_to_relation[i] for i in range(len(self.idx_to_relation))]
        relation_df.to_csv(os.path.join(self.output_dir, 'relation_embeddings.csv'), index=False)
        
        # Save mappings
        with open(os.path.join(self.output_dir, 'entity_mapping.csv'), 'w') as f:
            for entity, idx in self.entity_to_idx.items():
                f.write(f"{entity},{idx}\n")
                
        with open(os.path.join(self.output_dir, 'relation_mapping.csv'), 'w') as f:
            for relation, idx in self.relation_to_idx.items():
                f.write(f"{relation},{idx}\n")
        
        logger.info(f"Saved embeddings to {self.output_dir}")
        
    def load_embeddings(self, entity_file=None, relation_file=None):
        """
        Load pre-trained embeddings from files.
        
        Args:
            entity_file: Path to entity embeddings file
            relation_file: Path to relation embeddings file
            
        Returns:
            Boolean indicating success
        """
        entity_file = entity_file or os.path.join(self.output_dir, 'entity_embeddings.csv')
        relation_file = relation_file or os.path.join(self.output_dir, 'relation_embeddings.csv')
        
        if not os.path.exists(entity_file) or not os.path.exists(relation_file):
            logger.error("Embedding files not found")
            return False
            
        try:
            # Load entity embeddings
            entity_df = pd.read_csv(entity_file)
            entity_vectors = entity_df.drop('entity', axis=1).values
            entities = entity_df['entity'].tolist()
            self.entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
            self.idx_to_entity = {idx: entity for entity, idx in self.entity_to_idx.items()}
            
            # Load relation embeddings
            relation_df = pd.read_csv(relation_file)
            relation_vectors = relation_df.drop('relation', axis=1).values
            relations = relation_df['relation'].tolist()
            self.relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}
            self.idx_to_relation = {idx: relation for relation, idx in self.relation_to_idx.items()}
            
            # Create model with loaded embeddings
            embedding_dim = entity_vectors.shape[1]
            self.model = TransE(
                entity_count=len(self.entity_to_idx),
                relation_count=len(self.relation_to_idx),
                embedding_dim=embedding_dim
            )
            
            # Load weights into the model
            with torch.no_grad():
                self.model.entity_embeddings.weight.data = torch.tensor(entity_vectors, dtype=torch.float32)
                self.model.relation_embeddings.weight.data = torch.tensor(relation_vectors, dtype=torch.float32)
                
            logger.info(f"Loaded embeddings: {len(self.entity_to_idx)} entities, {len(self.relation_to_idx)} relations, dim={embedding_dim}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def query_similar_entities(self, entity_id, top_k=10):
        """
        Find entities similar to a given entity.
        
        Args:
            entity_id: ID of the entity to find similar entities to
            top_k: Number of similar entities to return
            
        Returns:
            List of similar entities with similarity scores
        """
        if self.model is None:
            logger.error("No model loaded. Train a model or load embeddings first.")
            return []
            
        if entity_id not in self.entity_to_idx:
            logger.error(f"Entity {entity_id} not found in the knowledge graph")
            return []
            
        # Get entity embedding
        entity_idx = self.entity_to_idx[entity_id]
        entity_embedding = self.model.entity_embeddings.weight[entity_idx].detach().cpu().numpy()
        
        # Calculate cosine similarity with all other entities
        all_embeddings = self.model.entity_embeddings.weight.detach().cpu().numpy()
        similarities = np.dot(all_embeddings, entity_embedding) / (
            np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(entity_embedding)
        )
        
        # Get top-k similar entities (excluding the query entity itself)
        similar_idxs = np.argsort(similarities)[::-1][1:top_k+1]
        similar_entities = []
        
        for idx in similar_idxs:
            similar_entities.append({
                'entity': self.idx_to_entity[idx],
                'similarity': float(similarities[idx])
            })
            
        return similar_entities
        
    def predict_missing_links(self, head_id, relation_type, top_k=10):
        """
        Predict missing links in the knowledge graph.
        
        Args:
            head_id: ID of the head entity
            relation_type: Type of relation
            top_k: Number of top predictions to return
            
        Returns:
            List of predicted tail entities with scores
        """
        if self.model is None:
            logger.error("No model loaded. Train a model or load embeddings first.")
            return []
            
        if head_id not in self.entity_to_idx or relation_type not in self.relation_to_idx:
            logger.error(f"Head entity or relation not found in the knowledge graph")
            return []
            
        # Get indices
        head_idx = self.entity_to_idx[head_id]
        relation_idx = self.relation_to_idx[relation_type]
        
        # Get embeddings
        head_embedding = self.model.entity_embeddings.weight[head_idx].detach()
        relation_embedding = self.model.relation_embeddings.weight[relation_idx].detach()
        
        # Calculate scores for all potential tail entities
        all_tail_embeddings = self.model.entity_embeddings.weight.detach()
        scores = torch.norm(head_embedding + relation_embedding - all_tail_embeddings, p=1, dim=1)
        scores = scores.cpu().numpy()
        
        # Get top-k predictions (lowest scores)
        top_idxs = np.argsort(scores)[:top_k]
        predictions = []
        
        for idx in top_idxs:
            predictions.append({
                'entity': self.idx_to_entity[idx],
                'score': float(scores[idx])
            })
            
        return predictions
        
    def find_events_for_person(self, person_email, top_k=10):
        """
        Find most relevant events for a person using embeddings.
        
        Args:
            person_email: Email address of the person
            top_k: Number of events to return
            
        Returns:
            List of relevant events with scores
        """
        person_id = f"Person_{person_email}"
        
        if person_id not in self.entity_to_idx:
            logger.error(f"Person {person_email} not found in the knowledge graph")
            return []
            
        # Get all event entities
        event_entities = [entity for entity in self.entity_to_idx.keys() if entity.startswith("Event_")]
        
        if not event_entities:
            logger.error("No events found in the knowledge graph")
            return []
            
        # Get person embedding
        person_idx = self.entity_to_idx[person_id]
        person_embedding = self.model.entity_embeddings.weight[person_idx].detach().cpu().numpy()
        
        # Calculate cosine similarity with all events
        event_idxs = [self.entity_to_idx[event] for event in event_entities]
        event_embeddings = self.model.entity_embeddings.weight[event_idxs].detach().cpu().numpy()
        
        similarities = np.dot(event_embeddings, person_embedding) / (
            np.linalg.norm(event_embeddings, axis=1) * np.linalg.norm(person_embedding)
        )
        
        # Sort events by similarity
        sorted_idxs = np.argsort(similarities)[::-1][:top_k]
        
        # Get event details from Neo4j
        events = []
        with self.driver.session() as session:
            for i in sorted_idxs:
                event_entity = event_entities[i]
                event_id = event_entity.split("_", 1)[1]
                
                query = """
                MATCH (e:Event {event_id: $event_id})
                OPTIONAL MATCH (e)-[:LOCATED_AT]->(l:Location)
                RETURN e.event_id as id,
                       e.subject as subject,
                       e.event_type as type,
                       e.event_date as date,
                       e.event_time as time,
                       l.name as location,
                       e.virtual_platform as platform,
                       e.virtual_link as link
                """
                
                result = session.run(query, event_id=event_id)
                record = result.single()
                
                if record:
                    events.append({
                        'id': record['id'],
                        'subject': record['subject'],
                        'type': record['type'],
                        'date': record['date'],
                        'time': record['time'],
                        'location': record['location'],
                        'platform': record['platform'],
                        'link': record['link'],
                        'similarity': float(similarities[i])
                    })
        
        return events 