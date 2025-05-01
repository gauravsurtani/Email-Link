#!/usr/bin/env python3
"""
Evaluation script for TransE knowledge graph embeddings in the email event extraction system.
This script provides comprehensive testing of embedding quality through:
1. Triple classification tests
2. Link prediction tests
3. Visual analysis of embedding performance

Usage:
    python evaluate_embeddings.py [--embeddings-dir DIR] [--test-ratio RATIO] [--k K]
"""

import os
import numpy as np
import argparse
import logging
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)

# Import from local modules
from embeddings.graph_embeddings import GraphEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(graph_embeddings, test_ratio=0.2, negative_ratio=1):
    """
    Create test data by splitting the triples from the knowledge graph.
    
    Args:
        graph_embeddings: GraphEmbeddings instance with loaded embeddings
        test_ratio: Proportion of triples to use for testing
        negative_ratio: Number of negative samples per positive sample
        
    Returns:
        train_triples: List of triples for training
        test_triples: List of positive test triples
        negative_triples: List of negative (corrupted) triples for testing
    """
    logger.info("Extracting triples from Neo4j for testing")
    
    # Extract all triples from graph
    all_triples, entity_to_idx, relation_to_idx = graph_embeddings.extract_triples()
    
    # Shuffle triples
    np.random.shuffle(all_triples)
    
    # Split into train and test
    test_size = int(len(all_triples) * test_ratio)
    test_triples = all_triples[:test_size]
    train_triples = all_triples[test_size:]
    
    logger.info(f"Created test set with {len(test_triples)} positive triples and {len(train_triples)} training triples")
    
    # Create negative triples by corrupting either head or tail
    negative_triples = []
    entities = list(entity_to_idx.keys())
    
    for h, r, t in test_triples:
        for _ in range(negative_ratio):
            if np.random.random() < 0.5:
                # Corrupt head
                false_h = h
                while false_h == h:
                    false_h = np.random.choice(entities)
                negative_triples.append((false_h, r, t))
            else:
                # Corrupt tail
                false_t = t
                while false_t == t:
                    false_t = np.random.choice(entities)
                negative_triples.append((h, r, false_t))
    
    logger.info(f"Created {len(negative_triples)} negative triples for testing")
    
    return train_triples, test_triples, negative_triples

def score_triple(model, h, r, t, entity_to_idx, relation_to_idx):
    """
    Calculate the TransE score for a triple.
    
    Args:
        model: TransE model
        h, r, t: Head, relation, tail entities
        entity_to_idx: Mapping from entity names to indices
        relation_to_idx: Mapping from relation names to indices
        
    Returns:
        score: TransE score (lower is better)
    """
    # Get indices
    h_idx = entity_to_idx.get(h)
    r_idx = relation_to_idx.get(r)
    t_idx = entity_to_idx.get(t)
    
    # Skip if any entity or relation is not found
    if h_idx is None or r_idx is None or t_idx is None:
        return float('inf')
    
    # Convert to tensors
    h_tensor = torch.tensor([h_idx], dtype=torch.long)
    r_tensor = torch.tensor([r_idx], dtype=torch.long)
    t_tensor = torch.tensor([t_idx], dtype=torch.long)
    
    # Get embeddings
    h_emb = model.entity_embeddings(h_tensor)
    r_emb = model.relation_embeddings(r_tensor)
    t_emb = model.entity_embeddings(t_tensor)
    
    # Calculate score: ||h + r - t||
    score = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1).item()
    
    return score

def evaluate_triple_classification(model, positive_triples, negative_triples, entity_to_idx, relation_to_idx, threshold=None):
    """
    Evaluate triple classification performance.
    
    Args:
        model: TransE model
        positive_triples: List of true triples
        negative_triples: List of false triples
        entity_to_idx: Mapping from entity names to indices
        relation_to_idx: Mapping from relation names to indices
        threshold: Classification threshold (if None, find optimal threshold)
        
    Returns:
        tuple: (metrics_dict, positive_scores, negative_scores)
    """
    logger.info("Evaluating triple classification")
    
    # Score all triples
    pos_scores = []
    valid_pos_triples = 0
    
    for h, r, t in positive_triples:
        score = score_triple(model, h, r, t, entity_to_idx, relation_to_idx)
        if score != float('inf'):
            pos_scores.append(score)
            valid_pos_triples += 1
    
    neg_scores = []
    valid_neg_triples = 0
    
    for h, r, t in negative_triples:
        score = score_triple(model, h, r, t, entity_to_idx, relation_to_idx)
        if score != float('inf'):
            neg_scores.append(score)
            valid_neg_triples += 1
    
    logger.info(f"Scored {valid_pos_triples}/{len(positive_triples)} positive and {valid_neg_triples}/{len(negative_triples)} negative triples")
    
    # Find optimal threshold if not provided
    if threshold is None:
        all_scores = pos_scores + neg_scores
        thresholds = np.percentile(all_scores, np.arange(0, 100, 5))
        best_f1 = 0
        best_threshold = np.mean(all_scores)
        
        for t in thresholds:
            y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
            y_pred = [1 if score < t else 0 for score in pos_scores + neg_scores]
            
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        threshold = best_threshold
    
    logger.info(f"Using classification threshold: {threshold}")
    
    # Make predictions
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_pred = [1 if score < threshold else 0 for score in pos_scores + neg_scores]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'pos_mean_score': np.mean(pos_scores) if pos_scores else float('nan'),
        'neg_mean_score': np.mean(neg_scores) if neg_scores else float('nan'),
        'valid_positive_triples': valid_pos_triples,
        'valid_negative_triples': valid_neg_triples
    }
    
    return metrics, pos_scores, neg_scores

def evaluate_link_prediction(model, test_triples, entity_to_idx, relation_to_idx, k=10, max_entities=500):
    """
    Evaluate link prediction performance.
    
    Args:
        model: TransE model
        test_triples: List of test triples
        entity_to_idx: Mapping from entity names to indices
        relation_to_idx: Mapping from relation names to indices
        k: k for Hits@k metric
        max_entities: Maximum number of entities to test against for efficiency
        
    Returns:
        tuple: (metrics_dict, ranks)
    """
    logger.info("Evaluating link prediction")
    
    entities = list(entity_to_idx.keys())
    ranks = []
    hits_at_k = 0
    
    # Use a subset of entities for efficiency
    if len(entities) > max_entities:
        test_entities = np.random.choice(entities, max_entities, replace=False)
    else:
        test_entities = entities
    
    # Test on a sample if there are too many triples
    sample_size = min(len(test_triples), 100)  # Adjust based on runtime constraints
    test_sample = np.random.choice(len(test_triples), sample_size, replace=False)
    sampled_triples = [test_triples[i] for i in test_sample]
    
    logger.info(f"Testing link prediction on {sample_size} triples against up to {len(test_entities)} entities")
    
    for idx, (h, r, t) in enumerate(sampled_triples):
        if idx > 0 and idx % 10 == 0:
            logger.info(f"Processed {idx}/{len(sampled_triples)} test triples")
        
        # Skip if any entity or relation is not found
        if h not in entity_to_idx or r not in relation_to_idx or t not in entity_to_idx:
            continue
            
        # Score for true triple
        true_score = score_triple(model, h, r, t, entity_to_idx, relation_to_idx)
        
        # Score for corrupted triples (corrupting tail)
        scores = []
        for e in test_entities:
            if e != t:  # Exclude the correct tail
                score = score_triple(model, h, r, e, entity_to_idx, relation_to_idx)
                scores.append(score)
        
        # Calculate rank (how many corrupted triples score better than the true triple)
        # Lower score is better in TransE
        rank = 1 + sum(1 for score in scores if score < true_score)
        ranks.append(rank)
        
        if rank <= k:
            hits_at_k += 1
    
    valid_triples = len(ranks)
    if valid_triples == 0:
        logger.warning("No valid triples for link prediction evaluation")
        return {
            'mean_rank': float('nan'),
            'mean_reciprocal_rank': float('nan'),
            f'hits@{k}': float('nan')
        }, []
    
    mean_rank = sum(ranks) / valid_triples
    mean_reciprocal_rank = sum(1/r for r in ranks) / valid_triples
    hits_at_k_ratio = hits_at_k / valid_triples
    
    metrics = {
        'mean_rank': mean_rank,
        'mean_reciprocal_rank': mean_reciprocal_rank,
        f'hits@{k}': hits_at_k_ratio,
        'valid_triples': valid_triples
    }
    
    return metrics, ranks

def create_visualizations(classification_results, link_prediction_results, pos_scores, neg_scores, ranks, results_dir):
    """
    Create visualization charts for the evaluation results.
    
    Args:
        classification_results: Results from triple classification
        link_prediction_results: Results from link prediction
        pos_scores: Scores for positive triples
        neg_scores: Scores for negative triples
        ranks: Ranks from link prediction
        results_dir: Directory to save visualizations
    """
    plt.style.use('ggplot')
    
    # Set up the visualization directory
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Skip visualizations if there's insufficient data
    if not pos_scores or not neg_scores:
        logger.warning("Insufficient data for classification visualizations")
    else:
        # Figure 1: Classification Metrics Bar Chart
        plt.figure(figsize=(10, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        values = [classification_results[m] for m in metrics]
        
        ax = sns.barplot(x=metrics, y=values, palette='viridis')
        plt.title('Triple Classification Performance', fontsize=15)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'classification_metrics.png', dpi=300)
        plt.close()
        
        # Figure 2: Score Distribution Histogram
        plt.figure(figsize=(12, 6))
        
        sns.histplot(pos_scores, color='green', alpha=0.6, label='Positive Triples', kde=True)
        sns.histplot(neg_scores, color='red', alpha=0.6, label='Negative Triples', kde=True)
        
        plt.axvline(x=classification_results['threshold'], color='blue', linestyle='--', 
                    label=f'Threshold: {classification_results["threshold"]:.3f}')
        
        plt.title('Score Distribution for Positive and Negative Triples', fontsize=15)
        plt.xlabel('TransE Score (Lower is Better)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'score_distribution.png', dpi=300)
        plt.close()
        
        # Figure 3: ROC Curve
        y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
        y_score = [-s for s in pos_scores + neg_scores]  # Negate scores since lower is better in TransE
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=15)
        plt.legend(loc="lower right", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'roc_curve.png', dpi=300)
        plt.close()
        
        # Figure 6: Confusion Matrix
        threshold = classification_results['threshold']
        y_pred = [1 if score < threshold else 0 for score in pos_scores + neg_scores]
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        
        plt.title('Confusion Matrix', fontsize=15)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
    
    # Skip link prediction visualizations if there's insufficient data
    if not ranks:
        logger.warning("Insufficient data for link prediction visualizations")
    else:
        # Figure 4: Link Prediction Metrics
        plt.figure(figsize=(10, 6))
        metrics = ['mean_reciprocal_rank', f'hits@{list(link_prediction_results.keys())[-2].split("@")[1]}']
        values = [link_prediction_results[m] for m in metrics]
        labels = ['Mean Reciprocal Rank', f'Hits@{list(link_prediction_results.keys())[-2].split("@")[1]}']
        
        ax = sns.barplot(x=labels, y=values, palette='viridis')
        plt.title('Link Prediction Performance', fontsize=15)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'link_prediction_metrics.png', dpi=300)
        plt.close()
        
        # Figure 5: Rank Distribution Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(ranks, bins=min(30, len(set(ranks))), kde=True)
        plt.axvline(x=link_prediction_results['mean_rank'], color='red', linestyle='--', 
                    label=f'Mean Rank: {link_prediction_results["mean_rank"]:.1f}')
        
        plt.title('Rank Distribution in Link Prediction', fontsize=15)
        plt.xlabel('Rank (Lower is Better)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=10)
        
        # Log scale for x-axis if ranks are widely distributed
        if max(ranks) > 100:
            plt.xscale('log')
            plt.xlim(0.9, max(ranks) * 1.1)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'rank_distribution.png', dpi=300)
        plt.close()
    
    logger.info(f"Visualizations saved to {viz_dir}")

def generate_report(classification_results, link_prediction_results, output_dir):
    """
    Generate a text report summarizing the evaluation results.
    
    Args:
        classification_results: Results from triple classification
        link_prediction_results: Results from link prediction
        output_dir: Directory to save the report
    """
    report = "# TransE Embedding Evaluation Report\n\n"
    
    # Classification results
    report += "## Triple Classification Performance\n\n"
    
    if classification_results['valid_positive_triples'] > 0 and classification_results['valid_negative_triples'] > 0:
        report += f"- **Accuracy**: {classification_results['accuracy']:.4f}\n"
        report += f"- **Precision**: {classification_results['precision']:.4f}\n"
        report += f"- **Recall**: {classification_results['recall']:.4f}\n"
        report += f"- **F1 Score**: {classification_results['f1']:.4f}\n"
        report += f"- **Threshold**: {classification_results['threshold']:.4f}\n"
        report += f"- **Mean Score (Positive Triples)**: {classification_results['pos_mean_score']:.4f}\n"
        report += f"- **Mean Score (Negative Triples)**: {classification_results['neg_mean_score']:.4f}\n"
        report += f"\nEvaluated on {classification_results['valid_positive_triples']} positive and {classification_results['valid_negative_triples']} negative triples.\n"
    else:
        report += "*Insufficient data for triple classification evaluation.*\n"
    
    # Link prediction results
    report += "\n## Link Prediction Performance\n\n"
    
    if link_prediction_results['valid_triples'] > 0:
        report += f"- **Mean Rank**: {link_prediction_results['mean_rank']:.2f}\n"
        report += f"- **Mean Reciprocal Rank**: {link_prediction_results['mean_reciprocal_rank']:.4f}\n"
        
        # Find Hits@k metrics
        hits_metrics = [key for key in link_prediction_results.keys() if key.startswith('hits@')]
        for metric in hits_metrics:
            k = metric.split('@')[1]
            report += f"- **Hits@{k}**: {link_prediction_results[metric]:.4f}\n"
            
        report += f"\nEvaluated on {link_prediction_results['valid_triples']} test triples.\n"
    else:
        report += "*Insufficient data for link prediction evaluation.*\n"
    
    # Interpretation
    report += "\n## Interpretation\n\n"
    
    if classification_results['valid_positive_triples'] > 0 and classification_results['valid_negative_triples'] > 0:
        f1 = classification_results['f1']
        if f1 > 0.8:
            report += "- The model shows **excellent** performance in distinguishing true from false relationships.\n"
        elif f1 > 0.6:
            report += "- The model shows **good** performance in distinguishing true from false relationships.\n"
        elif f1 > 0.4:
            report += "- The model shows **moderate** performance in distinguishing true from false relationships.\n"
        else:
            report += "- The model shows **poor** performance in distinguishing true from false relationships.\n"
    
    if link_prediction_results['valid_triples'] > 0:
        mrr = link_prediction_results['mean_reciprocal_rank']
        if mrr > 0.5:
            report += "- The model is **highly effective** at predicting missing links in the knowledge graph.\n"
        elif mrr > 0.3:
            report += "- The model is **moderately effective** at predicting missing links in the knowledge graph.\n"
        elif mrr > 0.1:
            report += "- The model has **limited effectiveness** at predicting missing links in the knowledge graph.\n"
        else:
            report += "- The model has **very poor** link prediction capabilities.\n"
    
    # Recommendations
    report += "\n## Recommendations\n\n"
    
    if classification_results['valid_positive_triples'] < 100 or link_prediction_results['valid_triples'] < 50:
        report += "- **Increase data volume**: The evaluation was performed on a small dataset. Adding more data would provide more reliable results.\n"
    
    if classification_results['valid_positive_triples'] > 0 and classification_results['f1'] < 0.6:
        report += "- **Improve model training**: Consider increasing the embedding dimension or the number of training epochs.\n"
        report += "- **Review relationship types**: Some relationship types may not be well represented in the training data.\n"
    
    if link_prediction_results['valid_triples'] > 0 and link_prediction_results['mean_reciprocal_rank'] < 0.3:
        report += "- **Enhance graph connectivity**: Add more relationships between entities to improve embedding quality.\n"
        report += "- **Try alternative models**: Consider TransH or RotatE for better handling of complex relationships.\n"
    
    # Save report
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to {report_path}")
    
    return report_path

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate TransE embeddings for email event extraction')
    
    parser.add_argument('--embeddings-dir', type=str, default='./output/embeddings',
                        help='Directory containing embedding files')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Proportion of triples to use for testing')
    parser.add_argument('--k', type=int, default=10,
                        help='k for Hits@k metric in link prediction')
    parser.add_argument('--max-entities', type=int, default=500,
                        help='Maximum number of entities to test against in link prediction')
    
    args = parser.parse_args()
    
    # Load environment configuration
    load_dotenv()
    
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD','12345678')
    
    if not neo4j_password:
        logger.error("Neo4j password not provided in .env file")
        print("Error: Neo4j password not provided in .env file")
        return 1
    
    # Initialize graph embeddings
    graph_embeddings = GraphEmbeddings(
        neo4j_uri, 
        neo4j_user, 
        neo4j_password,
        output_dir=args.embeddings_dir
    )
    
    # Create output directory
    results_dir = Path(args.embeddings_dir) / "evaluation"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load existing embeddings
    print("Loading embeddings...")
    if not graph_embeddings.load_embeddings():
        logger.error("Failed to load embeddings")
        print("Error: Failed to load embeddings. Run event_pipeline.py first to generate embeddings.")
        return 1
    
    print("Creating test data...")
    # Create test data
    train_triples, test_triples, negative_triples = create_test_data(
        graph_embeddings, 
        test_ratio=args.test_ratio
    )
    
    # Get model, entity and relation mappings
    model = graph_embeddings.model
    entity_to_idx = graph_embeddings.entity_to_idx
    relation_to_idx = graph_embeddings.relation_to_idx
    
    print("Evaluating triple classification...")
    # Evaluate triple classification
    classification_results, pos_scores, neg_scores = evaluate_triple_classification(
        model, 
        test_triples, 
        negative_triples,
        entity_to_idx,
        relation_to_idx
    )
    
    print("Evaluating link prediction...")
    # Evaluate link prediction
    link_prediction_results, ranks = evaluate_link_prediction(
        model, 
        test_triples, 
        entity_to_idx,
        relation_to_idx,
        k=args.k,
        max_entities=args.max_entities
    )
    
    # Print results
    print("\nTriple Classification Results:")
    print("-" * 50)
    for metric, value in classification_results.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\nLink Prediction Results:")
    print("-" * 50)
    for metric, value in link_prediction_results.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(
        classification_results,
        link_prediction_results,
        pos_scores,
        neg_scores,
        ranks,
        results_dir
    )
    
    # Generate report
    print("Generating evaluation report...")
    report_path = generate_report(
        classification_results,
        link_prediction_results,
        results_dir
    )
    
    # Save results to CSV
    pd.DataFrame([classification_results]).to_csv(
        results_dir / "classification_results.csv", index=False
    )
    
    pd.DataFrame([link_prediction_results]).to_csv(
        results_dir / "link_prediction_results.csv", index=False
    )
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to {results_dir}")
    print(f"Report available at {report_path}")
    
    # Close Neo4j connection
    graph_embeddings.close()
    
    return 0

if __name__ == "__main__":
    exit(main()) 