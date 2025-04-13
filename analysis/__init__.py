"""
Analysis module for querying and analyzing email graph data.
"""

from .queries import EmailGraphQueries
from .metrics import compute_communication_metrics

__all__ = [
    'EmailGraphQueries',
    'compute_communication_metrics'
] 