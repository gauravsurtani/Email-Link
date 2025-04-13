"""
Graph database module for building and managing the email graph.
"""

from .loader import EmailGraphLoader
from .schema import create_constraints

__all__ = [
    'EmailGraphLoader',
    'create_constraints'
] 