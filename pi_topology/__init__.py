"""
Π-Topology: Parallel State-Space Mathematics Framework
"""

from .core import PiFlow, Config
from .operations import Operations
from .constraints import Constraint, ConstraintSystem

__version__ = "0.1.0"
__author__ = "Π-Topology Research Group"
__email__ = "research@pi-topology.org"

__all__ = [
    'PiFlow',
    'Config',
    'Operations',
    'Constraint',
    'ConstraintSystem',
]

# Aliases for convenience
pi_merge = Operations.pi_merge
pi_involution = Operations.pi_involution