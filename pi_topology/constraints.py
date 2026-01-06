"""
Constraint system for Π-Involution
"""

from typing import List, Dict, Any, Union, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Constraint:
    """Base constraint class"""
    
    def __init__(self, constraint_type: str, target: Union[float, Tuple[float, float]], 
                 tolerance: float = 0.0, dimension: int = 0):
        """
        Initialize constraint
        
        Args:
            constraint_type: 'equals', 'less', 'greater', 'range'
            target: Target value or (min, max) for range
            tolerance: Allowable deviation
            dimension: Which dimension to apply to (for multi-dimensional values)
        """
        self.type = constraint_type
        self.target = target
        self.tolerance = tolerance
        self.dimension = dimension
        
        # Validate
        if constraint_type == 'range':
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                raise ValueError("Range constraint requires (min, max) target")
    
    def apply(self, values, mask=None):
        """
        Apply constraint to values
        
        Args:
            values: Array of values
            mask: Optional initial mask
        
        Returns:
            Updated mask
        """
        if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
            return self._apply_torch(values, mask)
        else:
            return self._apply_numpy(values, mask)
    
    def _apply_numpy(self, values, mask=None):
        """Apply constraint using NumPy"""
        if values.ndim > 2 and self.dimension < values.shape[-1]:
            # Multi-dimensional values
            target_values = values[..., self.dimension]
        else:
            target_values = values
        
        if mask is None:
            mask = np.ones_like(target_values, dtype=bool)
        
        if self.type == 'equals':
            condition = np.abs(target_values - self.target) <= self.tolerance
        elif self.type == 'less':
            condition = target_values <= (self.target + self.tolerance)
        elif self.type == 'greater':
            condition = target_values >= (self.target - self.tolerance)
        elif self.type == 'range':
            min_val, max_val = self.target
            condition = (target_values >= min_val - self.tolerance) & \
                       (target_values <= max_val + self.tolerance)
        else:
            raise ValueError(f"Unknown constraint type: {self.type}")
        
        return mask & condition
    
    def _apply_torch(self, values, mask=None):
        """Apply constraint using PyTorch"""
        if values.ndim > 2 and self.dimension < values.shape[-1]:
            # Multi-dimensional values
            target_values = values[..., self.dimension]
        else:
            target_values = values
        
        if mask is None:
            mask = torch.ones_like(target_values, dtype=torch.bool, device=values.device)
        
        if self.type == 'equals':
            condition = torch.abs(target_values - self.target) <= self.tolerance
        elif self.type == 'less':
            condition = target_values <= (self.target + self.tolerance)
        elif self.type == 'greater':
            condition = target_values >= (self.target - self.tolerance)
        elif self.type == 'range':
            min_val, max_val = self.target
            condition = (target_values >= min_val - self.tolerance) & \
                       (target_values <= max_val + self.tolerance)
        else:
            raise ValueError(f"Unknown constraint type: {self.type}")
        
        return mask & condition
    
    def __repr__(self):
        """String representation"""
        return f"Constraint(type={self.type}, target={self.target}, tol={self.tolerance})"


class ConstraintSystem:
    """System of multiple constraints"""
    
    def __init__(self, constraints: List[Dict[str, Any]]):
        """
        Initialize constraint system
        
        Args:
            constraints: List of constraint dictionaries
        """
        self.constraints = []
        for constraint_dict in constraints:
            constraint = Constraint(
                constraint_type=constraint_dict['type'],
                target=constraint_dict['target'],
                tolerance=constraint_dict.get('tolerance', 0.0),
                dimension=constraint_dict.get('dimension', 0)
            )
            self.constraints.append(constraint)
    
    def apply(self, values, mask=None):
        """
        Apply all constraints
        
        Args:
            values: Values to constrain
            mask: Initial mask
        
        Returns:
            Final mask after all constraints
        """
        current_mask = mask
        
        for i, constraint in enumerate(self.constraints):
            # Update dimension if not specified
            if constraint.dimension == 0 and values.ndim > 2:
                constraint.dimension = i if i < values.shape[-1] else 0
            
            current_mask = constraint.apply(values, current_mask)
        
        return current_mask
    
    def __len__(self):
        """Number of constraints"""
        return len(self.constraints)
    
    def __repr__(self):
        """String representation"""
        return f"ConstraintSystem({len(self)} constraints)"