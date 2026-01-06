"""
Π-Topology Operations
Π-Merge and Π-Involution implementations
"""

import warnings
from typing import List, Dict, Any, Union, Optional
import numpy as np
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Operations:
    """Π-Topology operations container"""
    
    @staticmethod
    def pi_merge(flow1: 'PiFlow', flow2: 'PiFlow', 
                 operation: str = 'concat') -> 'PiFlow':
        """
        Π-Merge operation: Combine two Π-Flows
        
        Args:
            flow1: First Π-Flow
            flow2: Second Π-Flow
            operation: Merge operation ('concat', 'add', 'multiply', 'max', 'min')
        
        Returns:
            New merged Π-Flow
        """
        from .core import PiFlow, Config
        
        start_time = time.time()
        
        # Verify compatibility
        if flow1._values.shape != flow2._values.shape:
            raise ValueError("Π-Flows must have same shape for merging")
        
        # Create new flow
        merged = PiFlow(f"({flow1.name}⨁{flow2.name})", flow1.config)
        
        # Perform merge operation
        if TORCH_AVAILABLE and isinstance(flow1._values, torch.Tensor):
            # Tensor operations
            if operation == 'concat':
                merged._values = torch.stack([flow1._values, flow2._values], dim=-1)
            elif operation == 'add':
                merged._values = flow1._values + flow2._values
            elif operation == 'multiply':
                merged._values = flow1._values * flow2._values
            elif operation == 'max':
                merged._values = torch.maximum(flow1._values, flow2._values)
            elif operation == 'min':
                merged._values = torch.minimum(flow1._values, flow2._values)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Copy contexts
            merged._contexts = flow1._contexts.clone()
            
        else:
            # NumPy operations
            if operation == 'concat':
                merged._values = np.stack([flow1._values, flow2._values], axis=-1)
            elif operation == 'add':
                merged._values = flow1._values + flow2._values
            elif operation == 'multiply':
                merged._values = flow1._values * flow2._values
            elif operation == 'max':
                merged._values = np.maximum(flow1._values, flow2._values)
            elif operation == 'min':
                merged._values = np.minimum(flow1._values, flow2._values)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Copy contexts
            merged._contexts = flow1._contexts.copy()
        
        # Update statistics
        merged.stats['merge_time'] = time.time() - start_time
        merged.stats['contexts_processed'] = flow1.stats['contexts_processed']
        
        if flow1.config.verbose:
            print(f"Π-Merge: {flow1.name} ⨁ {flow2.name} → {merged.name}")
        
        return merged
    
    @staticmethod
    def pi_involution(flow: 'PiFlow', 
                     constraints: List[Dict[str, Any]],
                     chunk_size: Optional[int] = None) -> 'PiFlow':
        """
        Π-Involution: Collapse state-space to satisfy constraints
        
        Args:
            flow: Π-Flow to process
            constraints: List of constraint dictionaries
            chunk_size: Override default chunk size
        
        Returns:
            Modified Π-Flow with solutions
        """
        from .constraints import ConstraintSystem
        
        start_time = time.time()
        
        # Initialize constraint system
        constraint_system = ConstraintSystem(constraints)
        
        # Get chunk size
        if chunk_size is None:
            chunk_size = flow.config.chunk_size
        
        # Check if we should use chunked processing
        total_contexts = np.prod(flow._values.shape[:-1]) if flow._values.ndim > 2 else flow._values.size
        use_chunking = flow.config.auto_chunking and total_contexts > chunk_size**2
        
        if use_chunking:
            flow = Operations._involution_chunked(flow, constraint_system, chunk_size)
        else:
            flow = Operations._involution_single(flow, constraint_system)
        
        # Update statistics
        flow.stats['involution_time'] = time.time() - start_time
        
        if flow.config.verbose:
            sol_count = len(flow._solutions) if flow._solutions is not None else 0
            print(f"Π-Involution: {sol_count} solutions found in {flow.stats['involution_time']:.3f}s")
        
        return flow
    
    @staticmethod
    def _involution_single(flow: 'PiFlow', 
                          constraint_system: 'ConstraintSystem') -> 'PiFlow':
        """Involution for entire space (no chunking)"""
        values = flow.get_values()
        contexts = flow.get_contexts()
        
        # Create initial mask
        if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
            mask = torch.ones(values.shape[:-1] if values.ndim > 2 else values.shape, 
                            dtype=torch.bool, device=values.device)
        else:
            mask = np.ones(values.shape[:-1] if values.ndim > 2 else values.shape, 
                          dtype=bool)
        
        # Apply all constraints
        mask = constraint_system.apply(values, mask)
        
        # Extract solutions
        if TORCH_AVAILABLE and isinstance(contexts, torch.Tensor):
            flow._solutions = contexts[mask].reshape(-1, contexts.shape[-1])
            flow._mask = mask
        else:
            flow._solutions = contexts[mask].reshape(-1, contexts.shape[-1])
            flow._mask = mask
        
        return flow
    
    @staticmethod
    def _involution_chunked(flow: 'PiFlow', 
                           constraint_system: 'ConstraintSystem',
                           chunk_size: int) -> 'PiFlow':
        """Chunked involution for large spaces"""
        import torch
        
        contexts = flow.get_contexts()
        values = flow.get_values()
        
        # Determine shape
        if values.ndim == 2:  # 2D values
            n_x, n_y = values.shape
            n_dims = 2
        else:  # ND values
            n_x, n_y = values.shape[:2]
            n_dims = values.ndim - 1
        
        if flow.config.verbose:
            print(f"Chunked processing: {n_x//chunk_size + 1}×{n_y//chunk_size + 1} chunks")
        
        # Collect solutions from all chunks
        all_solutions = []
        
        for i in range(0, n_x, chunk_size):
            for j in range(0, n_y, chunk_size):
                # Get chunk slices
                i_end = min(i + chunk_size, n_x)
                j_end = min(j + chunk_size, n_y)
                
                # Extract chunk
                if n_dims == 2:
                    chunk_values = values[i:i_end, j:j_end]
                    chunk_contexts = contexts[i:i_end, j:j_end]
                else:
                    chunk_values = values[i:i_end, j:j_end, :]
                    chunk_contexts = contexts[i:i_end, j:j_end, :]
                
                # Apply constraints to chunk
                chunk_mask = constraint_system.apply(chunk_values)
                
                # Extract solutions from chunk
                if isinstance(chunk_mask, torch.Tensor):
                    if chunk_mask.any():
                        chunk_solutions = chunk_contexts[chunk_mask]
                        all_solutions.append(chunk_solutions.cpu())
                else:
                    if np.any(chunk_mask):
                        chunk_solutions = chunk_contexts[chunk_mask]
                        all_solutions.append(chunk_solutions)
        
        # Combine all solutions
        if all_solutions:
            if isinstance(all_solutions[0], torch.Tensor):
                flow._solutions = torch.cat(all_solutions, dim=0)
            else:
                flow._solutions = np.concatenate(all_solutions, axis=0)
        else:
            flow._solutions = None
        
        flow._mask = None  # No global mask in chunked mode
        
        return flow 