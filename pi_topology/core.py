"""
Π-Topology Core Module
Main PiFlow class and core functionality
"""

import warnings
from typing import List, Tuple, Optional, Union, Callable, Dict, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found. GPU acceleration disabled.")


class Config:
    """Configuration manager for Π-Topology"""
    
    def __init__(self, device: str = 'auto', dtype: str = 'float32', 
                 chunk_size: int = 1024, verbose: bool = True):
        """
        Initialize configuration
        
        Args:
            device: 'auto', 'cpu', 'cuda', or 'mps'
            dtype: 'float16', 'float32', or 'float64'
            chunk_size: Size for chunked processing
            verbose: Enable verbose output
        """
        self.verbose = verbose
        
        # Device selection
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Data type
        dtype_map = {
            'float16': torch.float16 if TORCH_AVAILABLE else np.float16,
            'float32': torch.float32 if TORCH_AVAILABLE else np.float32,
            'float64': torch.float64 if TORCH_AVAILABLE else np.float64,
        }
        self.dtype = dtype_map.get(dtype, torch.float32 if TORCH_AVAILABLE else np.float32)
        
        # Chunk size optimization
        if self.device == 'cuda' and TORCH_AVAILABLE:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 4 * 1024**3:  # Less than 4GB
                self.chunk_size = 512
            elif gpu_memory < 8 * 1024**3:  # Less than 8GB
                self.chunk_size = 1024
            else:
                self.chunk_size = 2048
        else:
            self.chunk_size = chunk_size
        
        # Performance tuning
        self.use_mixed_precision = self.device in ['cuda', 'mps']
        self.auto_chunking = True
        
        if self.verbose:
            self.print_config()
    
    def print_config(self):
        """Print current configuration"""
        print("Π-Topology Configuration:")
        print(f"  Device: {self.device}")
        if TORCH_AVAILABLE and self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Data type: {self.dtype}")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Mixed precision: {self.use_mixed_precision}")
        print("-" * 50)


class PiFlow:
    """
    Π-Flow: Parallel State-Space Object
    
    A Π-Flow represents an entire state-space as a single mathematical object.
    All operations on Π-Flows are inherently parallel.
    """
    
    def __init__(self, name: str, config: Optional[Config] = None):
        """
        Initialize a Π-Flow
        
        Args:
            name: Identifier for the flow
            config: Configuration object (uses default if None)
        """
        self.name = name
        self.config = config or Config()
        self._values = None
        self._contexts = None
        self._func = None
        self._solutions = None
        self._mask = None
        
        # Statistics
        self.stats = {
            'creation_time': 0,
            'merge_time': 0,
            'involution_time': 0,
            'memory_used': 0,
            'contexts_processed': 0,
        }
    
    def create_from_domain(self, domains: List[np.ndarray], 
                          func: Callable) -> 'PiFlow':
        """
        Create Π-Flow from domain and function
        
        Args:
            domains: List of domain arrays for each dimension
            func: Function that maps contexts to values
        
        Returns:
            Self for method chaining
        """
        import time
        start_time = time.time()
        
        # Store function for later use
        self._func = func
        
        # Create grid of all contexts
        if TORCH_AVAILABLE and self.config.device != 'cpu':
            # Use PyTorch for GPU acceleration
            tensor_domains = [torch.tensor(d, device=self.config.device, 
                                         dtype=self.config.dtype) 
                            for d in domains]
            
            # Create meshgrid for all dimensions
            mesh = torch.meshgrid(*tensor_domains, indexing='ij')
            contexts = torch.stack(mesh, dim=-1)
            
            # Apply function
            with torch.no_grad():
                if self.config.use_mixed_precision and self.config.device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        self._values = func(*mesh)
                else:
                    self._values = func(*mesh)
            
            self._contexts = contexts
            
        else:
            # Use NumPy for CPU
            mesh = np.meshgrid(*domains, indexing='ij')
            contexts = np.stack(mesh, axis=-1)
            
            # Apply function
            self._values = func(*mesh)
            self._contexts = contexts
        
        # Update statistics
        self.stats['creation_time'] = time.time() - start_time
        self.stats['contexts_processed'] = np.prod([len(d) for d in domains])
        
        if self.config.verbose:
            total_contexts = np.prod([len(d) for d in domains])
            print(f"Created Π-Flow '{self.name}' with {total_contexts:,} contexts")
        
        return self
    
    def create_from_2d_domain(self, x_domain: np.ndarray, 
                             y_domain: np.ndarray, 
                             func: Callable) -> 'PiFlow':
        """
        Convenience method for 2D domains
        
        Args:
            x_domain: Domain for first dimension
            y_domain: Domain for second dimension
            func: Function f(x, y) -> value
        
        Returns:
            Self for method chaining
        """
        return self.create_from_domain([x_domain, y_domain], func)
    
    def get_values(self):
        """Get flow values"""
        if self._values is None:
            raise ValueError("Π-Flow has no values. Call create_from_domain first.")
        return self._values
    
    def get_contexts(self):
        """Get flow contexts"""
        if self._contexts is None:
            raise ValueError("Π-Flow has no contexts.")
        return self._contexts
    
    def get_solutions(self) -> np.ndarray:
        """
        Get solutions after involution
        
        Returns:
            Array of solutions (contexts that satisfy constraints)
        """
        if self._solutions is None:
            return np.array([])
        
        if TORCH_AVAILABLE and isinstance(self._solutions, torch.Tensor):
            return self._solutions.cpu().numpy()
        else:
            return self._solutions
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get flow statistics
        
        Returns:
            Dictionary with performance statistics
        """
        stats = self.stats.copy()
        if hasattr(self, '_solutions'):
            stats['solutions_found'] = len(self._solutions) if self._solutions is not None else 0
        else:
            stats['solutions_found'] = 0
        
        if TORCH_AVAILABLE and self.config.device == 'cuda':
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2
        
        return stats
    
    def __repr__(self) -> str:
        """String representation"""
        if self._values is None:
            return f"PiFlow('{self.name}') [uninitialized]"
        
        shape = self._values.shape
        return f"PiFlow('{self.name}') [shape={shape}, device={self.config.device}]"