"""
Flood Exposure Assessment in the Nile Basin Region
===================================================

Flood exposure assessment using DEM-derived flow accumulation and drainage 
network analysis. A comprehensive pipeline including DEM processing, hydrology 
analysis, tensor operations, and data cubes.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main modules for easier access
from src import dem_processing
from src import hydrology
from src import exposure
from src import tensors
from src import cubes
from src import visualization

__all__ = [
    "dem_processing",
    "hydrology",
    "exposure",
    "tensors",
    "cubes",
    "visualization",
]
