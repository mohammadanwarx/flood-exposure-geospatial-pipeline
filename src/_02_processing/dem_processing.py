"""
Digital Elevation Model (DEM) Processing Module

This module handles DEM data loading, preprocessing, and manipulation
for flood exposure analysis.
"""

import numpy as np
from typing import Tuple, Optional

# Optional rasterio import
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def load_dem(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load a DEM raster file.
    
    Parameters
    ----------
    filepath : str
        Path to the DEM raster file
    
    Returns
    -------
    Tuple[np.ndarray, dict]
        DEM array and metadata dictionary
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio required. Install: pip install rasterio")
    with rasterio.open(filepath) as src:
        dem = src.read(1)
        metadata = src.meta.copy()
    return dem, metadata


def fill_depressions(dem: np.ndarray) -> np.ndarray:
    """
    Fill depressions in DEM for hydrological analysis.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM array
        
    Returns
    -------
    np.ndarray
        Depression-filled DEM
    """
    # Simple iterative pit filling: raise any cell lower than all 8 neighbors
    # to the minimum of its neighbors. This is a basic approach and may be
    # slow for large rasters compared to specialized hydrology libraries.
    filled = dem.astype(float, copy=True)
    valid = np.isfinite(filled)

    max_iter = 1000
    for _ in range(max_iter):
        padded = np.pad(filled, 1, mode="edge")
        neighbors = [
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2],                     padded[1:-1, 2:],
            padded[2:, :-2],  padded[2:, 1:-1],  padded[2:, 2:]
        ]
        neighbor_min = np.minimum.reduce(neighbors)

        pits = valid & (filled < neighbor_min)
        if not np.any(pits):
            break

        filled[pits] = neighbor_min[pits]

    return filled


def calculate_slope(dem: np.ndarray, cellsize: float) -> np.ndarray:
    """
    Calculate slope from DEM.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM array
    cellsize : float
        Cell size in meters
        
    Returns
    -------
    np.ndarray
        Slope in degrees
    """
    # Calculate gradients
    dy, dx = np.gradient(dem, cellsize)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope


def resample_dem(dem: np.ndarray, metadata: dict, target_resolution: float) -> Tuple[np.ndarray, dict]:
    """
    Resample DEM to target resolution.
    
    Parameters
    ----------
    dem : np.ndarray
        Input DEM array
    metadata : dict
        Raster metadata
    target_resolution : float
        Target resolution in meters
        
    Returns
    -------
    Tuple[np.ndarray, dict]
        Resampled DEM and updated metadata
    """
    # TODO: Implement resampling
    return dem, metadata
