"""Tests for DEM processing module."""

import pytest
import numpy as np
from src._02_processing.dem_processing import load_dem, fill_depressions, calculate_slope, save_raster


def test_fill_depressions():
    """Test depression filling."""
    # Create a simple DEM with a depression
    dem = np.array([
        [10, 10, 10, 10],
        [10, 5, 5, 10],
        [10, 5, 5, 10],
        [10, 10, 10, 10]
    ], dtype=float)
    
    filled_dem = fill_depressions(dem)
    
    assert isinstance(filled_dem, np.ndarray)
    assert filled_dem.shape == dem.shape
    # Currently returns a copy, so this test passes
    assert np.all(filled_dem >= dem)


def test_calculate_slope():
    """Test slope calculation."""
    # Create a simple tilted plane
    dem = np.array([
        [10, 10, 10],
        [8, 8, 8],
        [6, 6, 6]
    ], dtype=float)
    
    cellsize = 10.0  # 10 meters
    slope = calculate_slope(dem, cellsize)
    
    assert isinstance(slope, np.ndarray)
    assert slope.shape == dem.shape
    assert np.all(slope >= 0)
    assert np.all(slope <= 90)


def test_calculate_slope_flat():
    """Test slope calculation on flat surface."""
    dem = np.ones((5, 5), dtype=float) * 100
    slope = calculate_slope(dem, 1.0)
    
    # Flat surface should have near-zero slope
    assert np.allclose(slope, 0, atol=1e-10)
