"""Tests for hydrology module."""

import pytest
import numpy as np
from src._03_analysis.hydrology import (
    calculate_flow_direction,
    calculate_flow_accumulation,
    extract_stream_network,
    calculate_twi
)


def test_calculate_flow_direction():
    """Test flow direction calculation."""
    # Create a simple DEM
    dem = np.array([
        [10, 9, 8],
        [9, 8, 7],
        [8, 7, 6]
    ], dtype=float)
    
    flow_dir = calculate_flow_direction(dem)
    
    assert isinstance(flow_dir, np.ndarray)
    assert flow_dir.shape == dem.shape
    assert flow_dir.dtype == np.int32


def test_calculate_flow_accumulation():
    """Test flow accumulation calculation."""
    # Create a simple flow direction array
    flow_dir = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.int32)
    
    flow_acc = calculate_flow_accumulation(flow_dir)
    
    assert isinstance(flow_acc, np.ndarray)
    assert flow_acc.shape == flow_dir.shape
    assert np.all(flow_acc >= 1)


def test_extract_stream_network():
    """Test stream network extraction."""
    # Create flow accumulation with some high values
    flow_acc = np.array([
        [1, 1, 1],
        [1, 10, 1],
        [1, 100, 1]
    ], dtype=float)
    
    threshold = 50.0
    streams = extract_stream_network(flow_acc, threshold)
    
    assert isinstance(streams, np.ndarray)
    assert streams.shape == flow_acc.shape
    assert streams.dtype == np.int32
    assert np.all((streams == 0) | (streams == 1))
    # Should have at least one stream cell
    assert np.any(streams == 1)


def test_calculate_twi():
    """Test Topographic Wetness Index calculation."""
    dem = np.array([
        [10, 9, 8],
        [9, 8, 7],
        [8, 7, 6]
    ], dtype=float)
    
    flow_acc = np.ones_like(dem) * 10
    slope = np.ones_like(dem) * 5  # 5 degrees
    cellsize = 10.0
    
    twi = calculate_twi(dem, flow_acc, slope, cellsize)
    
    assert isinstance(twi, np.ndarray)
    assert twi.shape == dem.shape
    assert not np.any(np.isnan(twi))
    assert not np.any(np.isinf(twi))
