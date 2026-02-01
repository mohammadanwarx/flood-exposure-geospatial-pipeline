"""Tests for exposure module."""

import pytest
import numpy as np
import geopandas as gpd
from src._03_analysis.exposure import (
    calculate_flood_depth,
    identify_flooded_areas,
    assess_population_exposure,
    generate_risk_map
)


def test_calculate_flood_depth():
    """Test flood depth calculation."""
    dem = np.array([
        [10, 8, 6],
        [8, 6, 4],
        [6, 4, 2]
    ], dtype=float)
    
    water_level = 7.0
    flood_depth = calculate_flood_depth(dem, water_level)
    
    assert isinstance(flood_depth, np.ndarray)
    assert flood_depth.shape == dem.shape
    # Only areas below water level should be flooded
    assert np.all(flood_depth >= 0)
    assert np.max(flood_depth) == 5.0  # 7 - 2 = 5


def test_identify_flooded_areas():
    """Test flooded area identification."""
    flood_depth = np.array([
        [0, 0.5, 1.0],
        [0, 1.5, 2.0],
        [0, 0, 0]
    ], dtype=float)
    
    flood_mask = identify_flooded_areas(flood_depth, threshold=0.0)
    
    assert isinstance(flood_mask, np.ndarray)
    assert flood_mask.shape == flood_depth.shape
    assert flood_mask.dtype == np.int32
    assert np.all((flood_mask == 0) | (flood_mask == 1))
    assert np.sum(flood_mask) == 4  # Four cells with depth > 0


def test_identify_flooded_areas_with_threshold():
    """Test flooded area identification with threshold."""
    flood_depth = np.array([
        [0, 0.5, 1.0],
        [0, 1.5, 2.0],
        [0, 0, 0]
    ], dtype=float)
    
    flood_mask = identify_flooded_areas(flood_depth, threshold=1.0)
    
    # Only cells with depth > 1.0
    assert np.sum(flood_mask) == 2  # Two cells


def test_assess_population_exposure():
    """Test population exposure assessment."""
    population = np.array([
        [100, 200, 300],
        [150, 250, 350],
        [50, 100, 150]
    ], dtype=float)
    
    flood_mask = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [0, 0, 0]
    ], dtype=np.int32)
    
    exposure_stats = assess_population_exposure(population, flood_mask)
    
    assert isinstance(exposure_stats, dict)
    assert 'exposed_population' in exposure_stats
    assert 'total_population' in exposure_stats
    assert 'exposure_rate_percent' in exposure_stats
    
    expected_exposed = 200 + 300 + 250 + 350
    assert np.isclose(exposure_stats['exposed_population'], expected_exposed)
    assert 0 <= exposure_stats['exposure_rate_percent'] <= 100


def test_generate_risk_map():
    """Test risk map generation."""
    flood_depth = np.array([
        [0, 0.5, 1.5],
        [2.5, 4.0, 0],
        [0.3, 1.0, 2.0]
    ], dtype=float)
    
    vulnerability = np.ones_like(flood_depth) * 0.5
    
    risk_map = generate_risk_map(flood_depth, vulnerability)
    
    assert isinstance(risk_map, np.ndarray)
    assert risk_map.shape == flood_depth.shape
    assert np.all(risk_map >= 0)
    assert np.all(risk_map <= 1.0)


def test_generate_risk_map_custom_weights():
    """Test risk map with custom depth weights."""
    flood_depth = np.array([[0, 1, 2, 3, 4]], dtype=float)
    vulnerability = np.ones_like(flood_depth)
    
    depth_weights = [(0.0, 0.0), (1.0, 0.3), (2.0, 0.6), (3.0, 1.0)]
    
    risk_map = generate_risk_map(flood_depth, vulnerability, depth_weights)
    
    assert isinstance(risk_map, np.ndarray)
    assert np.all(risk_map >= 0)
