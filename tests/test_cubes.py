"""Tests for data cubes module."""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from src._02_processing.cubes import (
    GeospatialCube,
    create_flood_cube,
    merge_cubes,
    resample_cube_temporal
)


@pytest.fixture
def sample_cube():
    """Create a sample geospatial cube for testing."""
    data = np.random.rand(5, 10, 10)
    times = [datetime(2024, 1, i+1) for i in range(5)]
    x_coords = np.arange(10)
    y_coords = np.arange(10)
    
    data_array = xr.DataArray(
        data,
        coords={'time': times, 'y': y_coords, 'x': x_coords},
        dims=['time', 'y', 'x'],
        name='test_variable'
    )
    
    return GeospatialCube(data_array)


def test_geospatial_cube_init():
    """Test GeospatialCube initialization."""
    data = xr.DataArray(
        np.random.rand(3, 5, 5),
        coords={'time': [1, 2, 3], 'y': range(5), 'x': range(5)},
        dims=['time', 'y', 'x'],
        name='test_variable'  # DataArray must be named for to_dataset()
    )
    
    cube = GeospatialCube(data)
    
    assert isinstance(cube, GeospatialCube)
    assert isinstance(cube.data, xr.Dataset)


def test_cube_get_shape(sample_cube):
    """Test getting cube shape."""
    shape = sample_cube.get_shape()
    
    assert isinstance(shape, dict)
    assert 'time' in shape
    assert 'y' in shape
    assert 'x' in shape
    assert shape['time'] == 5
    assert shape['y'] == 10
    assert shape['x'] == 10


def test_cube_get_extent(sample_cube):
    """Test getting spatial extent."""
    extent = sample_cube.get_extent()
    
    assert isinstance(extent, dict)
    assert 'x' in extent
    assert 'y' in extent
    assert extent['x'][0] <= extent['x'][1]
    assert extent['y'][0] <= extent['y'][1]


def test_cube_select_time_slice(sample_cube):
    """Test selecting a time slice."""
    time_slice = sample_cube.select_time_slice(0)
    
    assert isinstance(time_slice, xr.Dataset)
    assert 'time' not in time_slice.dims


def test_cube_select_spatial_window(sample_cube):
    """Test selecting a spatial window."""
    window = sample_cube.select_spatial_window(
        x_range=(2, 7),
        y_range=(3, 8)
    )
    
    assert isinstance(window, xr.Dataset)
    assert window.dims['x'] <= 6
    assert window.dims['y'] <= 6


def test_cube_aggregate_temporal(sample_cube):
    """Test temporal aggregation."""
    aggregated = sample_cube.aggregate_temporal(method='mean')
    
    assert isinstance(aggregated, xr.Dataset)
    assert 'time' not in aggregated.dims


def test_cube_aggregate_temporal_methods(sample_cube):
    """Test different aggregation methods."""
    methods = ['mean', 'max', 'min', 'sum', 'std']
    
    for method in methods:
        result = sample_cube.aggregate_temporal(method=method)
        assert isinstance(result, xr.Dataset)


def test_cube_compute_statistics(sample_cube):
    """Test computing statistics."""
    stats = sample_cube.compute_statistics('test_variable')
    
    assert isinstance(stats, dict)
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'median' in stats


def test_create_flood_cube():
    """Test creating a flood cube."""
    flood_maps = [np.random.rand(5, 5) for _ in range(3)]
    timestamps = [datetime(2024, 1, i+1) for i in range(3)]
    x_coords = np.arange(5)
    y_coords = np.arange(5)
    
    cube = create_flood_cube(flood_maps, timestamps, x_coords, y_coords)
    
    assert isinstance(cube, GeospatialCube)
    assert cube.data.dims['time'] == 3
    assert cube.data.dims['x'] == 5
    assert cube.data.dims['y'] == 5


def test_merge_cubes():
    """Test merging multiple cubes."""
    # Create two simple cubes
    data1 = xr.DataArray(
        np.random.rand(2, 3, 3),
        coords={'time': [1, 2], 'y': range(3), 'x': range(3)},
        dims=['time', 'y', 'x'],
        name='var1'
    )
    
    data2 = xr.DataArray(
        np.random.rand(2, 3, 3),
        coords={'time': [1, 2], 'y': range(3), 'x': range(3)},
        dims=['time', 'y', 'x'],
        name='var2'
    )
    
    cube1 = GeospatialCube(data1)
    cube2 = GeospatialCube(data2)
    
    merged = merge_cubes([cube1, cube2])
    
    assert isinstance(merged, GeospatialCube)
    assert 'var1' in merged.data
    assert 'var2' in merged.data
