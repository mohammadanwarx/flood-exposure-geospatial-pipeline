"""Tests for tensor operations module."""

import pytest
import numpy as np
from src._02_processing.tensors import (
    create_geospatial_tensor,
    normalize_tensor,
    apply_convolution,
    calculate_tensor_statistics,
    extract_patches
)


def test_create_geospatial_tensor():
    """Test tensor creation from arrays."""
    arrays = [
        np.random.rand(10, 10),
        np.random.rand(10, 10),
        np.random.rand(10, 10)
    ]
    
    tensor = create_geospatial_tensor(arrays)
    
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 10, 10)


def test_create_geospatial_tensor_empty():
    """Test tensor creation with empty list."""
    with pytest.raises(ValueError, match="empty"):
        create_geospatial_tensor([])


def test_create_geospatial_tensor_mismatched_shapes():
    """Test tensor creation with mismatched shapes."""
    arrays = [
        np.random.rand(10, 10),
        np.random.rand(10, 15)  # Different shape
    ]
    
    with pytest.raises(ValueError, match="same shape"):
        create_geospatial_tensor(arrays)


def test_normalize_tensor_minmax():
    """Test min-max normalization."""
    tensor = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[10, 20, 30], [40, 50, 60]]
    ], dtype=float)
    
    normalized = normalize_tensor(tensor, method='minmax')
    
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == tensor.shape
    # Values should be between 0 and 1
    assert np.all(normalized >= 0) and np.all(normalized <= 1)


def test_normalize_tensor_zscore():
    """Test z-score normalization."""
    tensor = np.random.rand(3, 10, 10)
    
    normalized = normalize_tensor(tensor, method='zscore')
    
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == tensor.shape


def test_normalize_tensor_robust():
    """Test robust normalization."""
    tensor = np.random.rand(2, 10, 10)
    
    normalized = normalize_tensor(tensor, method='robust')
    
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == tensor.shape


def test_normalize_tensor_invalid_method():
    """Test normalization with invalid method."""
    tensor = np.random.rand(2, 5, 5)
    
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalize_tensor(tensor, method='invalid')


def test_apply_convolution():
    """Test convolution operation."""
    tensor = np.random.rand(2, 10, 10)
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    
    result = apply_convolution(tensor, kernel)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == tensor.shape


def test_calculate_tensor_statistics():
    """Test tensor statistics calculation."""
    tensor = np.random.rand(3, 10, 10)
    
    stats = calculate_tensor_statistics(tensor)
    
    assert isinstance(stats, dict)
    assert len(stats) == 3  # Three bands
    
    for i in range(3):
        band_key = f'band_{i}'
        assert band_key in stats
        assert 'mean' in stats[band_key]
        assert 'std' in stats[band_key]
        assert 'min' in stats[band_key]
        assert 'max' in stats[band_key]


def test_extract_patches():
    """Test patch extraction."""
    tensor = np.random.rand(3, 20, 20)
    patch_size = (5, 5)
    
    patches = extract_patches(tensor, patch_size)
    
    assert isinstance(patches, np.ndarray)
    assert patches.ndim == 4
    assert patches.shape[1:] == (3, 5, 5)
    # Should have 16 patches (4x4 grid)
    assert patches.shape[0] == 16


def test_extract_patches_with_stride():
    """Test patch extraction with custom stride."""
    tensor = np.random.rand(2, 20, 20)
    patch_size = (5, 5)
    stride = (2, 2)
    
    patches = extract_patches(tensor, patch_size, stride)
    
    assert isinstance(patches, np.ndarray)
    assert patches.shape[1:] == (2, 5, 5)
    # With stride 2, should have more patches
    assert patches.shape[0] > 16
