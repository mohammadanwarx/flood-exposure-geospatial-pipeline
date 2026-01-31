"""
Tensor Operations Module

This module provides tensor-based operations for multi-dimensional
geospatial data analysis and processing.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import torch


def create_geospatial_tensor(arrays: List[np.ndarray]) -> np.ndarray:
    """
    Create a multi-dimensional tensor from geospatial arrays.
    
    Parameters
    ----------
    arrays : List[np.ndarray]
        List of 2D arrays to stack into tensor
        
    Returns
    -------
    np.ndarray
        3D tensor with shape (bands, rows, cols)
    """
    if not arrays:
        raise ValueError("Input arrays list is empty")
    
    # Check all arrays have same shape
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(f"All arrays must have the same shape. Got: {shapes}")
    
    tensor = np.stack(arrays, axis=0)
    return tensor


def normalize_tensor(tensor: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize tensor values.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor
    method : str
        Normalization method: 'minmax', 'zscore', 'robust'
        
    Returns
    -------
    np.ndarray
        Normalized tensor
    """
    if method == 'minmax':
        # Min-Max normalization [0, 1]
        min_val = np.min(tensor, axis=(1, 2), keepdims=True)
        max_val = np.max(tensor, axis=(1, 2), keepdims=True)
        normalized = (tensor - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'zscore':
        # Z-score standardization
        mean = np.mean(tensor, axis=(1, 2), keepdims=True)
        std = np.std(tensor, axis=(1, 2), keepdims=True)
        normalized = (tensor - mean) / (std + 1e-8)
    
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = np.median(tensor, axis=(1, 2), keepdims=True)
        q75 = np.percentile(tensor, 75, axis=(1, 2), keepdims=True)
        q25 = np.percentile(tensor, 25, axis=(1, 2), keepdims=True)
        iqr = q75 - q25
        normalized = (tensor - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def apply_convolution(tensor: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply convolution operation to tensor.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (bands, rows, cols)
    kernel : np.ndarray
        Convolution kernel
        
    Returns
    -------
    np.ndarray
        Convolved tensor
    """
    from scipy.ndimage import convolve
    
    result = np.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        result[i] = convolve(tensor[i], kernel, mode='constant', cval=0.0)
    
    return result


def calculate_tensor_statistics(tensor: np.ndarray) -> dict:
    """
    Calculate comprehensive statistics for tensor.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor
        
    Returns
    -------
    dict
        Dictionary containing various statistics
    """
    stats = {}
    
    for i in range(tensor.shape[0]):
        band = tensor[i]
        band_stats = {
            'mean': float(np.mean(band)),
            'std': float(np.std(band)),
            'min': float(np.min(band)),
            'max': float(np.max(band)),
            'median': float(np.median(band)),
            'q25': float(np.percentile(band, 25)),
            'q75': float(np.percentile(band, 75))
        }
        stats[f'band_{i}'] = band_stats
    
    return stats


def extract_patches(tensor: np.ndarray, patch_size: Tuple[int, int], 
                   stride: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Extract patches from tensor for analysis or training.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (bands, rows, cols)
    patch_size : Tuple[int, int]
        Size of patches (height, width)
    stride : Tuple[int, int], optional
        Stride for patch extraction (default: same as patch_size)
        
    Returns
    -------
    np.ndarray
        Array of patches (n_patches, bands, patch_h, patch_w)
    """
    if stride is None:
        stride = patch_size
    
    bands, height, width = tensor.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    patches = []
    for i in range(0, height - patch_h + 1, stride_h):
        for j in range(0, width - patch_w + 1, stride_w):
            patch = tensor[:, i:i+patch_h, j:j+patch_w]
            patches.append(patch)
    
    return np.array(patches)


def tensor_pca(tensor: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA on tensor bands.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (bands, rows, cols)
    n_components : int
        Number of principal components
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Transformed tensor and explained variance ratio
    """
    from sklearn.decomposition import PCA
    
    bands, rows, cols = tensor.shape
    
    # Reshape to (n_pixels, n_bands)
    reshaped = tensor.reshape(bands, -1).T
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(reshaped)
    
    # Reshape back to tensor
    result = transformed.T.reshape(n_components, rows, cols)
    
    return result, pca.explained_variance_ratio_


def resample_tensor(tensor: np.ndarray, scale_factor: float, 
                   method: str = 'bilinear') -> np.ndarray:
    """
    Resample tensor to different resolution.
    
    Parameters
    ----------
    tensor : np.ndarray
        Input tensor (bands, rows, cols)
    scale_factor : float
        Scale factor for resampling
    method : str
        Interpolation method: 'nearest', 'bilinear', 'bicubic'
        
    Returns
    -------
    np.ndarray
        Resampled tensor
    """
    from scipy.ndimage import zoom
    
    if method == 'nearest':
        order = 0
    elif method == 'bilinear':
        order = 1
    elif method == 'bicubic':
        order = 3
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    zoom_factors = (1, scale_factor, scale_factor)
    resampled = zoom(tensor, zoom_factors, order=order)
    
    return resampled
