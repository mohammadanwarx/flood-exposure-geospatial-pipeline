"""
Data Cubes Module

This module provides functionality for creating and analyzing
multi-dimensional geospatial data cubes (space-time or multi-attribute cubes).
"""

import numpy as np
import xarray as xr
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime


class GeospatialCube:
    """
    A class for managing geospatial data cubes.
    
    Attributes
    ----------
    data : xr.Dataset
        Xarray dataset containing the cube data
    spatial_dims : tuple
        Names of spatial dimensions (e.g., ('x', 'y'))
    temporal_dim : str
        Name of temporal dimension (if applicable)
    """
    
    def __init__(self, data: Union[xr.Dataset, xr.DataArray], 
                 spatial_dims: Tuple[str, str] = ('x', 'y'),
                 temporal_dim: Optional[str] = 'time'):
        """
        Initialize a GeospatialCube.
        
        Parameters
        ----------
        data : Union[xr.Dataset, xr.DataArray]
            Input data
        spatial_dims : Tuple[str, str]
            Names of spatial dimensions
        temporal_dim : str, optional
            Name of temporal dimension
        """
        if isinstance(data, xr.DataArray):
            self.data = data.to_dataset()
        else:
            self.data = data
        
        self.spatial_dims = spatial_dims
        self.temporal_dim = temporal_dim
    
    def get_shape(self) -> Dict[str, int]:
        """Get the shape of the cube."""
        return {dim: size for dim, size in self.data.dims.items()}
    
    def get_extent(self) -> Dict[str, Tuple[float, float]]:
        """Get the spatial extent of the cube."""
        extent = {}
        for dim in self.spatial_dims:
            if dim in self.data.coords:
                coords = self.data.coords[dim].values
                extent[dim] = (float(coords.min()), float(coords.max()))
        return extent
    
    def select_time_slice(self, time_index: Union[int, datetime]) -> xr.Dataset:
        """
        Select a time slice from the cube.
        
        Parameters
        ----------
        time_index : Union[int, datetime]
            Time index or datetime
            
        Returns
        -------
        xr.Dataset
            Selected time slice
        """
        if self.temporal_dim and self.temporal_dim in self.data.dims:
            return self.data.isel({self.temporal_dim: time_index})
        else:
            raise ValueError("No temporal dimension in cube")
    
    def select_spatial_window(self, x_range: Tuple[float, float], 
                            y_range: Tuple[float, float]) -> xr.Dataset:
        """
        Select a spatial window from the cube.
        
        Parameters
        ----------
        x_range : Tuple[float, float]
            X coordinate range (min, max)
        y_range : Tuple[float, float]
            Y coordinate range (min, max)
            
        Returns
        -------
        xr.Dataset
            Selected spatial window
        """
        x_dim, y_dim = self.spatial_dims
        
        subset = self.data.sel({
            x_dim: slice(x_range[0], x_range[1]),
            y_dim: slice(y_range[0], y_range[1])
        })
        
        return subset
    
    def aggregate_temporal(self, method: str = 'mean') -> xr.Dataset:
        """
        Aggregate cube along temporal dimension.
        
        Parameters
        ----------
        method : str
            Aggregation method: 'mean', 'max', 'min', 'sum', 'std'
            
        Returns
        -------
        xr.Dataset
            Temporally aggregated dataset
        """
        if not self.temporal_dim or self.temporal_dim not in self.data.dims:
            raise ValueError("No temporal dimension to aggregate")
        
        if method == 'mean':
            return self.data.mean(dim=self.temporal_dim)
        elif method == 'max':
            return self.data.max(dim=self.temporal_dim)
        elif method == 'min':
            return self.data.min(dim=self.temporal_dim)
        elif method == 'sum':
            return self.data.sum(dim=self.temporal_dim)
        elif method == 'std':
            return self.data.std(dim=self.temporal_dim)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def compute_statistics(self, variable: str) -> Dict[str, float]:
        """
        Compute statistics for a variable in the cube.
        
        Parameters
        ----------
        variable : str
            Variable name
            
        Returns
        -------
        Dict[str, float]
            Dictionary of statistics
        """
        if variable not in self.data:
            raise ValueError(f"Variable {variable} not found in cube")
        
        data_array = self.data[variable]
        
        return {
            'mean': float(data_array.mean().values),
            'std': float(data_array.std().values),
            'min': float(data_array.min().values),
            'max': float(data_array.max().values),
            'median': float(data_array.median().values)
        }
    
    def save(self, filepath: str, format: str = 'netcdf'):
        """
        Save cube to file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        format : str
            Output format: 'netcdf', 'zarr'
        """
        if format == 'netcdf':
            self.data.to_netcdf(filepath)
        elif format == 'zarr':
            self.data.to_zarr(filepath)
        else:
            raise ValueError(f"Unknown format: {format}")


def create_flood_cube(flood_maps: List[np.ndarray], 
                     timestamps: List[datetime],
                     x_coords: np.ndarray,
                     y_coords: np.ndarray,
                     variable_name: str = 'flood_depth') -> GeospatialCube:
    """
    Create a flood data cube from multiple flood maps.
    
    Parameters
    ----------
    flood_maps : List[np.ndarray]
        List of flood depth arrays
    timestamps : List[datetime]
        List of timestamps for each flood map
    x_coords : np.ndarray
        X coordinates
    y_coords : np.ndarray
        Y coordinates
    variable_name : str
        Name for the flood variable
        
    Returns
    -------
    GeospatialCube
        Flood data cube
    """
    # Stack arrays
    data_array = np.stack(flood_maps, axis=0)
    
    # Create xarray DataArray
    data = xr.DataArray(
        data_array,
        coords={
            'time': timestamps,
            'y': y_coords,
            'x': x_coords
        },
        dims=['time', 'y', 'x'],
        name=variable_name
    )
    
    return GeospatialCube(data, spatial_dims=('x', 'y'), temporal_dim='time')


def merge_cubes(cubes: List[GeospatialCube]) -> GeospatialCube:
    """
    Merge multiple geospatial cubes.
    
    Parameters
    ----------
    cubes : List[GeospatialCube]
        List of cubes to merge
        
    Returns
    -------
    GeospatialCube
        Merged cube
    """
    datasets = [cube.data for cube in cubes]
    merged = xr.merge(datasets)
    
    # Use spatial and temporal dims from first cube
    return GeospatialCube(
        merged,
        spatial_dims=cubes[0].spatial_dims,
        temporal_dim=cubes[0].temporal_dim
    )


def resample_cube_temporal(cube: GeospatialCube, 
                          frequency: str) -> GeospatialCube:
    """
    Resample cube to different temporal frequency.
    
    Parameters
    ----------
    cube : GeospatialCube
        Input cube
    frequency : str
        Target frequency (e.g., '1D', '1H', '1M')
        
    Returns
    -------
    GeospatialCube
        Resampled cube
    """
    if not cube.temporal_dim or cube.temporal_dim not in cube.data.dims:
        raise ValueError("Cube must have temporal dimension for resampling")
    
    resampled = cube.data.resample({cube.temporal_dim: frequency}).mean()
    
    return GeospatialCube(
        resampled,
        spatial_dims=cube.spatial_dims,
        temporal_dim=cube.temporal_dim
    )


def calculate_temporal_trends(cube: GeospatialCube, 
                             variable: str) -> xr.DataArray:
    """
    Calculate temporal trends for each pixel.
    
    Parameters
    ----------
    cube : GeospatialCube
        Input cube
    variable : str
        Variable name
        
    Returns
    -------
    xr.DataArray
        Trend coefficients for each pixel
    """
    if variable not in cube.data:
        raise ValueError(f"Variable {variable} not found in cube")
    
    if not cube.temporal_dim or cube.temporal_dim not in cube.data.dims:
        raise ValueError("Cube must have temporal dimension")
    
    data_array = cube.data[variable]
    
    # Simple linear trend calculation
    # TODO: Implement more sophisticated trend analysis
    trend = data_array.polyfit(dim=cube.temporal_dim, deg=1)
    
    return trend
