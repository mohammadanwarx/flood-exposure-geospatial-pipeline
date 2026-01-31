"""
Processing Module

Functions for processing geospatial data:
- Data cubes (xarray operations)
- Tensors (NumPy/PyTorch operations)
- DEM processing (fill depressions, slope)
"""

from .cubes import (
    GeospatialCube,
    create_flood_cube,
    merge_cubes,
    resample_cube_temporal,
    calculate_temporal_trends,
)

from .tensors import (
    create_geospatial_tensor,
    normalize_tensor,
    apply_convolution,
    calculate_tensor_statistics,
    extract_patches,
    tensor_pca,
    resample_tensor,
)

from .dem_processing import (
    load_dem,
    fill_depressions,
    calculate_slope,
    resample_dem,
)

__all__ = [
    # Cubes
    "GeospatialCube",
    "create_flood_cube",
    "merge_cubes",
    "resample_cube_temporal",
    "calculate_temporal_trends",
    # Tensors
    "create_geospatial_tensor",
    "normalize_tensor",
    "apply_convolution",
    "calculate_tensor_statistics",
    "extract_patches",
    "tensor_pca",
    "resample_tensor",
    # DEM
    "load_dem",
    "fill_depressions",
    "calculate_slope",
    "resample_dem",
]
