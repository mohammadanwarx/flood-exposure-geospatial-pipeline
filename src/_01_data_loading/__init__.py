"""
Data Loading Module

Functions for loading all types of geospatial data:
- Buildings (CSV, GeoJSON, Shapefile)
- Rainfall (CHIRPS netCDF files)
- Vector data (AOI, boundaries)
"""

from .buildings import (
    load_buildings_from_csv,
    filter_buildings_by_aoi,
    estimate_population_from_buildings,
    aggregate_population_by_grid,
    calculate_building_statistics,
    export_buildings,
)

from .rainfall_processing import (
    CHIRPSProcessor,
    create_rainfall_datacube,
)

__all__ = [
    # Buildings
    "load_buildings_from_csv",
    "filter_buildings_by_aoi", 
    "estimate_population_from_buildings",
    "aggregate_population_by_grid",
    "calculate_building_statistics",
    "export_buildings",
    # Rainfall
    "CHIRPSProcessor",
    "create_rainfall_datacube",
]
