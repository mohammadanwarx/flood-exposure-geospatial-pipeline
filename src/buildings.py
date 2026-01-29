"""
Building Data Processing Module

This module provides functions for loading building footprints,
filtering by AOI, and estimating population from building data.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
from typing import Optional, Dict, Tuple
from pathlib import Path


def load_buildings_from_csv(filepath: str, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Load building data from CSV file (e.g., Google Open Buildings).
    
    Parameters
    ----------
    filepath : str
        Path to CSV file with building data
    crs : str, optional
        Coordinate reference system, default EPSG:4326
        
    Returns
    -------
    gpd.GeoDataFrame
        Building footprints as geodataframe
    """
    df = pd.read_csv(filepath)
    
    # Assuming columns: latitude, longitude, area_in_meters, confidence
    # Adjust column names based on actual data format
    if 'geometry' in df.columns:
        # If geometry already exists (WKT format)
        from shapely import wkt
        df['geometry'] = df['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, crs=crs)
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        # Create point geometry from coordinates
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs=crs
        )
    else:
        raise ValueError("CSV must contain 'geometry' or 'latitude'/'longitude' columns")
    
    return gdf


def filter_buildings_by_aoi(buildings: gpd.GeoDataFrame, 
                            aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filter buildings that fall within area of interest.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprints
    aoi : gpd.GeoDataFrame
        Area of interest polygon
        
    Returns
    -------
    gpd.GeoDataFrame
        Filtered buildings within AOI
    """
    # Ensure same CRS
    if buildings.crs != aoi.crs:
        buildings = buildings.to_crs(aoi.crs)
    
    # Spatial join to filter buildings within AOI
    buildings_filtered = gpd.sjoin(buildings, aoi, predicate='within')
    
    return buildings_filtered


def estimate_population_from_buildings(buildings: gpd.GeoDataFrame,
                                      method: str = "area_based",
                                      persons_per_sqm: float = 0.05,
                                      persons_per_building: float = 3.5) -> gpd.GeoDataFrame:
    """
    Estimate population from building footprints.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprints with area information
    method : str, optional
        Estimation method: 'area_based', 'count_based', or 'mixed'
        - area_based: population = building_area * persons_per_sqm
        - count_based: population = num_buildings * persons_per_building
        - mixed: uses area if available, otherwise count
    persons_per_sqm : float, optional
        Average persons per square meter (default 0.05 = 50 persons per 1000 sqm)
    persons_per_building : float, optional
        Average persons per building (default 3.5)
        
    Returns
    -------
    gpd.GeoDataFrame
        Buildings with estimated population column
    """
    buildings = buildings.copy()
    
    if method == "area_based":
        if 'area_in_meters' not in buildings.columns:
            # Calculate area from geometry if not provided
            buildings['area_in_meters'] = buildings.geometry.area
        
        buildings['estimated_population'] = buildings['area_in_meters'] * persons_per_sqm
        
    elif method == "count_based":
        buildings['estimated_population'] = persons_per_building
        
    elif method == "mixed":
        if 'area_in_meters' in buildings.columns:
            buildings['estimated_population'] = buildings['area_in_meters'] * persons_per_sqm
        else:
            buildings['estimated_population'] = persons_per_building
    else:
        raise ValueError("Method must be 'area_based', 'count_based', or 'mixed'")
    
    return buildings


def aggregate_population_by_grid(buildings: gpd.GeoDataFrame,
                                 grid_size: float,
                                 crs: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Aggregate building population to regular grid cells.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with estimated_population column
    grid_size : float
        Grid cell size in units of CRS (meters if metric)
    crs : str, optional
        CRS for grid creation, uses buildings CRS if None
        
    Returns
    -------
    gpd.GeoDataFrame
        Grid cells with aggregated population
    """
    from shapely.geometry import box
    
    if crs is None:
        crs = buildings.crs
    elif buildings.crs != crs:
        buildings = buildings.to_crs(crs)
    
    # Get bounds
    minx, miny, maxx, maxy = buildings.total_bounds
    
    # Create grid
    cols = int(np.ceil((maxx - minx) / grid_size))
    rows = int(np.ceil((maxy - miny) / grid_size))
    
    grid_cells = []
    for i in range(cols):
        for j in range(rows):
            cell_minx = minx + i * grid_size
            cell_miny = miny + j * grid_size
            cell_maxx = cell_minx + grid_size
            cell_maxy = cell_miny + grid_size
            grid_cells.append(box(cell_minx, cell_miny, cell_maxx, cell_maxy))
    
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=crs)
    grid['cell_id'] = range(len(grid))
    
    # Spatial join buildings to grid
    joined = gpd.sjoin(buildings, grid, predicate='within')
    
    # Aggregate population by grid cell
    pop_by_cell = joined.groupby('cell_id')['estimated_population'].sum()
    
    grid = grid.merge(pop_by_cell, left_on='cell_id', right_index=True, how='left')
    grid['estimated_population'] = grid['estimated_population'].fillna(0)
    
    return grid


def calculate_building_statistics(buildings: gpd.GeoDataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics for building dataset.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprints
        
    Returns
    -------
    Dict[str, float]
        Dictionary of statistics
    """
    stats = {
        'total_buildings': len(buildings),
        'total_area_sqm': buildings.geometry.area.sum() if buildings.geometry.type[0] == 'Polygon' else buildings.get('area_in_meters', pd.Series([0])).sum(),
    }
    
    if 'estimated_population' in buildings.columns:
        stats['total_population'] = buildings['estimated_population'].sum()
        stats['avg_population_per_building'] = buildings['estimated_population'].mean()
    
    if 'area_in_meters' in buildings.columns:
        stats['avg_building_area'] = buildings['area_in_meters'].mean()
        stats['median_building_area'] = buildings['area_in_meters'].median()
    
    return stats


def export_buildings(buildings: gpd.GeoDataFrame, 
                    filepath: str,
                    format: str = "geojson") -> None:
    """
    Export building data to file.
    
    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprints to export
    filepath : str
        Output file path
    format : str, optional
        Output format: 'geojson', 'shapefile', 'gpkg'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() in ['geojson', 'json']:
        buildings.to_file(filepath, driver='GeoJSON')
    elif format.lower() in ['shapefile', 'shp']:
        buildings.to_file(filepath, driver='ESRI Shapefile')
    elif format.lower() in ['gpkg', 'geopackage']:
        buildings.to_file(filepath, driver='GPKG')
    else:
        raise ValueError(f"Unsupported format: {format}")
