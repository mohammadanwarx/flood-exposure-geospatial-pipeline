"""
Buildings Data Module

This module handles fetching building data from various APIs (primarily OSM/Overpass),
processing building geometries, enriching with attributes for exposure analysis,
and clipping to study areas.

Key Features:
- Multi-source building data fetching (Overpass API, osmnx)
- Automatic building classification and enrichment
- Study area clipping to focus analysis on relevant buildings
- Multiple export formats (GeoJSON, GeoPackage, Shapefile, CSV)
"""
import sys
from pathlib import Path
import requests
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Optional, Tuple
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)


class BuildingsAPIError(Exception):
    """Custom exception for buildings API errors."""
    pass


def fetch_buildings_from_overpass(bbox: Tuple[float, float, float, float],
                                 timeout: int = 60) -> gpd.GeoDataFrame:
    """
    Fetch building data from OpenStreetMap Overpass API.
    
    Parameters
    ----------
    bbox : Tuple[float, float, float, float]
        Bounding box as (south, west, north, east) in lat/lon
    timeout : int, optional
        Request timeout in seconds (default: 60)
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with building geometries and attributes
        
    Raises
    ------
    BuildingsAPIError
        If API request fails or returns invalid data
        
    Notes
    -----
    Overpass API bbox format: (south, west, north, east)
    """
    # Convert bbox format if needed: (S, W, N, E)
    south, west, north, east = bbox
    
    # Overpass QL query for buildings
    overpass_query = f"""
    [bbox:{south},{west},{north},{east}];
    (
        way["building"];
        relation["building"];
    );
    out geom;
    """
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    try:
        response = requests.post(
            overpass_url,
            data=overpass_query,
            timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise BuildingsAPIError(f"Failed to fetch from Overpass API: {str(e)}")
    
    # Parse OSM data
    try:
        import json
        # Try alternative: use osmnx if available
        try:
            import osmnx as ox
            gdf = ox.features_from_bbox(
                (west, south, east, north), {'building': True}
            )
            gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            return gdf
        except ImportError:
            # Fallback: manual parsing of Overpass JSON
            data = response.json()
            buildings = _parse_overpass_response(data)
            if not buildings:
                logger.warning("No buildings found in the specified area")
                return gpd.GeoDataFrame(
                    columns=['geometry', 'name', 'building_type'],
                    crs='EPSG:4326'
                )
            gdf = gpd.GeoDataFrame(buildings, crs='EPSG:4326')
            return gdf
    except Exception as e:
        raise BuildingsAPIError(f"Failed to parse Overpass response: {str(e)}")


def _parse_overpass_response(data: Dict) -> List[Dict]:
    """
    Parse Overpass JSON response and extract building features.
    
    Parameters
    ----------
    data : Dict
        Parsed JSON response from Overpass API
        
    Returns
    -------
    List[Dict]
        List of building feature dictionaries
    """
    buildings = []
    
    if 'elements' not in data:
        return buildings
    
    # Process ways and relations
    for element in data['elements']:
        try:
            if element.get('type') in ['way', 'relation']:
                if 'geometry' not in element:
                    continue
                
                coords = element['geometry']
                if len(coords) < 3:
                    continue
                
                geometry = Polygon([(c['lon'], c['lat']) for c in coords])
                
                tags = element.get('tags', {})
                
                building_dict = {
                    'geometry': geometry,
                    'osm_id': element.get('id'),
                    'name': tags.get('name', ''),
                    'building_type': tags.get('building', 'yes'),
                    'levels': tags.get('building:levels', None),
                    'height': tags.get('height', None),
                }
                
                buildings.append(building_dict)
        except Exception as e:
            logger.debug(f"Failed to parse element {element.get('id')}: {str(e)}")
            continue
    
    return buildings


def fetch_buildings_from_bbox(bbox: Tuple[float, float, float, float],
                             provider: str = 'overpass',
                             **kwargs) -> gpd.GeoDataFrame:
    """
    Fetch building data from specified provider using bounding box.
    
    Parameters
    ----------
    bbox : Tuple[float, float, float, float]
        Bounding box as (south, west, north, east)
    provider : str, optional
        Data provider: 'overpass' (default), 'osmnx'
    **kwargs
        Additional arguments to pass to provider-specific function
        
    Returns
    -------
    gpd.GeoDataFrame
        Buildings GeoDataFrame
        
    Raises
    ------
    BuildingsAPIError
        If provider is not supported or fetch fails
    """
    if provider.lower() == 'overpass':
        return fetch_buildings_from_overpass(bbox, **kwargs)
    elif provider.lower() == 'osmnx':
        return _fetch_buildings_osmnx(bbox, **kwargs)
    else:
        raise BuildingsAPIError(f"Unsupported provider: {provider}")


def _fetch_buildings_osmnx(bbox: Tuple[float, float, float, float],
                          **kwargs) -> gpd.GeoDataFrame:
    """
    Fetch buildings using osmnx library.
    
    Parameters
    ----------
    bbox : Tuple[float, float, float, float]
        Bounding box as (south, west, north, east)
    **kwargs
        Additional arguments for osmnx
        
    Returns
    -------
    gpd.GeoDataFrame
        Buildings GeoDataFrame
    """
    try:
        import osmnx as ox
    except ImportError:
        raise BuildingsAPIError("osmnx not installed. Install with: pip install osmnx")
    
    south, west, north, east = bbox
    
    try:
        gdf = ox.features_from_bbox(
            (west, south, east, north), {'building': True}
        )
        
        # Filter to only polygon geometries
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        
        # Ensure CRS is set
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
        
        return gdf
    except Exception as e:
        raise BuildingsAPIError(f"Failed to fetch buildings with osmnx: {str(e)}")


def filter_buildings_by_extent(gdf: gpd.GeoDataFrame,
                              extent_geom: Polygon) -> gpd.GeoDataFrame:
    """
    Filter buildings to those within a specified extent/geometry (clips to study area).
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Buildings GeoDataFrame
    extent_geom : Polygon
        Extent geometry for filtering/clipping
        
    Returns
    -------
    gpd.GeoDataFrame
        Filtered and clipped buildings GeoDataFrame (buildings within extent)
        
    Notes
    -----
    This function performs a spatial clip operation, keeping only buildings
    that intersect with the extent geometry. Buildings that cross the boundary
    are clipped to the extent.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    
    # Ensure CRS compatibility
    if isinstance(extent_geom, gpd.GeoSeries):
        extent_geom = extent_geom[0]
    
    # Clip buildings to extent (keeps only buildings within the area)
    clipped = gdf.clip(extent_geom)
    
    if len(clipped) == 0:
        logger.warning(f"No buildings found within extent. Original: {len(gdf)}, After clip: {len(clipped)}")
    else:
        logger.info(f"Clipped buildings: {len(gdf)} → {len(clipped)} (study area only)")
    
    return clipped


def enrich_buildings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Enrich buildings GeoDataFrame with computed attributes.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Buildings GeoDataFrame
        
    Returns
    -------
    gpd.GeoDataFrame
        Enriched GeoDataFrame with additional columns
    """
    gdf = gdf.copy()

    # Ensure 'building_type' column exists
    if 'building_type' not in gdf.columns:
        if 'building' in gdf.columns:
            gdf['building_type'] = gdf['building']
        else:
            gdf['building_type'] = 'yes'

    # Calculate building area
    gdf['area_sqm'] = gdf.geometry.to_crs('EPSG:3857').area

    # Calculate building centroid
    gdf['centroid'] = gdf.geometry.centroid

    # Convert levels to numeric if present
    if 'levels' in gdf.columns:
        gdf['levels'] = pd.to_numeric(gdf['levels'], errors='coerce')
        gdf['levels'].fillna(1, inplace=True)

    # Convert height to numeric if present
    if 'height' in gdf.columns:
        gdf['height'] = pd.to_numeric(gdf['height'], errors='coerce')
        # Estimate height if not available but levels are
        if 'levels' in gdf.columns:
            gdf['height'].fillna(gdf['levels'] * 3.0, inplace=True)
    else:
        if 'levels' in gdf.columns:
            gdf['height'] = gdf['levels'] * 3.0

    # Assign building class based on building_type
    gdf['building_class'] = gdf['building_type'].apply(_classify_building)

    return gdf


def _classify_building(building_type: str) -> str:
    """
    Classify building based on type.
    
    Parameters
    ----------
    building_type : str
        Building type from OSM
        
    Returns
    -------
    str
        Building class: 'residential', 'commercial', 'industrial', 'public', 'other'
    """
    if not isinstance(building_type, str):
        return 'other'
    
    building_type = building_type.lower()
    
    residential = ['house', 'apartment', 'residential', 'dwelling', 'detached']
    commercial = ['shop', 'commercial', 'retail', 'office', 'bank', 'supermarket']
    industrial = ['industrial', 'warehouse', 'factory', 'plant']
    public = ['school', 'hospital', 'church', 'government', 'civic', 'public']
    
    for keyword in residential:
        if keyword in building_type:
            return 'residential'
    for keyword in commercial:
        if keyword in building_type:
            return 'commercial'
    for keyword in industrial:
        if keyword in building_type:
            return 'industrial'
    for keyword in public:
        if keyword in building_type:
            return 'public'
    
    return 'other'


def save_buildings(gdf: gpd.GeoDataFrame, filepath: str) -> None:
    """
    Save buildings GeoDataFrame to file.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Buildings GeoDataFrame
    filepath : str
        Output filepath (supports .geojson, .gpkg, .shp)
    """
    filepath_lower = filepath.lower()
    
    if filepath_lower.endswith('.geojson'):
        gdf.to_file(filepath, driver='GeoJSON')
    elif filepath_lower.endswith('.gpkg'):
        gdf.to_file(filepath, driver='GPKG')
    elif filepath_lower.endswith('.shp'):
        gdf.to_file(filepath, driver='ESRI Shapefile')
    else:
        gdf.to_file(filepath)
    
    logger.info(f"Buildings saved to {filepath}")


def load_buildings(filepath: str) -> gpd.GeoDataFrame:
    """
    Load buildings GeoDataFrame from file.
    
    Parameters
    ----------
    filepath : str
        Input filepath
        
    Returns
    -------
    gpd.GeoDataFrame
        Buildings GeoDataFrame
    """
    gdf = gpd.read_file(filepath)
    logger.info(f"Loaded {len(gdf)} buildings from {filepath}")
    return gdf



# ===============================================
# Population Estimation Functions


def estimate_population_from_buildings(buildings_gdf: gpd.GeoDataFrame, 
                                       household_size: int = 6) -> pd.Series:
    """
    Estimate population from building count using household size assumption.
    
    ASSUMPTION: 
    -----------
    - One household per building
    - Average household size = 6 persons (default, customizable)
    - Linear population estimation: Population = N_buildings × household_size
    
    This is a simplified approach suitable for areas lacking detailed population data.
    Assumes residential use for all buildings; can be refined with building type filters.
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        GeoDataFrame containing building geometries and attributes
    household_size : int, optional
        Average persons per household per building (default: 6)
        Based on regional demographic data
        
    Returns
    -------
    pd.Series
        Estimated population per building
        
    Raises
    ------
    ValueError
        If GeoDataFrame is empty or household_size <= 0
        
    Notes
    -----
    In production use, this should be refined with:
    - Building type classification (residential, commercial, etc.)
    - Building area/footprint size weighting
    - Known population census data for calibration
    
    Examples
    --------
    >>> buildings_gdf['population'] = estimate_population_from_buildings(buildings_gdf)
    >>> total_pop = buildings_gdf['population'].sum()
    """
    if buildings_gdf.empty:
        raise ValueError("GeoDataFrame is empty. Cannot estimate population.")
    
    if household_size <= 0:
        raise ValueError(f"household_size must be positive, got {household_size}")
    
    # Create population column: one household per building
    population = pd.Series([household_size] * len(buildings_gdf), 
                          index=buildings_gdf.index)
    
    logger.info(
        f"Estimated population from {len(buildings_gdf)} buildings: "
        f"Total = {population.sum()} persons "
        f"(assumption: {household_size} persons/building)"
    )
    
    return population


def add_population_attribute(buildings_gdf: gpd.GeoDataFrame, 
                            household_size: int = 6) -> gpd.GeoDataFrame:
    """
    Add estimated population as a column to buildings GeoDataFrame.
    
    Parameters
    ----------
    buildings_gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    household_size : int, optional
        Average household size (default: 6)
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with 'population' column added
        
    Examples
    --------
    >>> buildings_with_pop = add_population_attribute(buildings_gdf, household_size=6)
    """
    gdf = buildings_gdf.copy()
    gdf['population'] = estimate_population_from_buildings(gdf, household_size)
    
    logger.info(
        f"Added population attribute to {len(gdf)} buildings. "
        f"Total population: {gdf['population'].sum()}"
    )
    
    return gdf



#==============================================
# fetch building for data acquistion 

if __name__ == "__main__":
    import geopandas as gpd
    import sys
    from pathlib import Path
    print("Loading AOI shapefile...")
    try:
        base_dir = Path(__file__).parent.parent.parent
    except NameError:
        base_dir = Path.cwd()
        # If not in project root, try to find 'data' folder up the tree
        if not (base_dir / "data").exists():
            for parent in [base_dir.parent, base_dir.parent.parent]:
                if (parent / "data").exists():
                    base_dir = parent
                    break

    aoi_path = base_dir / "data" / "raw" / "vector" / "AOI.shp"
    aoi = gpd.read_file(aoi_path)
    bbox = (aoi.bounds.miny[0], aoi.bounds.minx[0], aoi.bounds.maxy[0], aoi.bounds.maxx[0])
    print(f"AOI bounding box: {bbox}")
    print("Fetching buildings from OSMnx...")
    gdf = fetch_buildings_from_bbox(bbox, provider='osmnx')
    print(f"Fetched {len(gdf)} buildings within bbox. Enriching...")
    gdf = enrich_buildings(gdf)
    # Drop centroid geometry column before saving
    if 'centroid' in gdf.columns:
        gdf = gdf.drop(columns=['centroid'])
    output_path = base_dir / "data" / "raw" / "vector" / "buildings" / "buildings_aoi.geojson"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"Saved {len(gdf)} buildings to {output_path}")