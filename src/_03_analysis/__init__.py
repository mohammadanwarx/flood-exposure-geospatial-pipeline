"""
Analysis Module

Functions for analyzing geospatial data:
- Hydrology (flow direction, accumulation, watersheds)
- Flood exposure (flood depth, risk mapping)
"""

from .hydrology import (
    calculate_flow_direction,
    calculate_flow_accumulation,
    delineate_watersheds,
    extract_stream_network,
    calculate_twi,
)

from .exposure import (
    calculate_flood_depth,
    identify_flooded_areas,
    calculate_exposure_statistics,
    assess_population_exposure,
    calculate_vulnerability_index,
    generate_risk_map,
)

__all__ = [
    # Hydrology
    "calculate_flow_direction",
    "calculate_flow_accumulation",
    "delineate_watersheds",
    "extract_stream_network",
    "calculate_twi",
    # Exposure
    "calculate_flood_depth",
    "identify_flooded_areas",
    "calculate_exposure_statistics",
    "assess_population_exposure",
    "calculate_vulnerability_index",
    "generate_risk_map",
]
