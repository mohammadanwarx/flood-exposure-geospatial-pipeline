"""
Script to download and load Google Open Buildings data for AOI
"""

import sys
sys.path.append('src')

import geopandas as gpd
import pandas as pd
from pathlib import Path
from buildings import (
    load_buildings_from_csv, 
    filter_buildings_by_aoi,
    estimate_population_from_buildings,
    calculate_building_statistics,
    export_buildings
)

# Paths
AOI_PATH = "data/raw/vector/AOI.shp"
BUILDINGS_CSV = "data/raw/buildings/195_buildings.csv.gz"  # Sudan
OUTPUT_PATH = "data/processed/buildings_aoi.geojson"

def main():
    print("Loading AOI...")
    aoi = gpd.read_file(AOI_PATH)
    print(f"AOI bounds: {aoi.total_bounds}")
    
    print("\nLoading buildings...")
    # If you have the full CSV
    if Path(BUILDINGS_CSV).exists():
        buildings = load_buildings_from_csv(BUILDINGS_CSV)
        print(f"Loaded {len(buildings)} buildings from CSV")
        
        print("\nFiltering by AOI...")
        buildings_aoi = filter_buildings_by_aoi(buildings, aoi)
    else:
        print(f"Building file not found: {BUILDINGS_CSV}")
        print("\nDownload using:")
        print("gsutil -m cp gs://open-buildings-data/v3/polygons_s2_level_4_gzip/195_buildings.csv.gz data/raw/buildings/")
        return
    
    print(f"Buildings in AOI: {len(buildings_aoi)}")
    
    print("\nEstimating population...")
    buildings_aoi = estimate_population_from_buildings(
        buildings_aoi,
        method="area_based",
        persons_per_sqm=0.05  # Adjust for your region
    )
    
    print("\nBuilding statistics:")
    stats = calculate_building_statistics(buildings_aoi)
    for key, value in stats.items():
        print(f"  {key}: {value:,.2f}")
    
    print(f"\nExporting to {OUTPUT_PATH}...")
    export_buildings(buildings_aoi, OUTPUT_PATH, format="geojson")
    
    print("Done!")

if __name__ == "__main__":
    main()
