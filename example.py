"""
Example script demonstrating basic flood exposure analysis workflow.
"""

from src.data_io import load_raster, load_vector, save_vector
from src.preprocessing import reproject_vector, mask_raster_by_geometry
from src.analysis import compute_zonal_statistics
from src.visualization import plot_zonal_statistics
import os

# Define paths
DATA_DIR = "data/raw"
OUTPUT_DIR = "outputs"

# Example workflow
def main():
    """Run a basic flood exposure analysis."""
    
    print("Loading data...")
    # Load raster data (flood depth, population, etc.)
    # flood_data, metadata = load_raster(f"{DATA_DIR}/raster/flood_depth.tif")
    
    # Load vector data (administrative boundaries)
    # boundaries = load_vector(f"{DATA_DIR}/vector/boundaries.shp")
    
    print("Preprocessing...")
    # Ensure CRS match
    # boundaries = reproject_vector(boundaries, str(metadata['crs']))
    
    print("Computing zonal statistics...")
    # Calculate statistics
    # results = compute_zonal_statistics(
    #     f"{DATA_DIR}/raster/flood_depth.tif",
    #     boundaries,
    #     stats=['mean', 'max', 'sum', 'count']
    # )
    
    print("Saving results...")
    # Save results
    # save_vector(f"{OUTPUT_DIR}/flood_exposure_results.gpkg", results)
    
    print("Creating visualizations...")
    # Visualize
    # plot_zonal_statistics(
    #     results,
    #     'mean',
    #     title='Mean Flood Depth by Region',
    #     save_path=f"{OUTPUT_DIR}/flood_exposure_map.png"
    # )
    
    print("Analysis complete! Check the outputs folder.")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
