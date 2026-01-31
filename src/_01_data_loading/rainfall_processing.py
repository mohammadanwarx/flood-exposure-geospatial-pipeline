"""
CHIRPS Rainfall Processing Module

This module provides functionality for processing CHIRPS rainfall data:
- Merging multiple CHIRPS netCDF files
- Clipping datacubes to Area of Interest (AOI)
- Creating rainfall datacubes for geospatial analysis
"""

import xarray as xr
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import List, Union, Optional


class CHIRPSProcessor:
    """
    A class for processing CHIRPS (Climate Hazards Group InfraRed Precipitation 
    with Station data) rainfall datasets.
    """
    
    def __init__(self, chirps_dir: Union[str, Path], aoi_path: Union[str, Path]):
        """
        Initialize CHIRPS processor.
        
        Parameters
        ----------
        chirps_dir : Union[str, Path]
            Directory containing CHIRPS netCDF files
        aoi_path : Union[str, Path]
            Path to Area of Interest shapefile
        """
        self.chirps_dir = Path(chirps_dir)
        self.aoi_path = Path(aoi_path)
        self.aoi = None
        self.datacube = None
        
    def load_aoi(self) -> gpd.GeoDataFrame:
        """Load Area of Interest shapefile."""
        print(f"Loading AOI from: {self.aoi_path}")
        self.aoi = gpd.read_file(self.aoi_path)
        print(f"AOI CRS: {self.aoi.crs}")
        print(f"AOI bounds: {self.aoi.total_bounds}")
        return self.aoi
    
    def get_chirps_files(self, pattern: str = "chirps-v2.0.*.monthly.nc") -> List[Path]:
        """Get list of CHIRPS netCDF files from directory."""
        files = sorted(self.chirps_dir.glob(pattern))
        print(f"Found {len(files)} CHIRPS files")
        return files
    
    def merge_chirps_files(self, files: Optional[List[Path]] = None) -> xr.Dataset:
        """Merge multiple CHIRPS netCDF files into a single datacube."""
        if files is None:
            files = self.get_chirps_files()
        
        if not files:
            raise ValueError("No CHIRPS files found to merge")
        
        print(f"\nMerging {len(files)} CHIRPS files...")
        datasets = []
        for f in files:
            ds = xr.open_dataset(f)
            datasets.append(ds.load())  # Load into memory
            ds.close()
        
        self.datacube = xr.concat(datasets, dim='time')
        
        print(f"Merged datacube shape: {dict(self.datacube.sizes)}")
        return self.datacube
    
    def clip_to_aoi(self, datacube: Optional[xr.Dataset] = None) -> xr.Dataset:
        """Clip datacube to Area of Interest bounds."""
        if datacube is None:
            datacube = self.datacube
            
        if datacube is None:
            raise ValueError("No datacube available. Run merge_chirps_files() first.")
        
        if self.aoi is None:
            self.load_aoi()
        
        minx, miny, maxx, maxy = self.aoi.total_bounds
        print(f"\nClipping to AOI: lon [{minx:.4f}, {maxx:.4f}], lat [{miny:.4f}, {maxy:.4f}]")
        
        # Find coordinate names
        lon_dim = next((n for n in ['longitude', 'lon', 'x'] if n in datacube.dims), None)
        lat_dim = next((n for n in ['latitude', 'lat', 'y'] if n in datacube.dims), None)
        
        if not lon_dim or not lat_dim:
            raise ValueError(f"Could not identify lat/lon dimensions: {list(datacube.dims)}")
        
        lat_values = datacube[lat_dim].values
        lat_ascending = lat_values[0] < lat_values[-1]
        
        if lat_ascending:
            clipped = datacube.sel({lon_dim: slice(minx, maxx), lat_dim: slice(miny, maxy)})
        else:
            clipped = datacube.sel({lon_dim: slice(minx, maxx), lat_dim: slice(maxy, miny)})
        
        print(f"Clipped shape: {dict(clipped.sizes)}")
        self.datacube = clipped
        return clipped
    
    def process_chirps_for_aoi(self, save_path: Optional[Union[str, Path]] = None) -> xr.Dataset:
        """Complete workflow: merge CHIRPS files and clip to AOI."""
        print("="*60)
        print("CHIRPS Rainfall Processing Workflow")
        print("="*60)
        
        self.load_aoi()
        self.merge_chirps_files()
        self.clip_to_aoi()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving to: {save_path}")
            self.datacube.to_netcdf(save_path)
        
        print("\nProcessing Complete!")
        return self.datacube


def create_rainfall_datacube(chirps_dir: Union[str, Path],
                             aoi_path: Union[str, Path],
                             output_path: Optional[Union[str, Path]] = None) -> xr.Dataset:
    """Convenience function to create a rainfall datacube from CHIRPS data."""
    processor = CHIRPSProcessor(chirps_dir, aoi_path)
    return processor.process_chirps_for_aoi(save_path=output_path)


#=============================================================================
# LOAD AND CLIP CHIRPS DATA TO STUDY AREA (AOI)
#=============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Import processing modules for cubes and tensors
    from src._02_processing.cubes import GeospatialCube
    from src._02_processing.tensors import normalize_tensor, calculate_tensor_statistics
    
    # ==========================================================================
    # STEP 1: Define paths
    # ==========================================================================
    base_dir = Path(__file__).parent.parent.parent
    chirps_dir = base_dir / "data" / "raw" / "raster" / "Chirps_rainfall"
    aoi_path = base_dir / "data" / "raw" / "vector" / "AOI.shp"
    output_path = base_dir / "data" / "processed" / "chirps_clipped_aoi.nc"
    
    print("=" * 70)
    print("CHIRPS RAINFALL DATA - LOAD AND CLIP TO AOI")
    print("=" * 70)
    print(f"\nCHIRPS Directory: {chirps_dir}")
    print(f"AOI Shapefile: {aoi_path}")
    print(f"Output Path: {output_path}")
    
    # ==========================================================================
    # STEP 2: Load and clip CHIRPS data
    # ==========================================================================
    processor = CHIRPSProcessor(chirps_dir, aoi_path)
    
    # Load AOI
    aoi = processor.load_aoi()
    print(f"\nAOI Bounds: {aoi.total_bounds}")
    
    # Get CHIRPS files
    chirps_files = processor.get_chirps_files()
    print(f"\nFound {len(chirps_files)} CHIRPS files:")
    for f in chirps_files:
        print(f"  - {f.name}")
    
    # Merge all CHIRPS files into single datacube
    merged_datacube = processor.merge_chirps_files()
    print(f"\nMerged Datacube Dimensions: {dict(merged_datacube.sizes)}")

    # Clip to AOI
    clipped_datacube = processor.clip_to_aoi()
    print(f"Clipped Datacube Dimensions: {dict(clipped_datacube.sizes)}")
    
    # Delete existing file if it exists
    if output_path.exists():
        output_path.unlink()
    
    # Save to netCDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped_datacube.to_netcdf(output_path)
    print(f"\nSaved clipped datacube to: {output_path}")
    
    # ==========================================================================
    # STEP 3: Use GeospatialCube for datacube operations
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DATACUBE OPERATIONS (using cubes module)")
    print("=" * 70)
    
    # Load as GeospatialCube
    cube = GeospatialCube(clipped_datacube)
    
    # Get cube info
    print(f"\nCube Shape: {cube.get_shape()}")
    print(f"Cube Extent: {cube.get_extent()}")
    
    # Get variable name (precipitation)
    var_name = list(clipped_datacube.data_vars)[0]
    print(f"Variable: {var_name}")
    
    # Compute statistics (requires variable name)
    stats = cube.compute_statistics(var_name)
    print(f"\nRainfall Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Temporal aggregation - mean rainfall
    mean_rainfall_ds = cube.aggregate_temporal(method='mean')
    print(f"\nMean Rainfall Dataset: {list(mean_rainfall_ds.data_vars)}")
    
    # ==========================================================================
    # STEP 4: Use Tensor operations
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TENSOR OPERATIONS (using tensors module)")
    print("=" * 70)
    
    # Convert to NumPy array (tensor)
    rainfall_array = cube.to_numpy(var_name)
    print(f"\nRainfall Tensor Shape: {rainfall_array.shape}")
    print(f"  (time, latitude, longitude)")
    
    # Normalize the tensor
    normalized = normalize_tensor(rainfall_array)
    print(f"\nNormalized Tensor Range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # Calculate tensor statistics
    tensor_stats = calculate_tensor_statistics(rainfall_array)
    print(f"\nTensor Statistics:")
    print(f"  Mean: {tensor_stats['mean']:.2f} mm")
    print(f"  Std: {tensor_stats['std']:.2f} mm")
    print(f"  Min: {tensor_stats['min']:.2f} mm")
    print(f"  Max: {tensor_stats['max']:.2f} mm")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\n✓ Loaded {len(chirps_files)} CHIRPS files")
    print(f"✓ Clipped to AOI: {aoi.total_bounds}")
    print(f"✓ Created datacube: {dict(clipped_datacube.dims)}")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")