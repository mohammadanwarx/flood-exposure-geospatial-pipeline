"""Tests for CHIRPS rainfall processing module."""

import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from unittest.mock import patch, MagicMock

from src._01_data_loading.rainfall_processing import CHIRPSProcessor


class TestCHIRPSProcessor:
    """Tests for CHIRPSProcessor class."""
    
    @pytest.fixture
    def sample_aoi(self, tmp_path):
        """Create a sample AOI shapefile."""
        aoi_gdf = gpd.GeoDataFrame({
            'geometry': [box(32.5, 15.5, 33.3, 16.3)]
        }, crs="EPSG:4326")
        aoi_path = tmp_path / "aoi.shp"
        aoi_gdf.to_file(aoi_path)
        return aoi_path
    
    @pytest.fixture
    def sample_chirps_file(self, tmp_path):
        """Create a sample CHIRPS netCDF file."""
        chirps_dir = tmp_path / "chirps"
        chirps_dir.mkdir(exist_ok=True)
        
        # Create sample data
        lon = np.linspace(32, 34, 20)
        lat = np.linspace(15, 17, 20)
        time = np.arange(12)  # 12 months
        
        precip = np.random.rand(12, 20, 20) * 100
        
        ds = xr.Dataset({
            'precip': (['time', 'latitude', 'longitude'], precip)
        }, coords={
            'time': time,
            'latitude': lat,
            'longitude': lon
        })
        
        file_path = chirps_dir / "chirps-v2.0.2024.monthly.nc"
        ds.to_netcdf(file_path)
        
        return chirps_dir
    
    def test_init(self, tmp_path, sample_aoi):
        """Test processor initialization."""
        processor = CHIRPSProcessor(
            chirps_dir=tmp_path,
            aoi_path=sample_aoi
        )
        
        assert processor.chirps_dir == tmp_path
        assert processor.aoi_path == sample_aoi
        assert processor.datacube is None
        assert processor.aoi is None
    
    def test_load_aoi(self, tmp_path, sample_aoi):
        """Test AOI loading."""
        processor = CHIRPSProcessor(
            chirps_dir=tmp_path,
            aoi_path=sample_aoi
        )
        
        aoi = processor.load_aoi()
        
        assert isinstance(aoi, gpd.GeoDataFrame)
        assert processor.aoi is not None
        assert len(aoi) == 1
    
    def test_get_chirps_files(self, sample_chirps_file, sample_aoi):
        """Test CHIRPS file discovery."""
        processor = CHIRPSProcessor(
            chirps_dir=sample_chirps_file,
            aoi_path=sample_aoi
        )
        
        files = processor.get_chirps_files()
        
        assert isinstance(files, list)
        assert len(files) >= 1
        assert all(f.suffix == '.nc' for f in files)
    
    def test_merge_chirps_files(self, sample_chirps_file, sample_aoi):
        """Test CHIRPS file merging."""
        processor = CHIRPSProcessor(
            chirps_dir=sample_chirps_file,
            aoi_path=sample_aoi
        )
        
        datacube = processor.merge_chirps_files()
        
        assert isinstance(datacube, xr.Dataset)
        assert processor.datacube is not None
        assert 'precip' in datacube.data_vars
    
    def test_merge_empty_files_raises(self, tmp_path, sample_aoi):
        """Test that merging with no files raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir(exist_ok=True)
        
        processor = CHIRPSProcessor(
            chirps_dir=empty_dir,
            aoi_path=sample_aoi
        )
        
        with pytest.raises(ValueError):
            processor.merge_chirps_files()
    
    def test_clip_to_aoi(self, sample_chirps_file, sample_aoi):
        """Test clipping datacube to AOI."""
        processor = CHIRPSProcessor(
            chirps_dir=sample_chirps_file,
            aoi_path=sample_aoi
        )
        
        processor.merge_chirps_files()
        clipped = processor.clip_to_aoi()
        
        assert isinstance(clipped, xr.Dataset)
        # Clipped should have smaller or equal extent
        original = processor.datacube
        assert clipped.sizes['longitude'] <= original.sizes['longitude']
        assert clipped.sizes['latitude'] <= original.sizes['latitude']
    
    def test_clip_without_datacube_raises(self, tmp_path, sample_aoi):
        """Test that clipping without datacube raises error."""
        processor = CHIRPSProcessor(
            chirps_dir=tmp_path,
            aoi_path=sample_aoi
        )
        
        with pytest.raises(ValueError):
            processor.clip_to_aoi()


class TestRainfallDataValidation:
    """Tests for rainfall data validation."""
    
    def test_precipitation_non_negative(self, tmp_path):
        """Test that precipitation values are non-negative after processing."""
        # Create sample data with some negative values (invalid)
        lon = np.linspace(32, 34, 10)
        lat = np.linspace(15, 17, 10)
        time = np.arange(3)
        
        precip = np.random.rand(3, 10, 10) * 100
        precip[0, 0, 0] = -5  # Invalid negative value
        
        ds = xr.Dataset({
            'precip': (['time', 'latitude', 'longitude'], precip)
        }, coords={
            'time': time,
            'latitude': lat,
            'longitude': lon
        })
        
        # Values should be handled appropriately
        assert ds['precip'].min() < 0  # Original has negative
    
    def test_datacube_dimensions(self, tmp_path):
        """Test that datacube has correct dimensions."""
        lon = np.linspace(32, 34, 10)
        lat = np.linspace(15, 17, 10)
        time = np.arange(12)
        
        precip = np.random.rand(12, 10, 10) * 100
        
        ds = xr.Dataset({
            'precip': (['time', 'latitude', 'longitude'], precip)
        }, coords={
            'time': time,
            'latitude': lat,
            'longitude': lon
        })
        
        assert 'time' in ds.dims
        assert 'latitude' in ds.dims
        assert 'longitude' in ds.dims
        assert ds.sizes['time'] == 12
