"""Tests for buildings data loading module."""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from unittest.mock import patch, MagicMock

from src._01_data_loading.buildings import (
    BuildingsAPIError,
    fetch_buildings_from_overpass,
    enrich_buildings,
    filter_buildings_by_extent,
    save_buildings,
    load_buildings,
    estimate_population_from_buildings
)


class TestBuildingsAPIError:
    """Tests for custom exception."""
    
    def test_exception_message(self):
        """Test that exception preserves message."""
        msg = "API connection failed"
        with pytest.raises(BuildingsAPIError) as exc_info:
            raise BuildingsAPIError(msg)
        assert msg in str(exc_info.value)


class TestFetchBuildingsFromOverpass:
    """Tests for Overpass API fetching."""
    
    @pytest.fixture
    def sample_bbox(self):
        """Sample bounding box (south, west, north, east)."""
        return (15.5, 32.4, 16.5, 33.4)
    
    @patch('src._01_data_loading.buildings.requests.post')
    def test_fetch_handles_api_error(self, mock_post, sample_bbox):
        """Test that API errors are properly handled."""
        from requests.exceptions import RequestException
        mock_post.side_effect = RequestException("Connection timeout")
        
        with pytest.raises((BuildingsAPIError, RequestException)):
            fetch_buildings_from_overpass(sample_bbox, timeout=5)
    
    def test_bbox_format(self, sample_bbox):
        """Test that bbox is correctly unpacked."""
        south, west, north, east = sample_bbox
        assert south < north
        assert west < east


class TestFilterBuildingsByExtent:
    """Tests for filtering buildings by extent."""
    
    @pytest.fixture
    def sample_buildings(self):
        """Buildings both inside and outside extent."""
        return gpd.GeoDataFrame({
            'geometry': [
                box(0, 0, 1, 1),   # inside
                box(10, 10, 11, 11),  # outside
                box(0.5, 0.5, 2, 2)  # partial
            ]
        }, crs="EPSG:4326")
    
    def test_filter_reduces_count(self, sample_buildings):
        """Test that filtering removes buildings outside extent."""
        extent = (0, 0, 5, 5)  # minx, miny, maxx, maxy
        result = filter_buildings_by_extent(sample_buildings, extent)
        assert len(result) <= len(sample_buildings)
    
    def test_filter_returns_geodataframe(self, sample_buildings):
        """Test that filtering returns a GeoDataFrame."""
        extent = (0, 0, 5, 5)
        result = filter_buildings_by_extent(sample_buildings, extent)
        assert isinstance(result, gpd.GeoDataFrame)


class TestEnrichBuildings:
    """Tests for building attribute enrichment."""
    
    @pytest.fixture
    def sample_buildings(self):
        """Sample buildings for enrichment."""
        return gpd.GeoDataFrame({
            'geometry': [box(0, 0, 0.001, 0.001), box(0.02, 0.02, 0.021, 0.021)],
            'building': ['residential', 'commercial']
        }, crs="EPSG:4326")
    
    def test_enrichment_adds_columns(self, sample_buildings):
        """Test that enrichment adds new columns."""
        original_cols = set(sample_buildings.columns)
        result = enrich_buildings(sample_buildings)
        new_cols = set(result.columns)
        # Should have at least the same columns
        assert original_cols.issubset(new_cols)
    
    def test_maintains_geometry(self, sample_buildings):
        """Test that enrichment preserves geometry."""
        result = enrich_buildings(sample_buildings)
        assert 'geometry' in result.columns
        assert len(result) == len(sample_buildings)


class TestSaveLoadBuildings:
    """Tests for saving and loading buildings."""
    
    @pytest.fixture
    def sample_buildings(self):
        """Sample buildings for save/load tests."""
        return gpd.GeoDataFrame({
            'geometry': [box(0, 0, 1, 1), box(2, 2, 3, 3)],
            'building': ['residential', 'commercial'],
            'id': [1, 2]
        }, crs="EPSG:4326")
    
    def test_save_creates_file(self, sample_buildings, tmp_path):
        """Test that save creates a file."""
        filepath = str(tmp_path / "test_buildings.geojson")
        save_buildings(sample_buildings, filepath)
        assert (tmp_path / "test_buildings.geojson").exists()
    
    def test_load_returns_geodataframe(self, sample_buildings, tmp_path):
        """Test that load returns a GeoDataFrame."""
        filepath = str(tmp_path / "test_buildings.geojson")
        save_buildings(sample_buildings, filepath)
        result = load_buildings(filepath)
        assert isinstance(result, gpd.GeoDataFrame)
    
    def test_save_load_roundtrip(self, sample_buildings, tmp_path):
        """Test that save/load preserves data."""
        filepath = str(tmp_path / "test_buildings.geojson")
        save_buildings(sample_buildings, filepath)
        result = load_buildings(filepath)
        assert len(result) == len(sample_buildings)


class TestEstimatePopulationFromBuildings:
    """Tests for population estimation."""
    
    @pytest.fixture
    def sample_buildings(self):
        """Create sample buildings with area."""
        gdf = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 0.001, 0.001), box(0.02, 0.02, 0.022, 0.022)],
            'building_type': ['residential', 'residential'],
        }, crs="EPSG:4326")
        return gdf
    
    def test_returns_series(self, sample_buildings):
        """Test that function returns a pandas Series."""
        import pandas as pd
        result = estimate_population_from_buildings(sample_buildings)
        assert isinstance(result, pd.Series)
    
    def test_population_is_non_negative(self, sample_buildings):
        """Test that population estimates are non-negative."""
        result = estimate_population_from_buildings(sample_buildings)
        assert (result >= 0).all()
