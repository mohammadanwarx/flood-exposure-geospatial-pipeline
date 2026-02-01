"""Tests for flood propensity module."""

import pytest
import numpy as np

from src._03_analysis.flood_propensity import (
    compute_rainfall_indicators,
    compute_terrain_indicators,
    normalize_indicators,
    compute_flood_propensity_index
)


class TestComputeRainfallIndicators:
    """Tests for rainfall indicator computation."""
    
    @pytest.fixture
    def sample_rainfall_cube(self):
        """Create sample rainfall datacube (time, lat, lon)."""
        np.random.seed(42)
        # 24 months, 10x10 grid
        return np.random.rand(24, 10, 10) * 100
    
    def test_returns_dict_with_correct_keys(self, sample_rainfall_cube):
        """Test that function returns dict with expected keys."""
        result = compute_rainfall_indicators(sample_rainfall_cube)
        
        assert isinstance(result, dict)
        assert 'R1_mean' in result
        assert 'R2_p95' in result
        assert 'R3_cv' in result
    
    def test_output_shapes_match_spatial(self, sample_rainfall_cube):
        """Test that output shapes match spatial dimensions."""
        result = compute_rainfall_indicators(sample_rainfall_cube)
        
        expected_shape = (10, 10)  # lat, lon
        assert result['R1_mean'].shape == expected_shape
        assert result['R2_p95'].shape == expected_shape
        assert result['R3_cv'].shape == expected_shape
    
    def test_r1_mean_is_average(self, sample_rainfall_cube):
        """Test that R1 is the temporal mean."""
        result = compute_rainfall_indicators(sample_rainfall_cube)
        expected_mean = np.mean(sample_rainfall_cube, axis=0)
        
        np.testing.assert_array_almost_equal(result['R1_mean'], expected_mean)
    
    def test_r2_p95_is_95th_percentile(self, sample_rainfall_cube):
        """Test that R2 is the 95th percentile."""
        result = compute_rainfall_indicators(sample_rainfall_cube)
        expected_p95 = np.percentile(sample_rainfall_cube, 95, axis=0)
        
        np.testing.assert_array_almost_equal(result['R2_p95'], expected_p95)
    
    def test_handles_zero_rainfall(self):
        """Test that zero rainfall areas don't cause division errors."""
        rainfall = np.zeros((12, 5, 5))
        result = compute_rainfall_indicators(rainfall)
        
        assert np.all(np.isfinite(result['R3_cv']))


class TestComputeTerrainIndicators:
    """Tests for terrain indicator computation."""
    
    @pytest.fixture
    def sample_dem(self):
        """Create sample DEM."""
        np.random.seed(42)
        return np.random.rand(10, 10) * 500 + 300  # 300-800m elevation
    
    @pytest.fixture
    def sample_slope(self):
        """Create sample slope array (degrees)."""
        np.random.seed(42)
        return np.random.rand(10, 10) * 30  # 0-30 degrees
    
    @pytest.fixture
    def sample_flow_acc(self):
        """Create sample flow accumulation."""
        np.random.seed(42)
        return np.random.rand(10, 10) * 1000 + 1
    
    def test_returns_dict_with_correct_keys(self, sample_dem, sample_slope, sample_flow_acc):
        """Test that function returns dict with expected keys."""
        result = compute_terrain_indicators(sample_dem, sample_slope, sample_flow_acc)
        
        assert isinstance(result, dict)
        assert 'T1_flow_acc' in result
        assert 'T2_slope_inv' in result
    
    def test_output_shapes_match_input(self, sample_dem, sample_slope, sample_flow_acc):
        """Test that output shapes match input dimensions."""
        result = compute_terrain_indicators(sample_dem, sample_slope, sample_flow_acc)
        
        assert result['T1_flow_acc'].shape == sample_dem.shape
        assert result['T2_slope_inv'].shape == sample_dem.shape


class TestNormalizeIndicators:
    """Tests for indicator normalization."""
    
    @pytest.fixture
    def sample_rainfall_indicators(self):
        """Create sample rainfall indicators."""
        np.random.seed(42)
        return {
            'R1_mean': np.random.rand(10, 10) * 100,
            'R2_p95': np.random.rand(10, 10) * 150,
            'R3_cv': np.random.rand(10, 10) * 0.5
        }
    
    @pytest.fixture
    def sample_terrain_indicators(self):
        """Create sample terrain indicators."""
        np.random.seed(42)
        return {
            'T1_flow_acc': np.random.rand(10, 10) * 1000,
            'T2_slope_inv': np.random.rand(10, 10) * 10
        }
    
    def test_returns_two_dicts(self, sample_rainfall_indicators, sample_terrain_indicators):
        """Test that function returns two dictionaries."""
        R_norm, T_norm = normalize_indicators(
            sample_rainfall_indicators, sample_terrain_indicators
        )
        
        assert isinstance(R_norm, dict)
        assert isinstance(T_norm, dict)
    
    def test_output_keys(self, sample_rainfall_indicators, sample_terrain_indicators):
        """Test that output has correct keys."""
        R_norm, T_norm = normalize_indicators(
            sample_rainfall_indicators, sample_terrain_indicators
        )
        
        assert 'R1_norm' in R_norm
        assert 'R2_norm' in R_norm
        assert 'R3_norm' in R_norm
        assert 'T1_norm' in T_norm
        assert 'T2_norm' in T_norm
    
    def test_normalized_range(self, sample_rainfall_indicators, sample_terrain_indicators):
        """Test that normalized values are mostly in [0, 1]."""
        R_norm, T_norm = normalize_indicators(
            sample_rainfall_indicators, sample_terrain_indicators
        )
        
        for key, arr in R_norm.items():
            # Allow some values slightly outside due to percentile clipping
            assert np.nanmin(arr) >= -0.1
            assert np.nanmax(arr) <= 1.1


class TestComputeFloodPropensityIndex:
    """Tests for FPI computation."""
    
    @pytest.fixture
    def sample_normalized_inputs(self):
        """Create sample normalized inputs for FPI computation."""
        np.random.seed(42)
        shape = (10, 10)
        R_norm = {
            'R1_norm': np.random.rand(*shape),
            'R2_norm': np.random.rand(*shape),
            'R3_norm': np.random.rand(*shape)
        }
        T_norm = {
            'T1_norm': np.random.rand(*shape),
            'T2_norm': np.random.rand(*shape)
        }
        return R_norm, T_norm
    
    def test_fpi_range_0_to_1(self, sample_normalized_inputs):
        """Test that FPI is in [0, 1] range."""
        R_norm, T_norm = sample_normalized_inputs
        fpi = compute_flood_propensity_index(R_norm, T_norm)
        
        assert np.all(fpi >= 0)
        assert np.all(fpi <= 1)
    
    def test_fpi_shape_matches_input(self, sample_normalized_inputs):
        """Test that FPI shape matches input dimensions."""
        R_norm, T_norm = sample_normalized_inputs
        fpi = compute_flood_propensity_index(R_norm, T_norm)
        
        assert fpi.shape == (10, 10)
    
    def test_fpi_no_nan_values(self, sample_normalized_inputs):
        """Test that FPI has no NaN values."""
        R_norm, T_norm = sample_normalized_inputs
        fpi = compute_flood_propensity_index(R_norm, T_norm)
        
        assert np.all(np.isfinite(fpi))
    
    def test_custom_weights(self, sample_normalized_inputs):
        """Test that custom weights can be applied."""
        R_norm, T_norm = sample_normalized_inputs
        weights = np.array([0.3, 0.3, 0.1, 0.2, 0.1])
        
        fpi = compute_flood_propensity_index(R_norm, T_norm, weights=weights)
        assert np.all(fpi >= 0)
        assert np.all(fpi <= 1)


class TestFPIIntegration:
    """Integration tests for FPI pipeline."""
    
    def test_full_pipeline(self):
        """Test complete FPI computation pipeline."""
        np.random.seed(42)
        
        # Create realistic-looking data
        rainfall = np.random.rand(24, 20, 20) * 100
        dem = np.random.rand(20, 20) * 500 + 300
        slope = np.random.rand(20, 20) * 30 + 0.1  # Avoid zero slope
        flow_acc = np.random.rand(20, 20) * 1000 + 1
        
        # Compute indicators
        rain_ind = compute_rainfall_indicators(rainfall)
        terrain_ind = compute_terrain_indicators(dem, slope, flow_acc)
        
        # Verify all indicators computed
        assert len(rain_ind) == 3
        assert len(terrain_ind) == 2
        
        # Normalize indicators
        R_norm, T_norm = normalize_indicators(rain_ind, terrain_ind)
        
        # Compute FPI
        fpi = compute_flood_propensity_index(R_norm, T_norm)
        
        # Verify final output
        assert fpi.shape == (20, 20)
        assert 0 <= np.nanmean(fpi) <= 1
