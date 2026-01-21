# Flood Exposure Assessment in the Nile Basin Region Using DEM-Derived Flow Accumulation and Drainage Network Analysis

A comprehensive geospatial analysis pipeline for flood exposure assessment in the Nile Basin Region using DEM processing, hydrological analysis, tensor operations, and data cubes.

## Features

- **DEM Processing**: Load, preprocess, and analyze Digital Elevation Models
- **Hydrology Analysis**: Flow direction, accumulation, watershed delineation, and TWI calculation
- **Flood Exposure**: Calculate flood depth, identify exposed areas, and assess population risk
- **Tensor Operations**: Multi-dimensional array processing, normalization, and ML-ready patch extraction
- **Data Cubes**: Time-series flood analysis with xarray-based data cubes
- **Visualization**: Generate maps and plots for all analysis results

## Project Structure

```
flood-exposure-geospatial-pipeline/
├── data/
│   └── raw/
│       ├── raster/         # DEM, flood depth, population data
│       └── vector/         # Administrative boundaries, features
├── src/
│   ├── __init__.py
│   ├── dem_processing.py   # DEM loading, slope, resampling
│   ├── hydrology.py        # Flow direction, accumulation, TWI
│   ├── exposure.py         # Flood depth, exposure assessment, risk mapping
│   ├── tensors.py          # Tensor operations, normalization, PCA
│   ├── cubes.py            # Data cube management with xarray
│   └── visualization.py    # Plotting functions
├── tests/
│   ├── conftest.py
│   ├── test_dem_processing.py
│   ├── test_hydrology.py
│   ├── test_exposure.py
│   ├── test_tensors.py
│   ├── test_cubes.py
│   └── test_visualization.py
├── notebooks/              # Jupyter notebooks for analysis
├── outputs/                # Analysis results and figures
├── .gitignore
├── LICENSE
├── pyproject.toml          # Poetry configuration
├── requirements.txt        # Pip dependencies
└── README.md
```

## Installation

### Using Poetry (Recommended)

```bash
poetry install
poetry shell
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
from src.dem_processing import load_dem, calculate_slope
from src.hydrology import calculate_flow_direction, calculate_twi
from src.exposure import calculate_flood_depth, identify_flooded_areas
from src.visualization import plot_raster

# Load DEM
dem, metadata = load_dem("data/raw/raster/dem.tif")
cellsize = metadata['transform'][0]

# Calculate terrain derivatives
slope = calculate_slope(dem, cellsize)

# Hydrological analysis
flow_dir = calculate_flow_direction(dem)
twi = calculate_twi(dem, flow_acc, slope, cellsize)

# Flood exposure analysis
water_level = 50.0
flood_depth = calculate_flood_depth(dem, water_level)
flood_mask = identify_flooded_areas(flood_depth, threshold=0.1)

# Visualize results
plot_raster(flood_depth, title='Flood Depth (m)', cmap='Blues')
```

## Usage

See `example.py` for a complete workflow script, and the `notebooks/` directory for detailed analysis examples.

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_exposure.py
pytest tests/test_hydrology.py
```

## Modules Overview

### DEM Processing (`dem_processing.py`)
- `load_dem()` - Load DEM raster files
- `fill_depressions()` - Fill sinks for hydrological analysis
- `calculate_slope()` - Calculate terrain slope
- `resample_dem()` - Resample DEM to different resolution

### Hydrology (`hydrology.py`)
- `calculate_flow_direction()` - D8 flow direction algorithm
- `calculate_flow_accumulation()` - Flow accumulation analysis
- `extract_stream_network()` - Extract streams from flow accumulation
- `calculate_twi()` - Topographic Wetness Index

### Exposure (`exposure.py`)
- `calculate_flood_depth()` - Calculate flood depth from DEM and water level
- `identify_flooded_areas()` - Binary flood masks
- `assess_population_exposure()` - Population exposure statistics
- `generate_risk_map()` - Risk = Hazard × Vulnerability

### Tensors (`tensors.py`)
- `create_geospatial_tensor()` - Stack multiple bands into tensor
- `normalize_tensor()` - Min-max, z-score, robust normalization
- `extract_patches()` - Extract patches for ML training

### Data Cubes (`cubes.py`)
- `GeospatialCube` - Class for managing multi-dimensional geospatial data
- `create_flood_cube()` - Create time-series flood data cubes
- `merge_cubes()` - Combine multiple data cubes

## License

MIT License
