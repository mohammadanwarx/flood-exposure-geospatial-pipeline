# Source Code Documentation

## Project Structure

```
src/
├── __init__.py
├── _01_data_loading/       # 1. DATA LOADING
│   ├── __init__.py
│   ├── buildings.py        # Building footprints loading
│   └── rainfall_processing.py  # CHIRPS rainfall processing
├── _02_processing/         # 2. PROCESSING  
│   ├── __init__.py
│   ├── cubes.py            # Datacube operations (xarray)
│   ├── tensors.py          # Tensor operations (NumPy/PyTorch)
│   └── dem_processing.py   # DEM processing
└── _03_analysis/           # 3. ANALYSIS
    ├── __init__.py
    ├── hydrology.py        # Flow direction, accumulation
    └── exposure.py         # Flood exposure calculations
```

## Usage

```python
from src import data_loading, processing, analysis

# DATA LOADING
from src.data_loading import CHIRPSProcessor, create_rainfall_datacube

# PROCESSING
from src.processing import GeospatialCube, normalize_tensor, fill_depressions

# ANALYSIS
from src.analysis import calculate_flow_direction, calculate_flood_depth
```

## Modules

### data_loading
- `CHIRPSProcessor` - Process CHIRPS rainfall data
- `create_rainfall_datacube()` - Create rainfall datacube from CHIRPS
- `load_buildings_from_csv()` - Load building footprints
- `filter_buildings_by_aoi()` - Filter buildings to AOI

### processing
- `GeospatialCube` - Datacube class with xarray operations
- `normalize_tensor()` - Normalize arrays (min-max, z-score)
- `fill_depressions()` - Fill DEM depressions
- `calculate_slope()` - Calculate slope from DEM

### analysis
- `calculate_flow_direction()` - D8 flow direction
- `calculate_flow_accumulation()` - Flow accumulation
- `calculate_flood_depth()` - Flood depth calculation
- `identify_flooded_areas()` - Binary flood mask
- `generate_risk_map()` - Risk mapping
