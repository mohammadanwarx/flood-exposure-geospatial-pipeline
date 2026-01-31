"""
Flood Exposure Assessment in the Nile Basin Region
===================================================

A modular geospatial pipeline organized into 3 main categories:

src/
├── 01_data_loading/  # Load data (buildings, rainfall, vectors)
│   ├── buildings.py
│   └── rainfall_processing.py
├── 02_processing/    # Process data (cubes, tensors, DEM)
│   ├── cubes.py
│   ├── tensors.py
│   └── dem_processing.py
├── 03_analysis/      # Analyze data (hydrology, exposure)
│   ├── hydrology.py
│   └── exposure.py
"""

__version__ = "0.3.0"
__author__ = "Your Name"

# Import the 3 main subpackages (in logical order)
from . import _01_data_loading as data_loading
from . import _02_processing as processing
from . import _03_analysis as analysis

__all__ = [
    "data_loading",
    "processing",
    "analysis",
]
