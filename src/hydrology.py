"""
Hydrology Analysis Module

This module provides functions for hydrological analysis including
flow direction, flow accumulation, and watershed delineation.
"""

import numpy as np
from typing import Tuple, Optional


def calculate_flow_direction(dem: np.ndarray) -> np.ndarray:
    """
    Calculate flow direction using D8 algorithm.
    
    Parameters
    ----------
    dem : np.ndarray
        Depression-filled DEM
        
    Returns
    -------
    np.ndarray
        Flow direction array (D8: 1, 2, 4, 8, 16, 32, 64, 128)
    """
    rows, cols = dem.shape
    flow_dir = np.zeros_like(dem, dtype=np.int32)
    
    # D8 directions: E, SE, S, SW, W, NW, N, NE
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Compute D8 flow direction based on steepest downslope neighbor
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center = dem[r, c]
            max_drop = 0.0
            direction = 0
            for i in range(8):
                rr = r + dy[i]
                cc = c + dx[i]
                drop = center - dem[rr, cc]
                if drop > max_drop:
                    max_drop = drop
                    direction = powers[i]
            flow_dir[r, c] = direction
    
    return flow_dir


def calculate_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """
    Calculate flow accumulation from flow direction.
    
    Parameters
    ----------
    flow_dir : np.ndarray
        Flow direction array
        
    Returns
    -------
    np.ndarray
        Flow accumulation array
    """
    rows, cols = flow_dir.shape
    flow_acc = np.ones_like(flow_dir, dtype=np.float64)
    
    # Build downstream mapping and indegree for topological accumulation
    downstream_r = np.full_like(flow_dir, -1, dtype=np.int32)
    downstream_c = np.full_like(flow_dir, -1, dtype=np.int32)
    indegree = np.zeros_like(flow_dir, dtype=np.int32)

    # D8 directions: E, SE, S, SW, W, NW, N, NE
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    power_to_idx = {p: i for i, p in enumerate(powers)}

    for r in range(rows):
        for c in range(cols):
            p = flow_dir[r, c]
            if p == 0:
                continue
            i = power_to_idx.get(int(p), None)
            if i is None:
                continue
            rr = r + dy[i]
            cc = c + dx[i]
            if 0 <= rr < rows and 0 <= cc < cols:
                downstream_r[r, c] = rr
                downstream_c[r, c] = cc
                indegree[rr, cc] += 1

    # Kahn's algorithm for DAG accumulation
    queue = [(r, c) for r in range(rows) for c in range(cols) if indegree[r, c] == 0]
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        rr = downstream_r[r, c]
        cc = downstream_c[r, c]
        if rr >= 0 and cc >= 0:
            flow_acc[rr, cc] += flow_acc[r, c]
            indegree[rr, cc] -= 1
            if indegree[rr, cc] == 0:
                queue.append((rr, cc))
    
    return flow_acc


def delineate_watersheds(flow_dir: np.ndarray, pour_points: np.ndarray) -> np.ndarray:
    """
    Delineate watersheds from pour points.
    
    Parameters
    ----------
    flow_dir : np.ndarray
        Flow direction array
    pour_points : np.ndarray
        Array of pour point locations
        
    Returns
    -------
    np.ndarray
        Watershed delineation array
    """
    rows, cols = flow_dir.shape
    watersheds = np.zeros_like(flow_dir, dtype=np.int32)

    # D8 directions: E, SE, S, SW, W, NW, N, NE
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]
    power_to_idx = {p: i for i, p in enumerate(powers)}

    # Pour points may be binary or labeled; use nonzero values as IDs
    pour_ids = pour_points.astype(np.int32)

    for r in range(rows):
        for c in range(cols):
            if watersheds[r, c] != 0:
                continue
            path = []
            rr, cc = r, c
            while True:
                path.append((rr, cc))
                if pour_ids[rr, cc] != 0:
                    label = pour_ids[rr, cc]
                    break
                p = flow_dir[rr, cc]
                i = power_to_idx.get(int(p), None)
                if i is None:
                    label = 0
                    break
                nr = rr + dy[i]
                nc = cc + dx[i]
                if not (0 <= nr < rows and 0 <= nc < cols):
                    label = 0
                    break
                if watersheds[nr, nc] != 0:
                    label = watersheds[nr, nc]
                    break
                rr, cc = nr, nc

            for pr, pc in path:
                watersheds[pr, pc] = label

    return watersheds


def extract_stream_network(flow_acc: np.ndarray, threshold: float) -> np.ndarray:
    """
    Extract stream network from flow accumulation.
    
    Parameters
    ----------
    flow_acc : np.ndarray
        Flow accumulation array
    threshold : float
        Threshold for stream definition
        
    Returns
    -------
    np.ndarray
        Binary stream network (1 = stream, 0 = no stream)
    """
    streams = (flow_acc >= threshold).astype(np.int32)
    return streams


def calculate_twi(dem: np.ndarray, flow_acc: np.ndarray, slope: np.ndarray, 
                  cellsize: float) -> np.ndarray:
    """
    Calculate Topographic Wetness Index (TWI).
    
    TWI = ln(a / tan(β))
    where a is the upslope contributing area per unit contour length
    and β is the slope angle
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model
    flow_acc : np.ndarray
        Flow accumulation
    slope : np.ndarray
        Slope in degrees
    cellsize : float
        Cell size in meters
        
    Returns
    -------
    np.ndarray
        Topographic Wetness Index
    """
    # Convert flow accumulation to contributing area
    contributing_area = flow_acc * cellsize * cellsize
    
    # Convert slope to radians and calculate tan(slope)
    slope_rad = np.radians(slope)
    tan_slope = np.tan(slope_rad)
    
    # Avoid division by zero
    tan_slope = np.where(tan_slope == 0, 0.001, tan_slope)
    
    # Calculate TWI
    twi = np.log(contributing_area / (cellsize * tan_slope))
    
    return twi
