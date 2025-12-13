# navigation/frontier_detector.py
"""
Frontier detection based on Yamauchi (1997): 
  "Frontiers are boundaries between known free space and unknown space."

Implementation uses morphological operations on occupancy grid (np.ndarray).
Occupancy conventions:
  -1 = unknown
   0 = free
 100 = occupied
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple

def detect_frontiers(
    occupancy_grid: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float
) -> List[Tuple[float, float]]:
    """
    Detect frontier points in the occupancy grid.

    Args:
        occupancy_grid: 2D array (height, width) with values in {-1, 0, 100}
        resolution: map resolution (m/cell)
        origin_x, origin_y: map origin in world coords

    Returns:
        List of (x, y) frontier coordinates in world frame.
    """
    grid = occupancy_grid.copy()
    free = (grid == 0)
    unknown = (grid == -1)

    # Dilate free space to find adjacent unknown cells
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    dilated_free = ndimage.binary_dilation(free, structure=structure)

    # Frontiers = dilated_free ∩ unknown
    frontier_mask = dilated_free & unknown

    # Extract coordinates
    frontier_cells = np.argwhere(frontier_mask)  # (row, col) = (y, x) in grid

    frontiers = []
    for r, c in frontier_cells:
        world_x = origin_x + (c + 0.5) * resolution
        world_y = origin_y + (r + 0.5) * resolution
        frontiers.append((world_x, world_y))

    return frontiers