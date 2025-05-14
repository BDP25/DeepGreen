import math
import numpy as np
from typing import Literal
from sentinelhub import bbox_to_dimensions, CRS, BBox


def segment_aoi(
    aoi: BBox, resolution: int = 10, output: Literal["grid", "flat"] = "grid"
) -> np.array:
    """
    segments the given BBox into chunks based on the passed resolution
    return: array of smaller/segmented BBox objects
    """
    resolution = max(resolution, 10)

    width_px, height_px = bbox_to_dimensions(bbox=aoi, resolution=resolution)
    num_rows = math.ceil(height_px / 1500)
    num_cols = math.ceil(width_px / 1500)

    x_min, y_min, x_max, y_max = list(aoi)
    lat_step = (y_max - y_min) / num_rows
    lon_step = (x_max - x_min) / num_cols

    aois_grid = np.empty((num_rows, num_cols), dtype=object)

    for row in range(num_rows):
        for col in range(num_cols):
            min_x_new = x_min + col * lon_step
            min_y_new = y_max - (row + 1) * lat_step  # Corrected Y direction
            max_x_new = min(x_max, min_x_new + lon_step)
            max_y_new = min(y_max, min_y_new + lat_step)

            aois_grid[row, col] = BBox(
                bbox=(min_x_new, min_y_new, max_x_new, max_y_new), crs=CRS.WGS84
            )

    if output == "flat":
        aois_flat = aois_grid.flatten()
        return aois_flat
    elif output == "grid":
        return aois_grid
    else:
        return None


if __name__ == "__main__":
    # Example Usage
    bbox = BBox(bbox=(8.477, 47.336, 8.605, 48.417), crs=CRS.WGS84)
    segments = segment_aoi(bbox, output="grid")
    print(segments.ndim)
