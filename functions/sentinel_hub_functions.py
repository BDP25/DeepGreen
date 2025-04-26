###########################################################################################
# Modules
###########################################################################################

import os
from pathlib import Path

import sentinelhub

from dotenv import load_dotenv
import numpy as np
import json
import math
import pandas as pd
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
    SHConfig
)
import imageio.v2 as imageio
from typing import Literal, Dict, Any

###########################################################################################
# Credentials
###########################################################################################
load_dotenv()
CLIENT_ID = os.environ.get("SENTINEL_HUB_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")

###########################################################################################
# URL's
###########################################################################################
TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
PROCESS_API_URL = "https://services.sentinel-hub.com/api/v1/process"
CATALOG_API_URL = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"


####################################################################################
# AOI Helper Functions
###########################################################################################
def segment_aoi(
    aoi: sentinelhub.BBox, resolution: int, output: Literal["grid", "flat"]
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
    if output == "grid":
        return aois_grid


def create_aoi_bbox_dict(aoi_segments: np.array) -> Dict[str, BBox]:
    """
    creates simple aoi id's and maps them to the BBox objects
    return: Dict[aoi_id, BBox]
    """
    aoi_bbox_dict = {}
    for i, row in enumerate(aoi_segments):
        for j, aoi_bbox in enumerate(row):
            aoi_id = str(i).zfill(2) + str(j).zfill(2)
            aoi_bbox_dict[aoi_id] = aoi_bbox
    return aoi_bbox_dict


###########################################################################################
# API Functions
###########################################################################################
def create_aoi_catalog_dict(
    catalog: SentinelHubCatalog,
    time_interval: tuple,
    aoi_bbox_dict: dict,
    save_file=False,
    file_name="catalog.json",
    dir_path=Path("data"),
) -> Dict[str, Dict[str, list[float]]]:
    # Dict[aoi_id, Dict[time_stamp, could_coverage]]
    aoi_catalog_dict = {}
    for aoi_id, aoi_bbox in aoi_bbox_dict.items():
        # get catalog content for sub aoi
        aoi_catalog_dict[aoi_id] = get_aoi_catalog_results(
            catalog, time_interval, aoi_bbox
        )
    if save_file:
        dir_path.mkdir(exist_ok=True)
        filepath = dir_path / file_name
        with open(filepath, "w") as file:
            json.dump(aoi_catalog_dict, file)

    return aoi_catalog_dict


def get_aoi_catalog_results(
    catalog: SentinelHubCatalog, time_interval: tuple, aoi_bbox: BBox
) -> dict[Any, list[Any]]:
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=aoi_bbox,
        time=time_interval,
        filter=f"eo:cloud_cover < 100",
        fields={
            "include": ["properties.datetime", "properties.eo:cloud_cover"],
            "exclude": ["id"],
        },
    )
    aoi_catalog_results = {}
    for result in search_iterator:
        time_stamp = result["properties"]["datetime"]
        cloud_coverage = result["properties"]["eo:cloud_cover"]
        aoi_catalog_results[time_stamp] = [cloud_coverage]
    return aoi_catalog_results


def create_aoi_df(
    aoi_catalog_dict: Dict[str, Dict[str, list[float]]], aoi_bbox_dict: Dict[str, BBox]
) -> pd.DataFrame:
    rows = []
    for aoi_id, records in aoi_catalog_dict.items():
        bbox = aoi_bbox_dict[aoi_id]
        for time_stamp, values in records.items():
            row = {
                "aoi_id": aoi_id,
                "bbox": bbox,
                "time_stamp": time_stamp,
                "cloud_coverage_api": values[0]
            }
            if len(values) > 1:
                row["cloud_coverage_calculated"] = values[1]
            if len(values) > 2:
                row["file_name"] = values[2]
            rows.append(row)

    return pd.DataFrame(rows)


###########################################################################################
# Image Download Functions
###########################################################################################
def load_eval_script(path: str) -> str:
    with open(path, "r") as f:
        eval_script = f.read()
    return eval_script


def get_img(evalscript: str, timestamp: str, bbox: BBox, resolution: int, config: SHConfig) -> np.ndarray:
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    request_image = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=timestamp,
            )
        ],
        # TODO: für .tiff Bilder MimeType.TIFF setze
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=bbox_size,
        config=config,
    )
    imgs = request_image.get_data()

    return imgs[0]


def download_img(img, filename: str, dir_path=Path("images")) -> None:
    dir_path.mkdir(exist_ok=True)
    filepath = dir_path / filename
    imageio.imwrite(filepath, img)
