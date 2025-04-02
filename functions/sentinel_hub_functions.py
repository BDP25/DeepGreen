###########################################################################################
# Modules
###########################################################################################
import os

import sentinelhub
from dotenv import load_dotenv
import numpy as np
import requests
import datetime
import subprocess
import shlex
import json
import math
import folium
import calendar
from datetime import date, timedelta
import pandas as pd
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    BBoxSplitter,
    CustomGridSplitter,
    OsmSplitter,
    TileSplitter,
    UtmGridSplitter,
    UtmZoneSplitter,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    read_data,
    bbox_to_dimensions,
)
import imageio.v2 as imageio
from typing import Literal, Dict

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
    if resolution < 10:
        resolution = 10  # Enforce minimum resolution

    width_px, height_px = bbox_to_dimensions(bbox=aoi, resolution=resolution)
    num_rows = math.ceil(height_px / 1500)
    num_cols = math.ceil(width_px / 1500)

    x_min, y_min, x_max, y_max = list(aoi)
    lat_step = (y_max - y_min) / num_rows
    lon_step = (x_max - x_min) / num_cols

    aios_grid = np.empty((num_rows, num_cols), dtype=object)

    for row in range(num_rows):
        for col in range(num_cols):
            min_x_new = x_min + col * lon_step
            min_y_new = y_max - (row + 1) * lat_step  # Corrected Y direction
            max_x_new = min(x_max, min_x_new + lon_step)
            max_y_new = min(y_max, min_y_new + lat_step)

            aios_grid[row, col] = BBox(
                bbox=(min_x_new, min_y_new, max_x_new, max_y_new), crs=CRS.WGS84
            )

    if output == "flat":
        aois_flat = aios_grid.flatten()
        return aois_flat
    if output == "grid":
        return aios_grid


def create_aoi_bbox_map(aois: np.array) -> Dict[str, BBox]:
    aoi_bbox_map = {}
    for i, row in enumerate(aois):
        for j, aoi_bbox in enumerate(row):
            aoi_id = str(i).zfill(2) + str(j).zfill(2)
            aoi_bbox_map[aoi_id] = aoi_bbox
    return aoi_bbox_map


###########################################################################################
# API Functions
###########################################################################################
def create_aoi_catalog_dict(
    catalog: SentinelHubCatalog,
    time_interval: tuple,
    aoi_bbox_map: dict,
    save_file=False,
) -> dict:
    aoi_catalog_dict = {}
    for aoi_id in aoi_bbox_map.keys():
        aoi_id = aoi_id
        aoi_bbox = aoi_bbox_map[aoi_id]
        # get catalog content for sub aoi
        aoi_catalog_results = get_aoi_catalog_results(
            catalog, time_interval, aoi_bbox
        )
        aoi_catalog_dict[aoi_id] = aoi_catalog_results
    if save_file:
        if not os.path.exists("data"):
            os.makedirs("data")
        with open("data/AOI_CATALOG.json", "w") as file:
            json.dump(aoi_catalog_dict, file)

    return aoi_catalog_dict


def get_aoi_catalog_results(catalog: SentinelHubCatalog, time_interval: tuple, aoi_bbox: BBox):
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
    # aoi_catalog_results = list(search_iterator)
    aoi_catalog_results = {}
    for result in search_iterator:
        ts = result["properties"]["datetime"]
        cc = result["properties"]["eo:cloud_cover"]
        aoi_catalog_results[ts] = cc
    return aoi_catalog_results

def create_aoi_catalog_df(catalog: dict):
    # doesnt work for newer data
    print("creating AOI_CATALOG_DF")
    aoi_catalog_rows = []
    for AOI in catalog.keys():
        aoi_id = AOI
        aoi_recs = catalog[aoi_id]
        for AOI_REC in aoi_recs.keys():
            AOI_REC_TS = AOI_REC
            AOI_REC_CC = aoi_recs[AOI_REC]
            aoi_catalog_rows.append(
                {"AOI_ID": aoi_id, "TS": AOI_REC_TS, "CC": float(AOI_REC_CC)}
            )
            aoi_catalog_df = pd.DataFrame(aoi_catalog_rows)
            # create UID ({AOI_ID}_{TS})
            aoi_catalog_df["UID"] = (
                aoi_catalog_df["AOI_ID"] + "_" + aoi_catalog_df["TS"]
            )
            # set datatypes for columns
            aoi_catalog_df = aoi_catalog_df.astype(
                {"AOI_ID": "string", "TS": "string", "CC": "float", "UID": "string"}
            )
    print(f"> created DF has {len(aoi_catalog_df.index)} entries")
    return aoi_catalog_df
