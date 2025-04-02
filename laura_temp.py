from deepgreen import *
from functions import sentinel_hub_functions
import pandas as pd
import json
from dotenv import load_dotenv
import os
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

load_dotenv()

DATA_DIR = "data"
IMAGES_DIR = "images"


CONFIG = SHConfig()
CONFIG.sh_client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
CONFIG.sh_client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")
# Init Catalog
CATALOG = SentinelHubCatalog(config=CONFIG)

# AOI for Winterthur (estimate)
aoi_wt = BBox(bbox=(8.629496, 47.421583, 8.884906, 47.58301), crs=CRS.WGS84)

AOI_WT_segments = sentinel_hub_functions.segment_aoi(
    aoi=aoi_wt, resolution=10, output="grid"
)
aoi_bbox_map = sentinel_hub_functions.create_aoi_bbox_map(AOI_WT_segments)

TIME_INTERVAL = "2024-11-01", "2024-11-30"
MAX_CC = 20


# print(AOI_BBOX_MAP)

aoi_catalog_dict = sentinel_hub_functions.create_aoi_catalog_dict(
    CATALOG, TIME_INTERVAL, aoi_bbox_map, save_file=True
)
aoi_df = sentinel_hub_functions.create_aoi_catalog_df(aoi_catalog_dict)
AOI_CATALOG_2 = calculate_cc(aoi_catalog_dict, aoi_bbox_map, resolution=10, CONFIG=CONFIG)
AOI_DF_2 = create_aoi_df_2(AOI_CATALOG_2)
