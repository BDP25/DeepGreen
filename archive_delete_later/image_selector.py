from pathlib import Path

from archive_delete_later import cloud_coverage_functions, html_table_functions, sentinel_hub_functions
from dotenv import load_dotenv
import os
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
)

# paths
DATA_DIR = Path("../data")
IMAGES_DIR = Path("../images")

DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# credentials
load_dotenv()
CONFIG = SHConfig()
CONFIG.sh_client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
CONFIG.sh_client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")

# image download parameters
TIME_INTERVAL = ("2024-11-01", "2024-11-30")
MAX_CC = 20

# initialise catalog
catalog = SentinelHubCatalog(config=CONFIG)

# approximate area of interest (aoi) for Winterthur
aoi_wt = BBox(bbox=(8.629496, 47.421583, 8.884906, 47.58301), crs=CRS.WGS84)

# slice bbox into segments
aoi_segments_wt = sentinel_hub_functions.segment_aoi(
    aoi=aoi_wt, resolution=10, output="grid"
)
# create dict: key -> aoi_id, value: BBox
aoi_bbox_dict_wt = sentinel_hub_functions.create_aoi_bbox_dict(aoi_segments_wt)

# create dict: key -> aoi_id, value: dict(time_stamp, cloud_coverage_api)
aoi_catalog_dict_wt = sentinel_hub_functions.create_aoi_catalog_dict(
    catalog,
    TIME_INTERVAL,
    aoi_bbox_dict_wt,
)

# create dict: k:aoi_id, v: dict(k:time_stamp, v:[cloud_coverage_api, cloud_coverage_calculated, image_name])
aoi_catalog_dict_with_clouds = cloud_coverage_functions.calculate_cloud_coverage(
    aoi_catalog_dict_wt,
    aoi_bbox_dict_wt,
    resolution=10,
    config=CONFIG,
    download_images=True,
)

# create df with columns: aoi_id, bbox, time_stamp, cloud_coverage_api, cloud_coverage_calculated, (file_name)
aoi_df = sentinel_hub_functions.create_aoi_df(
    aoi_catalog_dict_with_clouds, aoi_bbox_dict_wt
)

aoi_df.to_csv(DATA_DIR / "aoi_df.csv", index=False)

# df = pd.read_csv("data/aoi_df.csv", dtype={"aoi_id": str})

# create html table and safe
html_table = html_table_functions.create_html_table(aoi_df)
html_table_functions.save_html_table(html_table, dir_path=DATA_DIR)
