import os

import pytest
from dotenv import load_dotenv
from sentinelhub import BBox, SHConfig, SentinelHubCatalog, CRS
from lib.AOISegment import AOISegment


@pytest.fixture
def catalog():
    load_dotenv()
    CONFIG = SHConfig()
    CONFIG.sh_client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    CONFIG.sh_client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")
    catalog = SentinelHubCatalog(config=CONFIG)
    return catalog


@pytest.fixture
def bbox_wt():
    return BBox(
        bbox=(8.477, 47.336, 8.605, 47.417), crs=CRS.WGS84
    )