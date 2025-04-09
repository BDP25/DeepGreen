import os
import re

import numpy as np
import pytest
from dotenv import load_dotenv
from sentinelhub import CRS, BBox, SHConfig, SentinelHubCatalog

from functions.sentinel_hub_functions import *


@pytest.fixture
def time_interval():
    return ("2024-11-01", "2024-11-30")


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
    return BBox(bbox=(8.629496, 47.421583, 8.884906, 47.58301), crs=CRS.WGS84)


@pytest.fixture
def aoi_segments(bbox_wt):
    aoi_segments = segment_aoi(
        aoi=bbox_wt, resolution=10, output="grid"
    )
    return aoi_segments


@pytest.fixture
def aoi_bbox_dict(aoi_segments):
    aoi_bbox_dict = create_aoi_bbox_dict(aoi_segments)
    return aoi_bbox_dict


@pytest.fixture
def aoi_catalog_dict(catalog, time_interval, aoi_bbox_dict):
    aoi_catalog_dict = create_aoi_catalog_dict(
        catalog,
        time_interval,
        aoi_bbox_dict,
    )
    return aoi_catalog_dict


def test_segment_aoi_flat(bbox_wt):
    aoi_segments_wt = segment_aoi(
        aoi=bbox_wt, resolution=10, output="flat"
    )
    assert len(aoi_segments_wt) == 4
    assert aoi_segments_wt.shape == (4,)


def test_segment_aoi_grid(bbox_wt):
    aoi_segments_wt = segment_aoi(
        aoi=bbox_wt, resolution=10, output="grid"
    )
    assert aoi_segments_wt.shape == (2, 2)


def test_create_aoi_bbox_dict(aoi_segments):
    aoi_bbox_dict = create_aoi_bbox_dict(aoi_segments)
    assert len(aoi_bbox_dict) == 4
    assert "0000" in aoi_bbox_dict
    assert isinstance(aoi_bbox_dict["0000"], BBox)


def test_create_aoi_catalog_dict(aoi_bbox_dict, time_interval, catalog):
    aoi_catalog_dict_wt = create_aoi_catalog_dict(
        catalog,
        time_interval,
        aoi_bbox_dict,
    )
    record = aoi_catalog_dict_wt["0000"]
    keys = list(record.keys())
    values =  [item for sublist in list(record.values()) for item in sublist]

    timestamp_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
    assert "0000" in aoi_catalog_dict_wt
    assert re.match(timestamp_pattern, keys[0])  # check time stamp
    assert isinstance(values[0], float)  # check cloud coverage (api value)


def test_get_aoi_catalog_results(catalog, time_interval, aoi_segments):
    aoi_catalog_results = get_aoi_catalog_results(
        catalog, time_interval, aoi_segments[0, 0]
    )
    timestamp_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
    keys = list(aoi_catalog_results.keys())
    values = [item for sublist in list(aoi_catalog_results.values()) for item in sublist]

    assert re.match(timestamp_pattern, keys[0])  # check time stamp
    assert isinstance(values[0], float)  # check cloud coverage (api value)


def test_create_aoi_df(aoi_catalog_dict, aoi_bbox_dict):
    df = create_aoi_df(aoi_catalog_dict, aoi_bbox_dict)
    assert "aoi_id" in df.columns
    assert "bbox" in df.columns
    assert "time_stamp" in df.columns
    assert "cloud_coverage_api" in df.columns

def test_load_eval_script():
    # TODO:  da wiiter teste (line 168)
    pass
