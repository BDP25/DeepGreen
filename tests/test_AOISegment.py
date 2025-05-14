import os
import json
import pytest
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lib.AOISegment import AOISegment
from sentinelhub import BBox, SHConfig, CRS


@pytest.fixture
def config():
    load_dotenv()
    config = SHConfig()
    config.sh_client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")
    return config


@pytest.fixture
def bbox():
    return BBox(bbox=(8.477, 47.336, 8.605, 47.417), crs=CRS.WGS84)


@pytest.fixture()
def time_interval():
    return ("2020-03-01", "2020-05-01")


@pytest.fixture
def mock_catalog_data():
    with open("tests/mock_files/catalog_response.json") as f:
        return json.load(f)


@pytest.fixture()
def aoi_segment(bbox, time_interval, config):
    # empty object before API call
    return AOISegment(bbox, time_interval, config)


@pytest.fixture
def mock_cloud_mask():
    return np.load("tests/mock_files/test_mask_cloud.npy")


@pytest.fixture
def mock_red_mask():
    return np.load("tests/mock_files/test_mask_red.npy")


@pytest.fixture
def mock_green_mask():
    return np.load("tests/mock_files/test_mask_green.npy")


@pytest.fixture
def mock_blue_mask():
    return np.load("tests/mock_files/test_mask_blue.npy")


@pytest.fixture
def mock_combined_mask():
    return np.load("tests/mock_files/test_mask_combined.npy")


def test_get_time_stamps_and_api_cloud_coverage(mocker, mock_catalog_data, aoi_segment):
    mocker.patch.object(
        aoi_segment, "send_catalogue_request", return_value=iter(mock_catalog_data)
    )

    # mocked method call
    aoi_segment.get_time_stamps_and_api_cloud_coverage()

    assert len(aoi_segment.df) == len(mock_catalog_data)
    for i, row in aoi_segment.df.iterrows():
        assert row["time_stamp"] == mock_catalog_data[i]["properties"]["datetime"]
        assert (
            row["cloud_coverage_api"]
            == mock_catalog_data[i]["properties"]["eo:cloud_cover"]
        )


def test_calculate_clouds(mocker, mock_cloud_mask, aoi_segment):
    df = pd.DataFrame(
        {
            "time_stamp": ["ts1"],
            "cloud_coverage_api": [15],
            "cloud_coverage_calculated": [None],
            "cloud_mask": [None],
            "combined_mask": [None],
        }
    )
    aoi_segment.df = df

    # Mock get_img to return the pre-saved cloud mask
    mocker.patch.object(aoi_segment, "get_img", return_value=mock_cloud_mask)

    # Act
    aoi_segment.calculate_clouds()

    # Assert
    assert aoi_segment.df.at[0, "cloud_mask"] == [mock_cloud_mask]
    assert isinstance(aoi_segment.df.at[0, "cloud_coverage_calculated"], float)
    assert 0 <= aoi_segment.df.at[0, "cloud_coverage_calculated"] <= 100


def test_extract_good_candidates(aoi_segment):
    df = pd.DataFrame(
        {
            "time_stamp": ["ts1", "ts2", "ts3", "ts4"],
            "cloud_coverage_api": [15, 90, 15, 90],
            "cloud_coverage_calculated": [15, 90, 90, 15],
            "cloud_mask": [None, None, None, None],
            "combined_mask": [None, None, None, None],
        }
    )
    # with default threshold of 20, onl first row should persist
    aoi_segment.df = df
    assert len(aoi_segment.df) == len(df)
    aoi_segment.extract_good_candidates()
    assert len(aoi_segment.df) == 1


def test_load_eval_script():
    path = "tests/mock_files/es_mock.js"
    eval_script = AOISegment.load_eval_script(path)
    assert isinstance(eval_script, str)
    assert eval_script is not None


def test_combine_masks(mock_cloud_mask, mock_red_mask, mock_green_mask, mock_blue_mask):
    combined_mask = AOISegment.combine_masks(
        mock_cloud_mask, mock_red_mask, mock_green_mask, mock_blue_mask
    )
    assert isinstance(combined_mask, np.ndarray)


def test_calculate_areas_pct(mock_combined_mask):
    pcts = AOISegment.calculate_areas_pct(mock_combined_mask)
    assert sum(pcts) == 100
