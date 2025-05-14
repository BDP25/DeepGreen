import pytest
from sentinelhub import BBox, CRS

from lib.aoi_segmentation import segment_aoi


@pytest.fixture()
def bbox():
    return BBox(bbox=(8.477, 47.336, 8.605, 47.417), crs=CRS.WGS84)


def test_segment_aoi_grid(bbox):
    segments = segment_aoi(bbox, output="grid")
    assert segments.ndim == 2

def test_segment_aoi_flat(bbox):
    segments = segment_aoi(bbox, output="flat")
    assert segments.ndim == 1

