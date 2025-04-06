from typing import Dict
import numpy as np
from sentinelhub import BBox, SHConfig
from functions import sentinel_hub_functions


def calculate_red_coverage(image, red_threshold=50, factor=1.2):
    """
    Calculate the proportion of red pixels in the image and visualize them.

    Parameters:
        image (ndarray): The satellite image with shape (H, W, 3).
        red_threshold (int): Minimum value of the red channel to consider a pixel as "red."
        factor (float): Minimum ratio of red to green/blue channels to consider a pixel as "red."

    Returns:
        float: Proportion of red pixels in the image.
    """
    # Split the image into its color channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Create a boolean mask for red pixels
    red_pixels = (
        (red_channel > red_threshold)
        & (red_channel > factor * green_channel)
        & (red_channel > factor * blue_channel)
    )

    # Count red pixels and calculate the proportion
    red_count = np.sum(red_pixels)
    total_pixels = image.shape[0] * image.shape[1]
    red_proportion = red_count / total_pixels

    # Create a visualization where red pixels are full-intensity red
    red_visualization = np.zeros_like(image)  # Start with a black image
    red_visualization[red_pixels] = [
        255,
        0,
        0,
    ]  # Set red pixels to full red (255, 0, 0)
    return red_proportion * 100


def calculate_cloud_coverage(
    aoi_catalog_dict: Dict[str, Dict[str, float]],
    aoi_bbox_dict: Dict[str, BBox],
    resolution: int,
    config: SHConfig,
    download_images: bool = False,
) -> dict[str, dict[str, list[float | str]]]:
    aoi_dict = {}
    eval_script = sentinel_hub_functions.load_evalscript(
        "evalscripts/es_true_color_clm.js"
    )
    for aoi_id, aoi_recs in aoi_catalog_dict.items():
        aoi_recs = aoi_catalog_dict[aoi_id]
        aoi_bbox = aoi_bbox_dict[aoi_id]
        new_recs = {}
        for time_stamp, cloud_coverage_api in aoi_recs.items():
            img = sentinel_hub_functions.get_img(
                evalscript=eval_script,
                timestamp=str(time_stamp)[:10],
                bbox=aoi_bbox,
                resolution=resolution,
                config=config,
            )
            cloud_coverage_calculated = calculate_red_coverage(
                img, red_threshold=50, factor=1.2
            )
            image_name = aoi_id + "_" + str(time_stamp)[:10] + ".png"
            if download_images:
                sentinel_hub_functions.download_img(img=img, filename=image_name)
                new_recs[time_stamp] = [
                    cloud_coverage_api,
                    cloud_coverage_calculated,
                    image_name,
                ]
            else:
                new_recs[time_stamp] = [cloud_coverage_api, cloud_coverage_calculated]
        aoi_dict[aoi_id] = new_recs
    return aoi_dict
