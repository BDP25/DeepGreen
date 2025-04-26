import os
import numpy as np
import pandas as pd
from git import Repo
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from archive_delete_later import sentinel_hub_functions
from sentinelhub import BBox, SentinelHubCatalog, DataCollection, CRS, SHConfig, SentinelHubRequest, bbox_to_dimensions, \
    MimeType


class AOISegment:
    def __init__(self, bbox: BBox, time_interval: tuple, configuration: SHConfig):
        # variables
        self.bbox = bbox
        self.time_interval = time_interval
        self.config = configuration
        self.catalog = SentinelHubCatalog(config=self.config)

        # project root
        repo = Repo(Path(__file__).resolve(), search_parent_directories=True)
        self.project_root = Path(repo.git.rev_parse("--show-toplevel"))

        # data containers
        self.df_all_data = pd.DataFrame(
            columns=["time_stamp", "cloud_coverage_api", "cloud_coverage_calculated"])
        self.df_land_use_evolution = pd.DataFrame(
            columns=["time_stamp", "cloud_coverage_api", "cloud_coverage_calculated", "buildup_pct",
                     "green_pct", "water_pct", "rest_pct"])

        # run calculations
        self.run_calculations()

    def run_calculations(self) -> None:
        """
        Executes all calculations for cloud coverage, buildup area, green area and water area
        """
        self.get_cloud_coverage_api()
        self.calculate_area_pct("cloud_coverage_calculated")
        self.df_all_data = self.df_all_data.drop_duplicates(subset='time_stamp', keep='first')
        self.extract_good_candidates(threshold=100)
        # add "water_pct" wne implemented
        for area_type in ["buildup_pct", "green_pct", "water_pct"]:
            self.calculate_area_pct(area_type)
        self.save_csv()

    def get_img(self, eval_script: str, time_stamp: str) -> np.ndarray:
        """
        Gets and image from the bbox based on a time stamp and eval script
        """
        request_image = SentinelHubRequest(
            evalscript=eval_script,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_stamp,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=self.bbox,
            size=bbox_to_dimensions(self.bbox, resolution=10),
            config=self.config,
        )
        data = request_image.get_data()  # [0] is the image
        return data[0]

    def get_cloud_coverage_api(self) -> None:
        """
        Gets the could coverage the api supplies
        """
        search_iterator = self.catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=self.bbox,
            time=self.time_interval,
            filter=f"eo:cloud_cover < 100",
            fields={
                "include": ["properties.datetime", "properties.eo:cloud_cover"],
                "exclude": ["id"],
            },
        )
        for result in search_iterator:
            time_stamp = result["properties"]["datetime"]
            cloud_coverage_api = result["properties"]["eo:cloud_cover"]
            self.df_all_data.loc[len(self.df_all_data)] = [time_stamp, cloud_coverage_api, None]

    def extract_good_candidates(self, threshold: int = 20) -> None:
        """Selects time stamps of the bbox with suitable cloud coverage"""
        filter_mask = (self.df_all_data["cloud_coverage_api"] < threshold) & (
                self.df_all_data["cloud_coverage_calculated"] < threshold)
        self.df_land_use_evolution = self.df_all_data[filter_mask]

    def calculate_area_pct(self, area_type: Literal[
        "cloud_coverage_calculated", "buildup_pct", "green_pct", "water_pct"]) -> float:
        eval_script_map = {
            "cloud_coverage_calculated": "es_clm_binary.js",
            "buildup_pct": "es_bua_binary.js",
            "green_pct": "es_gc_binary.js",
            "water_pct": ""
        }
        script = eval_script_map[area_type]
        if area_type == "cloud_coverage_calculated":
            df = self.df_all_data
        else:
            df = self.df_land_use_evolution
        for i, row in df.iterrows():
            # get image from hub with appropriate evalscript
            img = self.get_img(
                eval_script=sentinel_hub_functions.load_eval_script(self.project_root / "evalscripts" / script),
                time_stamp=str(row["time_stamp"])[:10])

            # calculate buildup area in %
            area_pct = self.area_from_pixels(img, area_type=area_type)

            # add value to df
            if area_type == "cloud_coverage_calculated":
                self.df_all_data.at[i, area_type] = area_pct
            else:
                self.df_land_use_evolution[i, area_type] = area_pct

    def calculate_rest_pcts(self) -> None:
        """calculate area that isn't build up, water, or green"""
        for i, row in self.df_land_use_evolution.iterrows():
            self.df_land_use_evolution.at[i, "rest_pct"] = 100 - row["buildup_pct"] - row["green_pct"] - row["water_pct"]

    @staticmethod
    def area_from_pixels(image: np.array,
                         area_type: Literal["cloud_coverage_calculated", "buildup_pct", "green_pct", "water_pct"]):
        # Split the image into its color channels
        red_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        blue_channel = image[:, :, 2]

        if area_type == "cloud_coverage_calculated":  # expect r0,g0,g0
            pixel_count = ((red_channel == 0) & (green_channel == 0) & (blue_channel == 0)).sum()

        elif area_type == "buildup_pct":  # expect r255,g0,b0
            pixel_count = ((red_channel == 255) & (green_channel == 0) & (blue_channel == 0)).sum()

        elif area_type == "green_pct":  # expect r0,g255,b0
            pixel_count = ((red_channel == 0) & (green_channel == 255) & (blue_channel == 0)).sum()

        elif area_type == "water_pct":  # expect r0,g0,b255
            pixel_count = ((red_channel == 0) & (green_channel == 0) & (blue_channel == 255)).sum()

        else:
            pixel_count = 0

        # Count pixels and calculate the proportion
        total_pixels = image.shape[0] * image.shape[1]
        proportion = pixel_count / total_pixels

        return proportion * 100

    def save_csv(self):
        """saves result df as csv, bbox and time internal used for naming"""
        file_name = f"{self.time_interval[0]}_{self.time_interval[0]}_{self.bbox.min_x}_{self.bbox.min_y}_{self.bbox.max_x}_{self.bbox.max_y}.csv"
        self.df_land_use_evolution.to_csv(self.project_root / "data" / file_name)


if __name__ == "__main__":
    # credentials
    load_dotenv()
    config = SHConfig()
    config.sh_client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")

    # area and time
    test_bbox = BBox(
        bbox=(8.629496, 47.5022965, 8.757201, 47.58301), crs=CRS.WGS84
    )
    test_time_interval = ("2024-06-19", "2024-06-20")

    # example
    aoi = AOISegment(bbox=test_bbox, time_interval=test_time_interval, configuration=config)
    print(aoi.df_land_use_evolution)
