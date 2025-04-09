from pathlib import Path

import pandas as pd
from sentinelhub import BBox, SentinelHubCatalog, DataCollection


class AOISegment:
    def __init__(self, bbox: BBox, time_interval: tuple):
        self.bbox = bbox
        self.time_interval = time_interval
        self.df_all_images = pd.DataFrame(
            columns=["time_stamp", "cloud_coverage_api", "cloud_coverage_calculated", "file_name"])
        self.df_candidates = pd.DataFrame(
            columns=["time_stamp", "cloud_coverage_api", "cloud_coverage_calculated", "file_name"])
        self.df_land_use_evolution = pd.DataFrame(
            columns=["time_stamp", "buildings_pct", "green_space_pct", "water_pct", "rest_pct"])

    def get_cloud_coverage_api(self, catalog: SentinelHubCatalog):
        # TODO: get cc values from the api for given bbox and time_interval and fill self.df_all_images with values
        pass

    def get_cloud_coverage_calculated(self, eval_script_cc):
        # TODO: calculate real cc values from the images of given bbox and time_interval and fill self.df_all_images with values
        pass

    def extract_good_candidates(self):
        # TODO: set thresholds for cloud_coverage_api and cloud_coverage_calculated, filter self.df_all_images, fill self.df_candidates
        pass

    def calculate_buildings_pcts(self):
        # TODO: calculate and fill into self.df_land_use_evolution
        pass

    def calculate_green_space_pcts(self):
        # TODO: calculate and fill into self.df_land_use_evolution
        pass

    def calculate_water_pcts(self):
        # TODO: calculate and fill into self.df_land_use_evolution
        pass

    def calculate_rest_pcts(self):
        # TODO: calculate and fill into self.df_land_use_evolution
        pass

    def save_csv(self):
        # TODO: save self.df_land_use_evolution as csv, PRESERVE TIME INTERVAL AND BBOX IN FILE NAME
        pass

    def get_img(self, eval_script, resolution, config):
        pass

    def download_img(self, img, filename: str, dir_path=Path("images")):
        pass

    def delete_img(self, filename: str, dir_path=Path("images")):
        pass
