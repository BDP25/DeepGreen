import os
import numpy as np
import pandas as pd
from git import Repo
from pathlib import Path
from dotenv import load_dotenv
from sentinelhub.api.catalog import CatalogSearchIterator
from sentinelhub import (
    BBox,
    SentinelHubCatalog,
    DataCollection,
    CRS,
    SHConfig,
    SentinelHubRequest,
    bbox_to_dimensions,
    MimeType,
)


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

        # eval scripts
        self.eval_script_cloud = self.load_eval_script(
            str(self.project_root / "eval_scripts" / "es_clm_binary.js")
        )
        self.eval_script_buildup = self.load_eval_script(
            str(self.project_root / "eval_scripts" / "es_bua_binary.js")
        )
        self.eval_script_green = self.load_eval_script(
            str(self.project_root / "eval_scripts" / "es_gc_binary.js")
        )
        self.eval_script_water = self.load_eval_script(
            str(self.project_root / "eval_scripts" / "es_w_binary.js")
        )

        # data containers for statistical values
        self.df = pd.DataFrame(
            columns=[
                "time_stamp",
                "cloud_coverage_api",
                "cloud_coverage_calculated",
                "cloud_mask",
                "combined_mask",
            ]
        )

    def run_and_save_calculations(self) -> None:
        """
        Executes all calculations for cloud coverage, buildup area, green area and water area
        Saves results in data directory (folder named by bbox coordinates)
        """
        # prepare save folder
        bbox_name = (
            f"{self.bbox.min_x}_{self.bbox.min_y}_{self.bbox.max_x}_{self.bbox.max_y}"
        )
        save_dir = self.project_root / "data" / bbox_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # prepare evolution tracking
        evolution_csv_path = save_dir / "evolution.csv"
        if not evolution_csv_path.exists():
            with open(evolution_csv_path, "w") as f:
                f.write(
                    "time_stamp,cloud_coverage_api,cloud_coverage_calculated,buildup_pct,green_pct,water_pct,empty_pct\n"
                )

        # get available time stamps and cloud coverage (api value)
        self.get_time_stamps_and_api_cloud_coverage()
        self.df = self.df.drop_duplicates(
            subset="time_stamp", keep="first"
        )

        # calculate could coverage based on could mask script
        self.calculate_clouds()

        # only use images with low cloud coverage in api and calculated value (default < 20 %)
        self.extract_good_candidates()
        for i, row in self.df.iterrows():
            timestamp_str = str(row["time_stamp"])[:10]
            npy_save_path = save_dir / f"{timestamp_str}_mask.npy"
            if npy_save_path.exists():
                print(f"Skipping {timestamp_str} - data already exists.")
                continue

            # get masks
            cloud_mask = np.array(row["cloud_mask"])[0]
            buildup_mask = self.get_img(bbox=self.bbox,
                                        eval_script=self.eval_script_buildup, time_stamp=timestamp_str,
                                        config=self.config
                                        )
            green_mask = self.get_img(bbox=self.bbox,
                                      eval_script=self.eval_script_green, time_stamp=timestamp_str, config=self.config
                                      )
            water_mask = self.get_img(bbox=self.bbox,
                                      eval_script=self.eval_script_water, time_stamp=timestamp_str, config=self.config
                                      )

            # combine masks
            combined_mask = self.combine_masks(
                cloud_mask, buildup_mask, green_mask, water_mask
            )

            # save full mask in df
            self.df.at[i, "combined_mask"] = [combined_mask]

            # calculate area % of masks
            cloud_pct, buildup_pct, green_pct, water_pct, empty_pct = (
                self.calculate_areas_pct(combined_mask)
            )

            # save evolution row for time stamp
            with open(evolution_csv_path, "a") as f:
                f.write(
                    f"{row["time_stamp"]},{row["cloud_coverage_api"]},{cloud_pct},{buildup_pct},{green_pct},{water_pct},{empty_pct}\n"
                )

            # save combined mask as .npy
            np.save(npy_save_path, combined_mask)

    @staticmethod
    def calculate_areas_pct(combined_mask) -> tuple[float, float, float, float, float]:
        """
        calculates % of clouds, built up, green space, water and empty areas
        returns a tuple of % values
        """
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]

        # white pixels
        cloud_pct = (
                np.sum(np.all(combined_mask == [255, 255, 255], axis=-1))
                / total_pixels
                * 100
        )

        # red pixels
        buildup_pct = (
                np.sum(np.all(combined_mask == [255, 0, 0], axis=-1)) / total_pixels * 100
        )

        # green pixels
        green_pct = (
                np.sum(np.all(combined_mask == [0, 255, 0], axis=-1)) / total_pixels * 100
        )

        # blue pixels
        water_pct = (
                np.sum(np.all(combined_mask == [0, 0, 255], axis=-1)) / total_pixels * 100
        )

        # black pixels
        empty_pct = (
                np.sum(np.all(combined_mask == [0, 0, 0], axis=-1)) / total_pixels * 100
        )

        return cloud_pct, buildup_pct, green_pct, water_pct, empty_pct

    @staticmethod
    def combine_masks(
            cloud_mask: np.array,
            red_mask: np.array,
            green_mask: np.array,
            blue_mask: np.array,
    ) -> np.array:
        """
        Takes clouds, built up, green space, water masks and combines them into one np array
        Pixel without assignment will be set as empty (black [0, 0, 0])
        """
        h, w, _ = cloud_mask.shape
        combined = np.full((h, w, 3), np.nan, dtype=float)  # Init with NaNs

        # clouds (white)
        cloud_free_mask = (
                (cloud_mask[..., 0] == 0)
                & (cloud_mask[..., 1] == 0)
                & (cloud_mask[..., 2] == 0)
        )
        combined[cloud_free_mask] = [255, 255, 255]

        # buildings (red)
        nan_pixels = np.isnan(combined[..., 0])
        red_pixels = (
                (red_mask[..., 0] == 255)
                & (red_mask[..., 1] == 0)
                & (red_mask[..., 2] == 0)
        )
        red_selection = nan_pixels & red_pixels
        combined[red_selection] = [255, 0, 0]

        # water (blue)
        nan_pixels = np.isnan(combined[..., 0])
        blue_mask_pixels = (
                (blue_mask[..., 0] == 0)
                & (blue_mask[..., 1] == 0)
                & (blue_mask[..., 2] == 255)
        )
        blue_selection = nan_pixels & blue_mask_pixels
        combined[blue_selection] = [0, 0, 255]

        # green areas (green)
        nan_pixels = np.isnan(combined[..., 0])
        green_mask_pixels = (
                (green_mask[..., 0] == 0)
                & (green_mask[..., 1] == 255)
                & (green_mask[..., 2] == 0)
        )
        green_selection = nan_pixels & green_mask_pixels
        combined[green_selection] = [0, 255, 0]

        # remaining (black)
        nan_pixels = np.isnan(combined[..., 0])
        combined[nan_pixels] = [0, 0, 0]

        return combined.astype(np.uint8)

    @staticmethod
    def load_eval_script(path: str) -> str:
        """
        takes apath to an evaluation script (*.js) and restudies it as string
        """
        with open(path, "r") as f:
            eval_script = f.read()
        return eval_script

    def extract_good_candidates(self, threshold: int = 20) -> None:
        """Selects time stamps with suitable cloud coverage, drop others"""
        filter_mask = (self.df["cloud_coverage_api"] < threshold) & (
                self.df["cloud_coverage_calculated"] < threshold
        )
        self.df = self.df[filter_mask]

    @staticmethod
    def get_img(bbox: BBox, eval_script: str, time_stamp: str, config: SHConfig) -> np.ndarray:
        """
        Gets an image from the bbox based on a time stamp and eval script
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
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution=10),
            config=config,
        )
        data = request_image.get_data()  # [0] is the image
        return data[0]

    def calculate_clouds(self) -> None:
        """
        calculates cloud coverage based on the sentinel's cloud mask (this is not the same as the api value)
        """
        for i, row in self.df.iterrows():
            # get image from hub and save in df
            cloud_mask = self.get_img(bbox=self.bbox,
                                      eval_script=self.eval_script_cloud,
                                      time_stamp=str(row["time_stamp"])[:10], config=self.config
                                      )

            self.df.at[i, "cloud_mask"] = [cloud_mask]

            # calculate cloud area in %
            red_channel = cloud_mask[:, :, 0]
            green_channel = cloud_mask[:, :, 1]
            blue_channel = cloud_mask[:, :, 2]

            pixel_count = np.sum(
                (red_channel == 0) & (green_channel == 0) & (blue_channel == 0)
            )
            total_pixels = cloud_mask.shape[0] * cloud_mask.shape[1]
            cloud_pct = pixel_count / total_pixels * 100

            self.df.at[i, "cloud_coverage_calculated"] = cloud_pct

    def get_time_stamps_and_api_cloud_coverage(self) -> None:
        """
        Gets time_stamps of available images and cloud coverage the api supplies
        """
        search_iterator = self.send_catalogue_request()
        for result in search_iterator:
            time_stamp = result["properties"]["datetime"]
            cloud_coverage_api = result["properties"]["eo:cloud_cover"]
            self.df.loc[len(self.df)] = [
                time_stamp,
                cloud_coverage_api,
                None,  # placeholder for "cloud_coverage_calculated"
                None,  # placeholder for "cloud_mask"
                None,  # placeholder for "combined_mask"
            ]

    def send_catalogue_request(self) -> CatalogSearchIterator:
        """
        sends a request to the SentinelHub catalog for the desired bbox and time interval.
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
        return search_iterator


if __name__ == "__main__":
    # Example Usage
    # credentials
    load_dotenv()
    config = SHConfig()
    config.sh_client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    config.sh_client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")

    # area and time
    test_bbox = BBox(bbox=(8.477, 47.336, 8.605, 47.417), crs=CRS.WGS84)
    test_time_interval = ("2020-03-01", "2020-05-01")

    # example
    aoi = AOISegment(
        bbox=test_bbox, time_interval=test_time_interval, configuration=config
    )
    aoi.run_and_save_calculations()
