# DeepGreen: Land Cover Evolution Analysis Using Sentinel-2

Authors: Alex Leccadito and Laura Conti

## Table of Contents
<!-- TOC -->
* [DeepGreen: Land Cover Evolution Analysis Using Sentinel-2](#deepgreen-land-cover-evolution-analysis-using-sentinel-2)
  * [Table of Contents](#table-of-contents)
  * [Project Description](#project-description)
  * [Setup](#setup)
    * [Requirements](#requirements)
    * [Sentinel Hub Login](#sentinel-hub-login)
    * [Test Execution](#test-execution)
  * [Repository Structure and Files](#repository-structure-and-files)
  * [`AOISegment` Class](#aoisegment-class)
    * [Parameters](#parameters)
    * [Primary Functionality](#primary-functionality)
    * [Supporting Methods](#supporting-methods)
    * [Data Output](#data-output)
    * [Usage Example](#usage-example)
  * [Analysis and Visualisation Notebooks](#analysis-and-visualisation-notebooks)
  * [Contributors](#contributors)
<!-- TOC -->


## Project Description

Our project enables the analysis and monitoring of land cover changes within a specified Area of Interest (AOI)
using Sentinel-2 satellite imagery. It focuses on identifying and quantifying four key surface types, clouds, built-up
areas, vegetation (green areas), and water bodies over a given time interval.

The analysis is powered by the Sentinel Hub API and uses custom JavaScript-based evalscripts to extract semantic
segmentation masks. The processed data is saved locally for each AOI and timestamp, along with a CSV file that logs
temporal changes in land cover statistics.

## Setup

### Requirements

To install the project dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Sentinel Hub Login

- Sign up for a free account at [planet.com](https://www.planet.com/account/)
    - 30'000 token are included in the free trial
- Once registered, store your Client ID and Secret as environment variables:
    - `SENTINEL_HUB_CLIENT_ID`
    - `SENTINEL_HUB_CLIENT_SECRET`

You'll need this login to send catalogue requests (to check which images are available and when) and to download images
from Sentinel Hub.

### Test Execution

```
python -m pytest tests
```

## Repository Structure and Files

```txt
.
├── data/  # dir not present in repository, this is an example layout
│   └── 8.629496_47.5022965_8.757201_47.58301/  # Example: bbox für zurich
│       ├── evolution.csv # collects numerical data for every valid time stamp
│       └── YYYY-MM-DD_masks.npy  # one combined ´*.npy´-mask for every valid time stamp
│
├── eval_scripts/  # evaluation scripts for built-up, clouds, green space and water
│   ├── es_bua_binary.js 
│   ├── es_clm_binary.js
│   ├── es_gc_binary.js
│   └── es_w_binary.js 
│
├── images/  # for visualisation purposes
│   ├── Winterthur-2024-06-19.jpeg
│   └── Zurich-2025-04-23.jpeg
│
├── lib/
│   ├── __init__.py
│   ├── aoi_segmentation.py  # segments big AOI's into smaller segments with max resolution
│   └── AOISegment.py # See next chapter for full description
│
└── tests/
    ├── mock_files/  # contains several mock files for tests
    ├── test_aoi_segmentation.py  # tests for aoi_segmentation.py
    └── test_AOISegment.py  # tests for AOISegment.py
```

## `AOISegment` Class

The AOISegment class queries and saves satellite image data from
Sentinel-2 for a specified Area of Interest (AOI) and time interval. It supports segmentation of clouds, built-up areas,
vegetation (green areas), and water bodies using JavaScript evaluation scripts (directory `eval_scripts/`) executed via
Sentinel Hub services.

### Parameters

- `bbox`: A `BBox` object defining the area of interest for data collection.
- `time_interval`: A tuple of two strings representing the start and end dates for data collection (e.g.,
  `("2020-03-01", "2020-05-01")`).
- `configuration`: A `SHConfig` object containing Sentinel Hub credentials and configuration.

### Primary Functionality

`run_and_save_calculations()`

Executes the main analysis pipeline:

1. Fetches available Sentinel-2 image timestamps and cloud cover metadata.
2. Computes cloud masks and coverage using Sentinel Hub evalscripts.
3. Filters out images with high cloud coverage (default threshold: 20%).
4. For each suitable image:

- Downloads masks for clouds, built-up area, green area, and water.
- Combines masks into a single RGB-coded mask (combined_mask).
- Calculates area percentages for each land cover class.
- Saves results as .npy masks and logs statistics into a CSV file.

All data is stored in a structured folder named after the AOI’s coordinates.

### Supporting Methods

- `calculate_areas_pct(combined_mask) → tuple`
    - Computes the percentage of cloud, built-up, green, water, and empty (black) areas from a combined RGB mask.

- `combine_masks(cloud_mask, red_mask, green_mask, blue_mask) → np.ndarray`
    - Combines four binary RGB masks (clouds, built-up, vegetation, water) into a single semantic mask. Pixels are
      labeled by
      color

      White &rarr; Clouds

      Red &rarr; Built-up

      Green &rarr; Vegetation

      Blue &rarr; Water

      Black &rarr; Empty

- `load_eval_script(path: str) → str`
    - Loads a Sentinel Hub evalscript (JavaScript code) from a file and returns it as a string.

- `extract_good_candidates(threshold: int = 20)`
    - Filters out timestamps where either the API-provided or calculated cloud coverage exceeds the specified
      threshold (
      default: 20%).

- `get_img(bbox, eval_script, time_stamp, config) → np.ndarray`
    - Downloads a processed image for a specific date using a custom evalscript via the Sentinel Hub API.

- `calculate_clouds()`
    - Downloads cloud masks for each timestamp and computes cloud coverage as a percentage of the total area.

- `get_time_stamps_and_api_cloud_coverage()`
    - Queries the Sentinel Hub Catalog API for available images within the AOI and time range, recording timestamp and
      cloud
      cover metadata.

- `send_catalogue_request() → CatalogSearchIterator`
    - Sends a catalog search request to Sentinel Hub for Sentinel-2 Level 2A images within the specified AOI and date
      range.

### Data Output

- `.npy` files: Combined RGB segmentation masks (clouds, built-up, vegetation, water).
- `evolution.csv`: Contains time-series statistics for each valid timestamp:
    - `time_stamp`
    - `cloud_coverage_api`
    - `cloud_coverage_calculated`
    - `buildup_pct`
    - `green_pct`
    - `water_pct`
    - `empty_pct`

### Usage Example

```txt
from sentinelhub import BBox, CRS, SHConfig
from dotenv import load_dotenv
import os

load_dotenv()
config = SHConfig()
config.sh_client_id = os.environ["SENTINEL_HUB_CLIENT_ID"]
config.sh_client_secret = os.environ["SENTINEL_HUB_CLIENT_SECRET"]

aoi = AOISegment(
  bbox=BBox(bbox=(8.477, 47.336, 8.605, 47.417), crs=CRS.WGS84),
  time_interval=("2020-03-01", "2020-05-01"),
  configuration=config
)
aoi.run_and_save_calculations()
```

## Analysis and Visualisation Notebooks

All computations were carried out on the dedicated Ray cluster provided for this project. Due to the large size of the
generated data, results were saved locally for further processing and are not included in the repository.

The notebook `area_visualisation.ipynb` demonstrates a simple example of how segmentation masks can be generated and
visualized.

The notebook `priority_aggregation.ipynb` shows how we did the priority aggregation for a set of images where we fill
empty/invalid pixels (cloud and empty spaces) by priority: building, water body, green space.

The notebook `file_aggregation_temporal.ipynb` demonstrates how we create 2024 land-cover composites by first enforcing temporal consistency across masks and then filling any remaining empty or cloud pixels in order of priority—building, water body, green space.

The notebook `time_evolution.ipynb` allows you to create an animation from a folder containing .npy mask files. Whether
the files have been aggregated or not does not affect the method—visualization works the same either way.

## Contributors

[![Laura](https://github.com/conlalaura.png?size=100)](https://github.com/conlalaura)
[![Alex](https://github.com/leccato.png?size=100)](https://github.com/leccato)

