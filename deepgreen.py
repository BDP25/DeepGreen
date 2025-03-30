###########################################################################################
# Modules
###########################################################################################
import os
from dotenv import load_dotenv
import numpy as np
import requests
import datetime
import subprocess
import shlex
import json
import math
import folium
import calendar
from datetime import date, timedelta
import pandas as pd
from sentinelhub.api.catalog import SentinelHubCatalog
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    BBoxSplitter,
    CustomGridSplitter,
    OsmSplitter,
    TileSplitter,
    UtmGridSplitter,
    UtmZoneSplitter,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    read_data,
    bbox_to_dimensions,
)
import imageio.v2 as imageio


###########################################################################################
# Credentials
###########################################################################################
load_dotenv()

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")


###########################################################################################
# URL's
###########################################################################################
TOKEN_URL = "https://services.sentinel-hub.com/oauth/token"
PROCESS_API_URL = "https://services.sentinel-hub.com/api/v1/process"
CATALOG_API_URL = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"

####################################################################################
# AOI Helper Functions 
###########################################################################################

def segment_aoi(aoi, resolution, output):
    if resolution < 10:
        resolution = 10  # Enforce minimum resolution

    width_px, height_px = bbox_to_dimensions(bbox = aoi, resolution = resolution)
    print(f"AOI has Dimensions: {height_px}px x {width_px}px (h x w)")

    num_rows = math.ceil(height_px / 1500)
    num_cols = math.ceil(width_px / 1500)

    print(f"AOI has been split into Grid with Dimensions: {num_rows} x {num_cols} (rows x cols)")

    x_min, y_min, x_max, y_max = list(aoi)
    lat_step = (y_max - y_min) / num_rows
    lon_step = (x_max - x_min) / num_cols

    AOIs_flat = []  # Flat list of sub-AOIs
    AOIs_grid = []  # 2D list of sub-AOIs (grid)

    for row in range(num_rows):
        row_AOIs = []
        for col in range(num_cols):
            min_x_new = x_min + col * lon_step
            #min_y_new = y_min + row * lat_step
            min_y_new = y_max - (row + 1) * lat_step
            max_x_new = min(x_max, min_x_new + lon_step)
            max_y_new = min(y_max, min_y_new + lat_step)
            row_AOIs.append(BBox(bbox = (min_x_new, min_y_new, max_x_new, max_y_new), crs = CRS.WGS84))
            AOIs_flat.append(BBox(bbox = (min_x_new, min_y_new, max_x_new, max_y_new), crs = CRS.WGS84))
        AOIs_grid.append(row_AOIs)

    if output == "flat":
        return AOIs_flat
    elif output =="grid":
        return AOIs_grid
    else:
        return AOIs_flat, AOIs_grid

def create_aoi_bbox_map(aois):
    aoi_bbox_map = {}
    for i, row in enumerate(aois):
        for j, aoi_bbox in enumerate(row):
            aoi_id = str(i).zfill(2)+str(j).zfill(2)
            aoi_bbox_map[aoi_id] = aoi_bbox
    return aoi_bbox_map

#TODO: adapt function to new BBox Type (vorher Liste mit 4 Koordinaten, neu Typ BBox aus Sentinel Lib)
def display_aois(AOIs, AOI, map_size=(800, 600)):
    """
    Displays multiple bounding boxes on an interactive map using Folium.
    """

    map_center_lon = (AOI[0] + AOI[2]) / 2
    map_center_lat = (AOI[1] + AOI[3]) / 2

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=10, width=map_size[0], height=map_size[1])
    x_min, y_min, x_max, y_max = AOI
    folium.Rectangle(
            bounds=[(y_min, x_min), (y_max, x_max)],
            fill=True,
            fill_color = "#e6e6e6",
            fill_opacity = 0.5,
            color="red",
            weight = 2,
            popup=f"AOI"
        ).add_to(m)
    
    if isinstance(AOIs[0][0], list):
        for i, col in enumerate(AOIs):
            for j, sub_AOI in enumerate(col):
                x_min, y_min, x_max, y_max = sub_AOI
                folium.Rectangle(
                    bounds=[(y_min, x_min), (y_max, x_max)],
                    fill = True,
                    fill_color= "blue",
                    fill_opacity = 0.2,
                    color = "blue",
                    weight = 0.5,
                    popup=f"AOI [{i},{j}]"
                ).add_to(m)

    else:
        for i, sub_AOI in enumerate(AOIs):
            x_min, y_min, x_max, y_max = sub_AOI
            folium.Rectangle(
                bounds=[(y_min, x_min), (y_max, x_max)],
                    fill = True,
                    fill_color= "blue",
                    fill_opacity = 0.2,
                    color = "blue",
                    weight = 0.5,
                popup=f"AOI {i+1}"
            ).add_to(m)

    return m 

#TODO: adapt function to new BBox Type (vorher Liste mit 4 Koordinaten, neu Typ BBox aus Sentinel Lib)
def display_aoi(AOI, map_size=(800, 600)):
    """
    Displays multiple bounding boxes on an interactive map using Folium.
    """

    map_center_lon = (AOI[0] + AOI[2]) / 2
    map_center_lat = (AOI[1] + AOI[3]) / 2

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=10, width=map_size[0], height=map_size[1])
    x_min, y_min, x_max, y_max = AOI
    folium.Rectangle(
            bounds=[(y_min, x_min), (y_max, x_max)],
            fill=True,
            fill_color = "blue",
            fill_opacity = 0.2,
            #color="black",
            weight = 0,
            popup=f"AOI"
        ).add_to(m)
    

    return m 




###########################################################################################
# Time Helper Functions 
###########################################################################################

def get_month_range(year_month):
    """
    Returns the first and last day of a given year_month
    """
    try:
        year, month = map(int, year_month.split('-'))
        if not (1 <= month <= 12):
            return None  # Invalid month

        _, last_day_num = calendar.monthrange(year, month)
        first_day = date(year, month, 1).strftime("%Y-%m-%d")
        last_day = date(year, month, last_day_num).strftime("%Y-%m-%d")

        return first_day, last_day

    except ValueError:
        return None 
# get_month_range(2020-01)
# > 2020-01-01, 2020-01-31

def get_days_in_range(start_date, end_date):
    """
    Returns a list of all days within a given date range (inclusive).
    """
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        if start > end:
            return None  # Invalid date range

        days = []
        current_date = start
        while current_date <= end:
            days.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        return days

    except ValueError:
        return None  # Invalid date format
# get_days_in_range(2020-01-01, 2020-01-03)
# > [2020-01-01, 2020-01-02, 2020-01-03]

def get_all_days_in_month(year_month):
    """
    Returns a list of all days in a given year and month.
    """
    try:
        year, month = map(int, year_month.split('-'))
        if not (1 <= month <= 12):
            return None  # Invalid month

        _, last_day_num = calendar.monthrange(year, month)
        start_date = date(year, month, 1)
        end_date = date(year, month, last_day_num)

        days = []
        current_date = start_date
        while current_date <= end_date:
            days.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)

        return days

    except ValueError:
        return None  # Invalid input format
# get_all_days_in_month(2020-01)
# > [2020-01-01, 2020-01-02, ..., 2020-01-30, 2020-01-31]

def get_months_of_year(year_str):
    """
    Returns a list of all months of a given year in the format "YYYY-MM".
    """
    try:
        year = int(year_str)
        if year < 0: #basic input validation.
            return None

        months = []
        for month in range(1, 13):
            months.append(f"{year}-{month:02}")  # Format month with leading zero

        return months

    except ValueError:
        return None  # Invalid year format
# get_months_of_year(2020)
# > [2020-01, 2020-02, ..., 2020-11, 2020-12]

###########################################################################################
# API Functions 
###########################################################################################

## Get API Access Token
def get_access_token():
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(TOKEN_URL, data=payload, headers=headers)
    response.raise_for_status() 
    return response.json()["access_token"]

## access catalog content for AOI and Time range
def get_aoi_catalog_results(CATALOG, TIME_INTERVALL, AOI_ID, AOI_BBOX):
    search_iterator = CATALOG.search(
        DataCollection.SENTINEL2_L2A,
        bbox=AOI_BBOX,
        time=TIME_INTERVALL,
        filter=f"eo:cloud_cover < 100",
        fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": ["id"]},
    )
    #aoi_catalog_results = list(search_iterator)
    aoi_catalog_results = {}
    for result in search_iterator:
        TS = result['properties']['datetime']
        CC = result['properties']['eo:cloud_cover']
        aoi_catalog_results[TS] = CC
    return aoi_catalog_results


def create_aoi_catalog(CATALOG, TIME_INTERVALL, AOI_BBOX_MAP, save_file = False):
    AOI_CATALOG_DICT = {}
    for AOI_ID in AOI_BBOX_MAP.keys():
        AOI_ID = AOI_ID
        AOI_BBOX = AOI_BBOX_MAP[AOI_ID]
        # get catalog content for sub aoi
        AOI_CATALOG_RESULTS = get_aoi_catalog_results(CATALOG, TIME_INTERVALL, AOI_ID, AOI_BBOX)
        AOI_CATALOG_DICT[AOI_ID] = AOI_CATALOG_RESULTS
    if save_file:
        # TODO data path dynamisch nöd so en seich
        with open('data/AOI_CATALOG.json', 'w') as fp:
            json.dump(AOI_CATALOG_DICT, fp)

    return AOI_CATALOG_DICT
    
def create_aoi_catalog_df(AOI_CATALOG):
    # doesnt work for 2020
    print("creating AOI_CATALOG_DF")
    AOI_CATALOG_ROWS = []
    for AOI in AOI_CATALOG.keys():
        AOI_ID = AOI
        AOI_RECS = AOI_CATALOG[AOI_ID]
        for AOI_REC in AOI_RECS:
            AOI_REC_TS = AOI_REC['properties']['datetime']
            AOI_REC_CC = AOI_REC['properties']['eo:cloud_cover']
            AOI_CATALOG_ROWS.append({"AOI_ID": AOI_ID, "TS": AOI_REC_TS, "CC": float(AOI_REC_CC)})
            AOI_CATALOG_DF = pd.DataFrame(AOI_CATALOG_ROWS)
            # create UID ({AOI_ID}_{TS})
            AOI_CATALOG_DF["UID"] = AOI_CATALOG_DF["AOI_ID"]+"_"+AOI_CATALOG_DF["TS"]
            # set datatypes for columns
            AOI_CATALOG_DF = AOI_CATALOG_DF.astype({
            "AOI_ID": "string",
            "TS": "string",
            "CC": "float",
            "UID": "string"
            })
    print(f"> created DF has {len(AOI_CATALOG_DF.index)} entries")
    return AOI_CATALOG_DF  

def create_aoi_catalog_df_old_data(AOI_CATALOG):
    # doesnt work for newer data
    print("creating AOI_CATALOG_DF")
    AOI_CATALOG_ROWS = []
    for AOI in AOI_CATALOG.keys():
        AOI_ID = AOI
        AOI_RECS = AOI_CATALOG[AOI_ID]
        for AOI_REC in AOI_RECS.keys():
            AOI_REC_TS = AOI_REC
            AOI_REC_CC = AOI_RECS[AOI_REC]
            AOI_CATALOG_ROWS.append({"AOI_ID": AOI_ID, "TS": AOI_REC_TS, "CC": float(AOI_REC_CC)})
            AOI_CATALOG_DF = pd.DataFrame(AOI_CATALOG_ROWS)
            # create UID ({AOI_ID}_{TS})
            AOI_CATALOG_DF["UID"] = AOI_CATALOG_DF["AOI_ID"]+"_"+AOI_CATALOG_DF["TS"]
            # set datatypes for columns
            AOI_CATALOG_DF = AOI_CATALOG_DF.astype({
            "AOI_ID": "string",
            "TS": "string",
            "CC": "float",
            "UID": "string"
            })
    print(f"> created DF has {len(AOI_CATALOG_DF.index)} entries")
    return AOI_CATALOG_DF  

def filter_aoi_catalog_df(AOI_CATALOG_DF, MAXCC):
    # filter out rows based on cloud coverage:
    print(f"removing entries with cloud coverage > {MAXCC} ")
    print(f"> found {len(AOI_CATALOG_DF[(AOI_CATALOG_DF['CC']>MAXCC)])} entries to remove")
    AOI_CATALOG_DF_F = AOI_CATALOG_DF[AOI_CATALOG_DF['CC'] <= MAXCC]
    print(f"> filtered dataframe has {len(AOI_CATALOG_DF_F.index)} entries")
    return AOI_CATALOG_DF_F

def filter_aoi_df(AOI_DF_2, MAXCC):
    # filter out rows based on cloud coverage:
    print(f"removing entries with cloud coverage > {MAXCC} ")
    print(f"> found {len(AOI_DF_2[(AOI_DF_2['CC_CLC']>MAXCC)])} entries to remove")
    AOI_DF_2_F = AOI_DF_2[AOI_DF_2['CC_CLC'] <= MAXCC]
    print(f"> filtered dataframe has {len(AOI_DF_2_F.index)} entries")
    return AOI_DF_2_F

def create_ts_dict(AOI_CATALOG_DF):
    TS_DICT = {}
    print(f"extracting timestamps for every AOI")
    # Group the dataframe by AOI_ID and extract timestamps
    for aoi_id, group in AOI_CATALOG_DF.groupby("AOI_ID"):
        TS_DICT[aoi_id] = group["TS"].tolist()
    
    return TS_DICT

## get catalog content for selected AOI
def get_catalog_content(AOI, maxCC, start_date, end_date, access_token):
    url = CATALOG_API_URL
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {access_token}"
    }
    data = {
        "collections": [
          "sentinel-2-l2a"
        ],
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "bbox": AOI,
        "limit": 100,
        "filter": {
          "op": "<",
          "args": [
            {
              "property": "eo:cloud_cover"
            },
            maxCC
          ]
        },
      "filter-lang": "cql2-json"
    }

    response = requests.post(url, headers=headers, json=data)
    content = response.content
    return(content)

## extract metadata
def extract_metadata(content):
    content = json.loads(content)
    features = content["features"]
    all_data = []
    for feature in features:
        feature_id = feature["id"]
        feature_geometry = feature["geometry"]
        #feature_geometry_coordinates = feature_geometry["coordinates"][0]
        feature_properties = feature["properties"]
        timestamp =  feature_properties["datetime"]
        cc = feature_properties["eo:cloud_cover"]
        row_data = {"ID": feature_id, 
                    #"Coordinates": feature_geometry_coordinates, 
                    "Timestamp": timestamp, 
                    "CloudCoverage": cc}
        all_data.append(row_data)
    return(all_data)

## get image
def get_satellite_image_2(AOI, maxCC, time, access_token, es, image_name):
  url = "https://services.sentinel-hub.com/api/v1/process"
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {access_token}"
  }
  data = {
    "input": {
      "bounds": {
        "bbox": AOI
      },
      "data": [
        {
          "dataFilter": {
            "timeRange": {
              "from": f"{time}",
              "to": f"{time}"
            },
            "maxCloudCoverage": maxCC
          },
          "processing": {
            "harmonizeValues": False
          },
          "type": "sentinel-2-l2a"
        }
      ]
    },
    "output": {
      "resx": 10,
      "resy": 10,
      "responses": [
        {
          "identifier": "default",
          "format": {
            "type": "image/png"
          }
        }
      ]
    },
    "evalscript": es
  }

  response = requests.post(url, headers=headers, json=data)

  if response.status_code == 200:
      with open(f"images/{image_name}.png", "wb") as f:
          f.write(response.content)
      print("Image downloaded successfully!")
  else:
      print("Error:", response.text)

def get_satellite_image(AOI, maxCC, time, access_token, dim, es, image_name):
  url = "https://services.sentinel-hub.com/api/v1/process"
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {access_token}"
  }
  data = {
    "input": {
      "bounds": {
        "bbox": AOI
      },
      "data": [
        {
          "dataFilter": {
            "timeRange": {
              "from": f"{time}",
              "to": f"{time}"
            },
            "maxCloudCoverage": maxCC
          },
          "processing": {
            "harmonizeValues": False
          },
          "type": "sentinel-2-l2a"
        }
      ]
    },
    "output": {
      "width": dim[0],
      "height": dim[1],
      "responses": [
        {
          "identifier": "default",
          "format": {
            "type": "image/png"
          }
        }
      ]
    },
    "evalscript": es
  }


  response = requests.post(url, headers=headers, json=data)
  return response

    #if response.status_code == 200:
   #    with open(f"images/{image_name}.png", "wb") as f:
   #        f.write(response.content)
   #    print("Image downloaded successfully!")
   #else:
   #    print("Error:", response.text)

### neuuuu

def get_img(evalscript, timestamp, bbox, resolution, CONFIG):
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    request_image = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=timestamp,
            )
        ],
        # TODO: für .tiff Bilder MimeType.TIFF setze
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=bbox_size,
        config=CONFIG,
    )
    imgs = request_image.get_data()
    
    return imgs[0]




def download_img(img, path, filename):
    save_path = os.path.join(path, filename)
    imageio.imwrite(save_path, img)


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
    red_pixels = (red_channel > red_threshold) & \
                 (red_channel > factor * green_channel) & \
                 (red_channel > factor * blue_channel)
    
    # Count red pixels and calculate the proportion
    red_count = np.sum(red_pixels)
    total_pixels = image.shape[0] * image.shape[1]
    red_proportion = red_count / total_pixels

    # Create a visualization where red pixels are full-intensity red
    red_visualization = np.zeros_like(image)  # Start with a black image
    red_visualization[red_pixels] = [255, 0, 0]  # Set red pixels to full red (255, 0, 0)
    
    # Show the visualization
    #plt.imshow(red_visualization)
    #plt.title("Red Pixel Visualization (Clouds in Red)")
    #plt.axis('off')  # Optional: Remove axes for a cleaner look
    #plt.show()

    #print(f"Proportion of Red Pixels (clouds): {red_proportion:.2%}")
    return red_proportion*100


def calculate_cc(AOI_CATALOG, AOI_BBOX_MAP, resolution, CONFIG):
    AOI_CATALOG_2 = {}
    evalscript = load_evalscript('evalscripts/es_true_color_clm.js')
    time_per_img = 2.42 # seconds
    print(f"Calculating Cloud Coverage")
    n_imgs= sum(len(keys) for keys in [list(AOI_CATALOG[primary_key].keys()) for primary_key in AOI_CATALOG])
    print(f"Estimated Time: {np.floor(n_imgs*time_per_img)} seconds, {(n_imgs*time_per_img)/60} minutes")
    for aoi_id in AOI_CATALOG.keys():
        aoi_bbox = AOI_BBOX_MAP[aoi_id]
        recordings = AOI_CATALOG[aoi_id]
        recordings_new = {}
        for recording in recordings.keys():
            timestamp =  recording
            cc_api = recordings[timestamp]
            img = get_img(evalscript = evalscript, timestamp = str(timestamp)[:10], bbox = aoi_bbox, resolution = resolution, CONFIG = CONFIG)
            cc_clc = calculate_red_coverage(img, red_threshold=50, factor=1.2)
            recordings_new[timestamp] = [cc_api, cc_clc]
        AOI_CATALOG_2[aoi_id] = recordings_new
    return AOI_CATALOG_2

def create_aoi_df_2(AOI_CATALOG_2):
    rows = []
    for aoi_id, recordings in AOI_CATALOG_2.items():
        for timestamp, values in recordings.items():
            rows.append({
                'AOI_ID': aoi_id,
                'Timestamp': timestamp,
                'CC_API': values[0],
                'CC_CLC': values[1]
            })

    AOI_CATALOG_df = pd.DataFrame(rows)
    return AOI_CATALOG_df

def load_evalscript(path):
    with open(path, 'r') as f:
        evalscript = f.read()
    return evalscript

#######

