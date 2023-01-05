import itertools
import re
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from pyproj import Geod, Transformer

from contour_loader import (
    get_area_contour,
    get_area_contours_from_prefecture,
    get_main_islands_contours,
    get_pref_contour,
)
from distance_transform import run_distance_transform, show_distance_transform
from models import FarthestPoint, ScalingParameters
from station_loader import get_station_locations, get_station_locations_in_area
from voronoi import get_voronoi_division, show_voronoi

IMAGE_SIZE = 2000
DEBUG = False


def show_plot(data: Iterable[tuple[Any, Any]]) -> None:
    fig, ax = plt.subplots()
    for x, y in data:
        ax.scatter(x, y, s=1)
    ax.grid()
    st.pyplot(fig)


# 駅の位置と島の輪郭を出力するデバッグ用関数
def show_stations_and_islands(
    station_locations: pd.DataFrame,
    island_contours: tuple[npt.NDArray[np.float64], ...],
) -> None:
    flatten_island_contours = np.vstack(island_contours)
    stations_np = station_locations[["lon", "lat"]].values
    show_plot(
        [
            (flatten_island_contours[:, 0], flatten_island_contours[:, 1]),
            (stations_np[:, 0], stations_np[:, 1]),
        ]
    )


# 緯度経度座標列(UTM座標)から平面画像上の座標に正規化し変換
@st.experimental_memo
def normalize(
    station_df: pd.DataFrame,
    island_contours: tuple[npt.NDArray[np.float64], ...],
) -> tuple[tuple[npt.NDArray, ...], pd.DataFrame, ScalingParameters]:
    station_coordinates = station_df[["lon", "lat"]].values

    combi = np.vstack((station_coordinates, *island_contours))

    x, y = combi.T
    xmin, xmax, _, _ = cv2.minMaxLoc(x)
    ymin, ymax, _, _ = cv2.minMaxLoc(y)
    if (xmax - xmin) > (ymax - ymin):
        scale = xmax - xmin
    else:
        scale = ymax - ymin

    normx = ((x - xmin) / scale) * (IMAGE_SIZE)
    normy = ((y - ymin) / scale) * (IMAGE_SIZE)
    combi_norm = np.vstack((normx, normy)).T

    ret_station_coordinates, all_island_contours = np.vsplit(combi_norm, [len(station_df)])
    ret_station_df = station_df.copy()
    ret_station_df["lon"] = ret_station_coordinates[:, 0]
    ret_station_df["lat"] = ret_station_coordinates[:, 1]

    sections = list(itertools.accumulate(len(c) for c in island_contours))[:-1]
    ret_island_contours = np.vsplit(all_island_contours, sections)

    return ret_island_contours, ret_station_df, ScalingParameters((1 / scale * IMAGE_SIZE), xmin, ymin)


def show_informations(
    station_df_norm: pd.DataFrame, station_df_org: pd.DataFrame, farthest_point: FarthestPoint, nearest_station_location
) -> None:

    lon, lat = farthest_point.lonlat
    image_x, image_y = farthest_point.image_xy

    def d(row: pd.Series):
        lon1, lat1 = row[["lon", "lat"]]
        lon2, lat2 = nearest_station_location
        return pow(lon2 - lon1, 2) + pow(lat2 - lat1, 2)

    station_df_norm["dist"] = station_df_norm[["lon", "lat"]].apply(d, axis=1, result_type="expand")
    station_df_norm = station_df_norm.sort_values("dist")
    nearest_station_name = station_df_norm.iloc[0]["name"]
    station_pos = station_df_org[station_df_org["name"] == nearest_station_name].iloc[0, [1, 2]].values

    g = Geod(ellps="WGS84")
    _, _, distance_2d = g.inv(station_pos[0], station_pos[1], lon, lat)

    st.text(
        f"最遠点(緯度経度): ({lat=:.5f}, {lon=:.5f})\n"
        + f"最遠点(画像上): ({image_x}, {IMAGE_SIZE-image_y})\n"
        + f"最近傍の駅: {nearest_station_name}駅 (lat={station_pos[1]:.6g}, lon={station_pos[0]:.6g}, dist={distance_2d/1000:.3f}km)"
    )


st.set_page_config(page_title="駅から一番遠い場所", layout="wide")
st.title("駅から一番遠い場所")

TRANSFORMER_PREF_MAP = {
    "北海道": "北海道",
    "本州": "長野県",
    "四国": "高知県",
    "九州": "熊本県",
    "沖縄本島": "沖縄県",
    "東京23区": "東京都",
    "大阪市": "大阪府",
}

tab_names = ("北海道", "本州", "四国", "九州", "沖縄本島", "東京23区", "大阪市", "全国")
tabs = dict(zip(tab_names, st.tabs(tab_names)))


for area_name in ("北海道", "本州", "四国", "九州", "沖縄本島"):
    with tabs[area_name]:
        transformer_pref = TRANSFORMER_PREF_MAP[area_name]
        island_contours = (get_area_contour(area_name, transformer_pref),)
        stations = get_station_locations(area_name, transformer_pref)

        island_contours, stations_norm, scaling_parameters = normalize(stations.utm, island_contours)
        if DEBUG:
            show_stations_and_islands(stations_norm, island_contours)

        dist, farthest_point = run_distance_transform(
            stations_norm, island_contours, stations.transformer, scaling_parameters, IMAGE_SIZE
        )
        facets, centers, nearest_station_location = get_voronoi_division(stations_norm, farthest_point, IMAGE_SIZE)

        show_informations(stations_norm, stations.lonlat, farthest_point, nearest_station_location)

        col1, col2 = st.columns(2)
        with col1:
            show_distance_transform(dist, farthest_point)
        with col2:
            show_voronoi(facets, centers, island_contours, nearest_station_location, IMAGE_SIZE)


PATTERNS_MAP = {"東京23区": re.compile(r"区$"), "大阪市": re.compile(r"^大阪府大阪市")}

for area_name in ("東京23区", "大阪市"):
    with tabs[area_name]:
        island_contours = get_area_contours_from_prefecture(
            TRANSFORMER_PREF_MAP[area_name], PATTERNS_MAP[area_name], TRANSFORMER_PREF_MAP[area_name]
        )
        stations = get_station_locations_in_area(area_name, TRANSFORMER_PREF_MAP[area_name], island_contours)

        island_contours, stations_norm, scaling_parameters = normalize(stations.utm, island_contours)
        if DEBUG:
            show_stations_and_islands(stations_norm, island_contours)

        dist, farthest_point = run_distance_transform(
            stations_norm, island_contours, stations.transformer, scaling_parameters, IMAGE_SIZE
        )
        facets, centers, nearest_station_location = get_voronoi_division(stations_norm, farthest_point, IMAGE_SIZE)

        show_informations(stations_norm, stations.lonlat, farthest_point, nearest_station_location)

        col1, col2 = st.columns(2)
        with col1:
            show_distance_transform(dist, farthest_point)
        with col2:
            show_voronoi(facets, centers, island_contours, nearest_station_location, IMAGE_SIZE)


with tabs["全国"]:
    island_contours = get_main_islands_contours()
    stations = get_station_locations("全国", "東京都")

    island_contours, stations_norm, scaling_parameters = normalize(stations.utm, island_contours)
    if DEBUG:
        show_stations_and_islands(stations_norm, island_contours)

    dist, farthest_point = run_distance_transform(
        stations_norm, island_contours, stations.transformer, scaling_parameters, IMAGE_SIZE
    )
    facets, centers, nearest_station_location = get_voronoi_division(stations_norm, farthest_point, IMAGE_SIZE)

    show_informations(stations_norm, stations.lonlat, farthest_point, nearest_station_location)

    col1, col2 = st.columns(2)
    with col1:
        show_distance_transform(dist, farthest_point)
    with col2:
        show_voronoi(facets, centers, island_contours, nearest_station_location, IMAGE_SIZE)
