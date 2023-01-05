import itertools
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from pyproj import Transformer

from contour_loader import (
    get_area_contour,
    get_area_contours_from_prefecture,
    get_main_islands_contours,
    get_pref_contour,
)
from station_loader import get_station_locations, get_station_locations_in_area

IMAGE_SIZE = 2000
DEBUG = False


@dataclass(frozen=True)
class ScalingParameters:
    scale: float
    xmin: int
    ymin: int


@dataclass(frozen=True)
class FarthestPoint:
    lonlat: tuple[float]
    utm_xy: tuple[float]
    image_xy: tuple[int]


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


# ボロノイ分割の実行
@st.experimental_memo
def get_voronoi_division(
    station_df: pd.DataFrame,
) -> tuple[tuple[npt.NDArray[np.float_]], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    points = station_df[["lon", "lat"]].values
    subdiv = cv2.Subdiv2D((0, 0, IMAGE_SIZE + 1, IMAGE_SIZE + 1))
    for p in points:
        subdiv.insert((p[0], p[1]))
    facets, centers = subdiv.getVoronoiFacetList([])

    _, nearest_station_location = more_itertools.first_true(
        zip(facets, centers), pred=lambda c: cv2.pointPolygonTest(c[0], farthest_point.image_xy, measureDist=False) >= 0
    )

    return facets, centers, nearest_station_location


# ボロノイ分割の結果を可視化
def show_voronoi(
    facets: tuple[npt.NDArray[np.float_]],
    centers: npt.NDArray[np.float_],
    island_contours: Iterable[npt.NDArray[np.float64]],
    nearest_station_location,
) -> None:
    img = np.zeros((IMAGE_SIZE + 100, IMAGE_SIZE + 100, 3), np.uint8)
    for p in centers:
        cv2.drawMarker(img, (p + 50).astype(int), (0, 0, 255), thickness=3)
    cv2.polylines(img, [(f + 50).astype(int) for f in facets], True, (0, 255, 255), thickness=2)

    cv2.drawMarker(
        img,
        (nearest_station_location + 50).astype(int),
        (0, 200, 0),
        thickness=7,
        markerType=cv2.MARKER_STAR,
        markerSize=40,
    )

    for one_island in island_contours:
        cv2.polylines(
            img,
            [np.array([(f + 50).astype(int) for f in one_island])],
            True,
            (255, 255, 255),
            thickness=4,
        )
    img = cv2.flip(img, 0)
    st.image(img, channels="BGR", caption="Voronoi")


# 距離変換の結果を得る
def run_distance_transform(
    station_df: pd.DataFrame,
    island_contours: tuple[npt.NDArray[np.float64], ...],
    transformer: Transformer,
    scaling_parameters: ScalingParameters,
) -> tuple[npt.NDArray[np.float64], FarthestPoint]:
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 1), 255, dtype=np.uint8)
    for p in station_df[["lon", "lat"]].values:
        cv2.rectangle(img, p.astype(int), (p + 1).astype(int), (0, 0, 0), thickness=2)
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    mask = np.zeros_like(dist, dtype=np.uint8)
    for one_island in island_contours:
        cv2.fillPoly(
            mask,
            [np.array([f.astype(int) for f in one_island])],
            (255, 255, 255),
        )
    dist = dist * (mask / 255)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    _, _, _, max_loc = cv2.minMaxLoc(dist)

    # distance最大の点
    utmx = (max_loc[0] / scaling_parameters.scale) + scaling_parameters.xmin
    utmy = (max_loc[1] / scaling_parameters.scale) + scaling_parameters.ymin
    lat, lon = transformer.transform(utmy, utmx, direction="INVERSE")

    return dist, FarthestPoint(lonlat=(lon, lat), utm_xy=(utmx, utmy), image_xy=max_loc)


# 距離変換の結果を可視化
def show_distance_transform(
    dist: npt.NDArray[np.float64],
    farthest_point: FarthestPoint,
) -> None:
    dist_u8 = cv2.applyColorMap((dist * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    dist_u8 = cv2.copyMakeBorder(dist_u8, 50, 50, 50, 50, cv2.BORDER_CONSTANT, (0, 0, 0))
    cv2.drawMarker(
        dist_u8,
        [xy + 50 for xy in farthest_point.image_xy],
        (0, 0, 255),
        markerSize=50,
        markerType=cv2.MARKER_TILTED_CROSS,
        thickness=5,
    )
    dist_u8 = cv2.flip(dist_u8, 0)
    st.image(dist_u8, channels="BGR", caption="Distance Transform")


def show_informations(
    station_df: pd.DataFrame,
    farthest_point: FarthestPoint,
    nearest_station_location
) -> None:

    lon, lat = farthest_point.lonlat
    image_x, image_y = farthest_point.image_xy

    def d(row: pd.Series):
        lon1, lat1 = row[["lon", "lat"]]
        lon2, lat2 = nearest_station_location
        return pow(lon2 - lon1, 2) + pow(lat2 - lat1, 2)

    station_df["dist"] = station_df[["lon", "lat"]].apply(d, axis=1, result_type="expand")
    station_df = station_df.sort_values("dist")
    nearest_station_name = station_df.iloc[0]["name"]

    st.text(
        f"最遠点(緯度経度): ({lat:.9f}, {lon:.9f})\n最遠点(画像上): ({image_x}, {IMAGE_SIZE-image_y})\n最近傍の駅: {nearest_station_name}駅"
    )


st.set_page_config(page_title="駅から一番遠い場所", layout="wide")
st.title("駅から一番遠い場所")

TRANSFORMER_PREF_MAP = {"北海道": "北海道", "本州": "長野県", "四国": "高知県", "九州": "熊本県", "沖縄本島": "沖縄県"}

tab_names = ("北海道", "本州", "四国", "九州", "沖縄本島", "東京23区", "大阪市", "全国")
tabs = dict(zip(tab_names, st.tabs(tab_names)))


for area_name in ("北海道", "本州", "四国", "九州", "沖縄本島"):
    with tabs[area_name]:
        transformer_pref = TRANSFORMER_PREF_MAP[area_name]
        island_contours = (get_area_contour(area_name, transformer_pref),)
        station_df, transformer = get_station_locations(area_name, transformer_pref)

        island_contours, station_df, scaling_parameters = normalize(station_df, island_contours)
        if DEBUG:
            show_stations_and_islands(station_df, island_contours)

        dist, farthest_point = run_distance_transform(station_df, island_contours, transformer, scaling_parameters)
        facets, centers, nearest_station_location = get_voronoi_division(station_df)

        show_informations(station_df, farthest_point, nearest_station_location)

        col1, col2 = st.columns(2)
        with col1:
            show_distance_transform(dist, farthest_point)
        with col2:
            show_voronoi(facets, centers, island_contours, nearest_station_location)


for area_name in ("東京23区",):
    with tabs[area_name]:
        island_contours = get_area_contours_from_prefecture("東京都", re.compile(r"区$"), "東京都")
        station_df, transformer = get_station_locations_in_area("東京23区", "東京都", island_contours)

        island_contours, station_df, scaling_parameters = normalize(station_df, island_contours)
        if DEBUG:
            show_stations_and_islands(station_df, island_contours)

        dist, farthest_point = run_distance_transform(station_df, island_contours, transformer, scaling_parameters)
        facets, centers, nearest_station_location = get_voronoi_division(station_df)

        show_informations(station_df, farthest_point, nearest_station_location)

        col1, col2 = st.columns(2)
        with col1:
            show_distance_transform(dist, farthest_point)
        with col2:
            show_voronoi(facets, centers, island_contours, nearest_station_location)


"""
with tab_osaka:
    island_contours = get_area_contours_from_prefecture("大阪府", re.compile(r"^大阪府大阪市"), "大阪府")
    station_locations, transformer = get_station_locations_in_area("大阪市", "大阪府", island_contours)

    island_contours, station_locations, scaling_parameters = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours, transformer, scaling_parameters)


with tab_main_islands:
    island_contours = get_main_islands_contours()
    station_locations, transformer = get_station_locations("全国", "東京都")

    island_contours, station_locations, scaling_parameters = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours, transformer, scaling_parameters)
"""
