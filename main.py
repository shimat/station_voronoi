import itertools
import re
from collections import deque
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import streamlit as st

from contour_loader import (
    get_area_contour,
    get_area_contours_from_prefecture,
    get_main_islands_contours,
    get_pref_contour,
)
from station_loader import get_station_locations, get_station_locations_in_area

IMAGE_SIZE = 2000
DEBUG = False


def show_plot(data: Iterable[tuple[Any, Any]]) -> None:
    colors = deque(("blue", "red", "green"))

    fig, ax = plt.subplots()
    for x, y in data:
        ax.scatter(x, y, c=colors[0], s=1)
        colors.rotate()
    ax.grid()
    st.pyplot(fig)


def show_islands_and_stations(
    station_locations: npt.NDArray[np.float64],
    island_contours: tuple[npt.NDArray[np.float64], ...],
) -> None:
    flatten_island_contours = np.vstack(island_contours)
    show_plot(
        [
            (flatten_island_contours[:, 0], flatten_island_contours[:, 1]),
            (station_locations[:, 0], station_locations[:, 1]),
        ]
    )


def normalize(
    station_locations: npt.NDArray[np.float64],
    island_contours: tuple[npt.NDArray[np.float64], ...],
) -> tuple[tuple[npt.NDArray[np.float64], ...], npt.NDArray[np.float64], float]:

    combi = np.vstack((station_locations, *island_contours))

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

    ret_station_locations, all_island_contours = np.vsplit(combi_norm, [len(station_locations)])
    sections = list(itertools.accumulate(len(c) for c in island_contours))[:-1]
    ret_island_contours = np.vsplit(all_island_contours, sections)

    return ret_island_contours, ret_station_locations, (1 / scale * IMAGE_SIZE)


@st.experimental_memo
def get_voronoi_result(points: npt.NDArray[np.float64]) -> tuple[Any, Any]:
    subdiv = cv2.Subdiv2D((0, 0, IMAGE_SIZE + 1, IMAGE_SIZE + 1))
    for p in points:
        subdiv.insert((p[0], p[1]))
    facets, centers = subdiv.getVoronoiFacetList([])
    return facets, centers


def voronoi(points: npt.NDArray[np.float64], island_contours: Iterable[npt.NDArray[np.float64]]) -> None:
    facets, centers = get_voronoi_result(points)

    img = np.zeros((IMAGE_SIZE + 100, IMAGE_SIZE + 100, 3), np.uint8)
    for p in centers:
        cv2.drawMarker(img, (p + 50).astype(int), (0, 0, 255), thickness=3)
    cv2.polylines(img, [(f + 50).astype(int) for f in facets], True, (0, 255, 255), thickness=2)

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


def distance_transform(
    station_locations: npt.NDArray[np.float64],
    island_contours: tuple[npt.NDArray[np.float64], ...],
    normalizing_scale: float
) -> None:
    img = np.full((IMAGE_SIZE, IMAGE_SIZE, 1), 255, dtype=np.uint8)
    for p in station_locations:
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

    _, _, _, max_loc = cv2.minMaxLoc(dist)

    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist_u8 = cv2.applyColorMap((dist * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    cv2.drawMarker(dist_u8, max_loc, (0, 0, 255), markerSize=30, thickness=5)
    dist_u8 = cv2.copyMakeBorder(dist_u8, 50, 50, 50, 50, cv2.BORDER_CONSTANT, (0, 0, 0))

    dist_u8 = cv2.flip(dist_u8, 0)
    st.image(dist_u8, channels="BGR", caption="Distance Transform")
    st.text(f"最遠点: ({max_loc[0]}, {IMAGE_SIZE-max_loc[1]})")
    
    lon, lat = max_loc[0]/normalizing_scale, (IMAGE_SIZE-max_loc[1])/normalizing_scale
    st.text(f"最遠点: ({lon}, {lat})")


st.set_page_config(page_title="駅からの距離")
st.title("駅からの距離")

tab_hokkaido, tab_honshu, tab_shikoku, tab_kyushu, tab_okinawa, tab_tokyo, tab_osaka, tab_main_islands = st.tabs(
    ("北海道", "本州", "四国", "九州", "沖縄本島", "東京23区", "大阪市", "全国")
)

with tab_hokkaido:
    island_contours = (get_area_contour("北海道", "北海道"),)
    station_locations = get_station_locations("北海道", "北海道")

    island_contours, station_locations, normalizing_scale = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours, normalizing_scale)

"""
with tab_honshu:
    island_contours = (get_area_contour("本州", "長野県"),)
    station_locations = get_station_locations("本州", "長野県")

    island_contours, station_locations = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_shikoku:
    island_contours = (get_area_contour("四国", "高知県"),)
    station_locations = get_station_locations("四国", "高知県")

    island_contours, station_locations = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_kyushu:
    island_contours = (get_area_contour("九州", "熊本県"),)
    station_locations = get_station_locations("九州", "熊本県")

    island_contours, station_locations = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_okinawa:
    island_contours = (get_pref_contour("沖縄県", "沖縄県"),)
    station_locations = get_station_locations("沖縄本島", "沖縄県")

    island_contours, station_locations = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_tokyo:
    island_contours = get_area_contours_from_prefecture("東京都", re.compile(r"区$"), "東京都")
    station_locations = get_station_locations_in_area("東京23区", "東京都", island_contours)

    island_contours, station_locations = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_osaka:
    island_contours = get_area_contours_from_prefecture("大阪府", re.compile(r"^大阪府大阪市"), "大阪府")
    station_locations = get_station_locations_in_area("大阪市", "大阪府", island_contours)

    island_contours, station_locations = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_main_islands:
    island_contours = get_main_islands_contours()
    station_locations = get_station_locations("全国", "東京都")

    island_contours, station_locations = normalize(station_locations, island_contours)
    if DEBUG:
        show_islands_and_stations(station_locations, island_contours)

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)
"""