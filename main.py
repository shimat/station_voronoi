import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from collections import deque
from typing import Any, Iterable
from sklearn.preprocessing import MinMaxScaler
from contour_loader import get_pref_contour, get_island_contour, get_main_islands_contours
from station_loader import get_station_locations_in_area
from transformers import get_transformer

IMAGE_SIZE = 2000


def show_plot(data: Iterable[tuple[Any, Any]]) -> None:
    colors = deque(("blue", "red", "green"))

    fig, ax = plt.subplots()
    for x, y in data:
        ax.scatter(x, y, c=colors[0], s=1)
        colors.rotate()
    ax.grid()
    st.pyplot(fig)


def normalize(
    island_contours: npt.NDArray[np.float64],
    station_locations: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # combi = np.vstack((island_contours, station_locations))
    # combi_norm = scaler.fit_transform(combi)
    # island_contours, station_locations = np.vsplit(combi_norm, [len(island_contours)])

    combi = np.vstack((island_contours, station_locations))

    x, y = combi.T
    xmin, xmax, _, _ = cv2.minMaxLoc(x)
    ymin, ymax, _, _ = cv2.minMaxLoc(y)
    if (xmax - xmin) > (ymax - ymin):
        scale = (xmax - xmin)
    else:
        scale = (ymax - ymin)

    normx = ((x-xmin)/scale) * (IMAGE_SIZE)
    normy = ((y-ymin)/scale) * (IMAGE_SIZE)
    combi_norm = np.vstack((normx, normy)).T

    island_contours, station_locations = np.vsplit(combi_norm, [len(island_contours)])

    return island_contours, station_locations


def voronoi(points: npt.NDArray[np.float64], island_contours: npt.NDArray[np.float64]) -> None:
    subdiv = cv2.Subdiv2D((0, 0, IMAGE_SIZE+1, IMAGE_SIZE+1))
    for p in points:
        subdiv.insert((p[0], p[1]))

    facets, centers = subdiv.getVoronoiFacetList([])

    img = np.zeros((IMAGE_SIZE+100, IMAGE_SIZE+100, 3), np.uint8)
    for p in centers:
        cv2.drawMarker(img, (p+50).astype(int), (0, 0, 255), thickness=3)
    cv2.polylines(img, [(f+50).astype(int) for f in facets], True, (0, 255, 255), thickness=2)

    cv2.polylines(img, [np.array([(f+50).astype(int) for f in island_contours])], True, (255, 255, 255), thickness=4)
    img = cv2.flip(img, 0)

    st.image(img, channels="BGR", caption="Voronoi")
    cv2.imwrite("voronoi.png", img)


def distance_transform(points: npt.NDArray[np.float64], island_contours: npt.NDArray[np.float64]) -> None:
    img = np.full((IMAGE_SIZE+100, IMAGE_SIZE+100, 1), 255, dtype=np.uint8)
    for p in station_locations:
        cv2.rectangle(img, (p+50).astype(int), (p+51).astype(int), (0, 0, 0), thickness=2)
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    mask = np.zeros_like(dist, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([(f+50).astype(int) for f in island_contours])], (255, 255, 255))
    dist = dist * (mask / 255)

    _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)

    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    dist_u8 = cv2.applyColorMap((dist*255).astype(np.uint8), cv2.COLORMAP_HOT)

    cv2.drawMarker(dist_u8, maxLoc, (0, 0, 255), markerSize=30, thickness=5)

    dist_u8 = cv2.flip(dist_u8, 0)
    st.image(dist_u8, channels="BGR", caption="distance")


tab_hokkaido, tab_honshu, tab_shikoku, tab_kyushu, tab_4islands = st.tabs(("北海道", "本州", "四国", "九州", "全国"))

with tab_hokkaido:
    transformer = get_transformer("北海道", "")
    scaler = MinMaxScaler((0, IMAGE_SIZE))

    island_contours = get_pref_contour("北海道", transformer)
    station_locations = get_station_locations_in_area("北海道", transformer)

    island_contours, station_locations = normalize(island_contours, station_locations)
    show_plot([
        (island_contours[:, 0], island_contours[:, 1]),
        (station_locations[:, 0], station_locations[:, 1])])

    voronoi(station_locations, island_contours)

    distance_transform(station_locations, island_contours)


with tab_honshu:
    transformer = get_transformer("長野県", "")
    scaler = MinMaxScaler((0, IMAGE_SIZE))

    island_contours = get_island_contour("本州", transformer)
    station_locations = get_station_locations_in_area("本州", transformer)

    island_contours, station_locations = normalize(island_contours, station_locations)
    show_plot([
        (island_contours[:, 0], island_contours[:, 1]),
        (station_locations[:, 0], station_locations[:, 1])])

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_shikoku:
    transformer = get_transformer("高知県", "")
    scaler = MinMaxScaler((0, IMAGE_SIZE))

    island_contours = get_island_contour("四国", transformer)
    station_locations = get_station_locations_in_area("四国", transformer)

    island_contours, station_locations = normalize(island_contours, station_locations)
    show_plot([
        (island_contours[:, 0], island_contours[:, 1]),
        (station_locations[:, 0], station_locations[:, 1])])

    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_kyushu:
    transformer = get_transformer("熊本県", "")
    scaler = MinMaxScaler((0, IMAGE_SIZE))

    island_contours = get_island_contour("九州", transformer)
    station_locations = get_station_locations_in_area("九州", transformer)

    island_contours, station_locations = normalize(island_contours, station_locations)
    show_plot([
        (island_contours[:, 0], island_contours[:, 1]),
        (station_locations[:, 0], station_locations[:, 1])])
        
    voronoi(station_locations, island_contours)
    distance_transform(station_locations, island_contours)


with tab_4islands:
    transformer = get_transformer("東京都", "")

    island_contours = get_main_islands_contours(transformer)
    show_plot([
        (island_contours[:, 0], island_contours[:, 1]),])
