import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from collections import deque
from typing import Any, Iterable
from sklearn.preprocessing import MinMaxScaler
from data_loader import get_all_station_locations_transformed, get_pref_contour_transformed, get_island_contour, get_main_islands_contours_transformed
from transformers import get_transformer

IMAGE_SIZE = 2000


def show_plot(data: Iterable[tuple[Any, Any]]) -> None:
    colors = deque(("blue", "red", "green"))

    fig, ax = plt.subplots()
    for x, y in data:
        ax.scatter(x, y, c=colors[0], s=10)
        colors.rotate()
    ax.grid()
    st.pyplot(fig)


def voronoi(points: npt.NDArray[np.float64], island_contours: npt.NDArray[np.float64]) -> None:
    subdiv = cv2.Subdiv2D((0, 0, IMAGE_SIZE+1, IMAGE_SIZE+1))
    for p in points:
        subdiv.insert((p[0], p[1]))

    facets, centers = subdiv.getVoronoiFacetList([])
    #print(facets)

    img = np.zeros((IMAGE_SIZE+100, IMAGE_SIZE+100, 3), np.uint8)
    for p in centers:
        cv2.drawMarker(img, (p+50).astype(int), (0, 0, 255), thickness=3)
    #print(facets)
    cv2.polylines(img, [(f+50).astype(int) for f in facets], True, (0, 255, 255), thickness=2)

    cv2.polylines(img, [np.array([(f+50).astype(int) for f in island_contours])], True, (255, 255, 255), thickness=4)
    #img_mask = np.zeros_like(img)
    #cv2.fillPoly(img_mask, [np.array([(f+50).astype(int) for f in pref_contour])], (255, 255, 255))
    #img = cv2.bitwise_and(img, img_mask)
    img = cv2.flip(img, 0)
    #img_mask = cv2.flip(img_mask, 0)

    st.image(img, channels="BGR", caption="Voronoi")
    #st.image(img_mask, channels="BGR", caption="Mask")


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


tab1, tab2 = st.tabs(("北海道", "全国"))

with tab1:
    transformer = get_transformer("北海道", "")
    scaler = MinMaxScaler((0, IMAGE_SIZE))

    island_contours = get_pref_contour_transformed("北海道", transformer)
    station_locations = get_all_station_locations_transformed("csv", transformer)
    #show_plot([
    #    (pref_contour[:, 0], pref_contour[:, 1]),
    #    (station_locations[:, 0], station_locations[:, 1])])

    combi = np.vstack((island_contours, station_locations))
    combi_norm = scaler.fit_transform(combi)
    island_contours, station_locations = np.vsplit(combi_norm, [len(island_contours)])
    show_plot([
        (island_contours[:, 0], island_contours[:, 1]),
        (station_locations[:, 0], station_locations[:, 1])])

    voronoi(station_locations, island_contours)

    distance_transform(station_locations, island_contours)


with tab2:
    transformer = get_transformer("東京都", "")

    island_contours = get_main_islands_contours_transformed(transformer)
    show_plot([
        (island_contours[:, 0], island_contours[:, 1]),])