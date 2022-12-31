import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from collections import deque
from typing import Any, Iterable
from sklearn.preprocessing import MinMaxScaler
from data_loader import get_all_station_locations_transformed, get_pref_contour_transformed
from transformers import get_transformer

IMAGE_SIZE = 1000


def show_plot(data: Iterable[tuple[Any, Any]]) -> None:
    colors = deque(("blue", "red", "green"))

    fig, ax = plt.subplots()
    for x, y in data:
        ax.scatter(x, y, c=colors[0], s=10)
        colors.rotate()
    ax.grid()
    st.pyplot(fig)


def voronoi(points: npt.NDArray[np.float64], pref_contour: npt.NDArray[np.float64]) -> None:
    subdiv = cv2.Subdiv2D((0, 0, IMAGE_SIZE+1, IMAGE_SIZE+1))
    for p in points:
        subdiv.insert((p[0], p[1]))

    facets, centers = subdiv.getVoronoiFacetList([])
    #print(facets)

    img = np.zeros((IMAGE_SIZE+100, IMAGE_SIZE+100, 3), np.uint8)
    for p in centers:
        cv2.drawMarker(img, (p+50).astype(int), (0, 0, 255), thickness=2)
    #print(facets)
    cv2.polylines(img, [(f+50).astype(int) for f in facets], True, (255, 255, 255), thickness=1)

    print(pref_contour)
    print(type(pref_contour))
    cv2.polylines(img, [[[int(f[0]), int(f[1])] for f in pref_contour.astype(int)]], True, (255, 255, 255), thickness=1)
    img = cv2.flip(img, 0)

    st.image(img, channels="BGR", caption="Voronoi")

transformer = get_transformer("北海道", "")
scaler = MinMaxScaler((0, IMAGE_SIZE))

pref_contour = get_pref_contour_transformed("北海道", transformer)
station_locations = get_all_station_locations_transformed("csv", transformer)
#show_plot([
#    (pref_contour[:, 0], pref_contour[:, 1]),
#    (station_locations[:, 0], station_locations[:, 1])])

combi = np.vstack((pref_contour, station_locations))
combi_norm = scaler.fit_transform(combi)
pref_contour, station_locations = np.vsplit(combi_norm, [len(pref_contour)])
show_plot([
    (pref_contour[:, 0], pref_contour[:, 1]),
    (station_locations[:, 0], station_locations[:, 1])])

voronoi(station_locations, pref_contour)


img = np.full((IMAGE_SIZE+100, IMAGE_SIZE+100, 1), 255, dtype=np.uint8)
for p in station_locations:
    cv2.rectangle(img, (p+50).astype(int), (p+51).astype(int), (0, 0, 0), thickness=2)
img = cv2.flip(img, 0)
dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)

cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
dist = cv2.applyColorMap((dist*255).astype(np.uint8), cv2.COLORMAP_JET)
st.image(dist, caption="distance")