from typing import Iterable

import cv2
import more_itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st

from models import FarthestPoint


# ボロノイ分割の実行
@st.experimental_memo
def get_voronoi_division(
    station_df: pd.DataFrame, farthest_point: FarthestPoint, image_size: int
) -> tuple[tuple[npt.NDArray[np.float_]], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    points = station_df[["lon", "lat"]].values
    subdiv = cv2.Subdiv2D((0, 0, image_size + 1, image_size + 1))
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
    image_size: int,
) -> None:
    img = np.zeros((image_size + 100, image_size + 100, 3), np.uint8)
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
