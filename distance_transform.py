import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from pyproj import Transformer

from models import FarthestPoint, ScalingParameters


# 距離変換の結果を得る
def run_distance_transform(
    station_df: pd.DataFrame,
    island_contours: tuple[npt.NDArray[np.float64], ...],
    transformer: Transformer,
    scaling_parameters: ScalingParameters,
    image_size: int,
) -> tuple[npt.NDArray[np.float64], FarthestPoint]:
    img = np.full((image_size, image_size, 1), 255, dtype=np.uint8)
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
