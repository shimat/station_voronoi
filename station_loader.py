import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import streamlit as st
from pyproj import Transformer

from transformers import get_transformer


@dataclass
class StationLocations:
    lonlat: pd.DataFrame
    utm: pd.DataFrame
    transformer: Transformer


@st.experimental_memo
def get_station_locations(area_name: str, transformer_pref: str) -> StationLocations:
    def map_row(row: pd.Series):
        lat, lon = transformer.transform(row["lat"], row["lon"])
        return {"name": row["name"], "lon": lon, "lat": lat}

    paths = tuple(_find_csv_files_in_island(area_name))
    if not paths:
        raise ValueError(f"Not supported area_name name: {area_name}")

    lonlat = pd.concat(_load_station_csv(str(csv_path)) for csv_path in paths)
    transformer = get_transformer(transformer_pref, "")
    utm = lonlat.apply(map_row, axis=1, result_type="expand")
    return StationLocations(lonlat=lonlat, utm=utm, transformer=transformer)


def get_station_locations_in_area(
    area_name: str, transformer_pref: str, area_contours: tuple[npt.NDArray, ...]
) -> StationLocations:
    stations = get_station_locations(area_name, transformer_pref)
    locations = stations.utm[["lon", "lat"]].values
    inside_points_indices = [
        i
        for i, p in enumerate(locations)
        if any(
            cv2.pointPolygonTest(contour.astype(np.float32), (p[0], p[1]), measureDist=False) >= 0
            for contour in area_contours
        )
    ]
    return StationLocations(
        lonlat=stations.lonlat.iloc[inside_points_indices, :],
        utm=stations.utm.iloc[inside_points_indices, :],
        transformer=stations.transformer,
    )


def _load_station_csv(file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name, encoding="utf-8-sig", names=("name", "lon", "lat"))
    df = df[df["name"].str.match("(?!#)")]
    return df


def _find_csv_files_in_island(island_name: str) -> Iterable[Path]:
    paths = Path("csv").glob("*.csv")
    match island_name:
        case "全国":
            return paths
        case "北海道":
            return filter(lambda p: re.search(r"^北海道.+|^道南.+|^札幌.+|^函館.+", p.name), paths)
        case "本州":
            ret = set(paths)
            for n in ("北海道", "四国", "九州", "沖縄本島"):
                ret -= set(_find_csv_files_in_island(n))
            return ret
        case "四国":
            return filter(lambda p: re.search(r"^四国.+|^阿佐海岸.+|^土佐.+|^伊予.+|^高松琴平.+", p.name), paths)
        case "九州":
            return filter(
                lambda p: re.search(
                    r"^九州.*|^西日本鉄道.+|^熊本.+|^南阿蘇.+|^平成筑豊.+|^島原.+|^肥薩.+|^くま川.+|^甘木.+|^松浦.+|^福岡.+|^北九州.+", p.name
                ),
                paths,
            )
        case "東京23区":
            return filter(
                lambda p: re.search(
                    r"^東日本旅客鉄道(山手線|東海道線|南武線|京浜東北線|赤羽線|埼京線|総武本線|京葉線|中央本線|常磐線|東北本線).+|"
                    + r"^東京.+|^京王.+|^小田急.+|^東急.+|^京浜急行.+|^ゆりかもめ.+|^首都圏.+|^西武.+|^東武.+|^京成.+",
                    p.name,
                ),
                paths,
            )
        case "大阪市":
            return filter(
                lambda p: re.search(r"^西日本旅客鉄道.+|^北?大阪.+|^京阪.+|^阪神.+|^阪急.+|^南海.+|^近畿.+|^阪堺.+", p.name),
                paths,
            )
        case "沖縄本島":
            return filter(lambda p: re.search(r"^沖縄.+", p.name), paths)
        case _:
            raise ValueError(f"Not supported island name: {island_name}")


if __name__ == "__main__":
    print(list(_find_csv_files_in_island("四国")))
