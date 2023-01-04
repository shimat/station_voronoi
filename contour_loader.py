import functools
import itertools
import json
import re
from typing import Any, Iterable

import more_itertools
import numpy as np
import numpy.typing as npt
import shapely.geometry
import streamlit as st

from transformers import get_transformer


def get_pref_contour(pref_name: str, transformer_pref: str) -> npt.NDArray[np.float64]:
    with open("geojson/prefectures.geojson", encoding="utf-8-sig") as file:
        geojson = json.load(file)

    feature = more_itertools.first_true(geojson["features"], lambda f: f["properties"]["name"] == pref_name)
    if not feature:
        raise KeyError(f"{pref_name} not found")

    coordinates = itertools.chain.from_iterable(feature["geometry"]["coordinates"])
    longest = sorted(coordinates, key=lambda c: len(c))[-1]
    transformer = get_transformer(transformer_pref, "")
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in longest])


# @st.experimental_memo
def get_area_contour(name: str, transformer_pref: str) -> npt.NDArray[np.float64]:
    with open("geojson/prefectures.geojson", encoding="utf-8-sig") as file:
        gj = json.load(file)

    shapes = _collect_area_shapes(name, gj)
    merged_shape = next(shapes)
    for s in shapes:
        merged_shape = merged_shape.union(s)

    coordinates = [list(poly.exterior.coords) for poly in merged_shape.geoms]
    longest = sorted(coordinates, key=lambda c: len(c))[-1]
    transformer = get_transformer(transformer_pref, "")
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in longest])


def get_main_islands_contours() -> tuple[npt.NDArray[np.float64], ...]:
    return (
        get_area_contour("北海道", "東京都"),
        get_area_contour("本州", "東京都"),
        get_area_contour("四国", "東京都"),
        get_area_contour("九州", "東京都"),
    )


# @st.experimental_memo
def get_area_contours_from_prefecture(
    pref_name: str, pattern: re.Pattern, transformer_pref: str
) -> tuple[npt.NDArray[np.float64], ...]:
    with open(f"geojson/{pref_name}.geojson", encoding="utf-8-sig") as file:
        gj = json.load(file)

    shapes = [shapely.geometry.shape(f["geometry"]) for f in gj["features"] if pattern.search(f["properties"]["name"])]
    # 西に離れた島 (大阪沖埋立処分場) が強調されてしまうので除外
    if pref_name == "大阪府":
        shapes = sorted(shapes, key=lambda s: s.bounds[0])
        polygons = sorted(shapes[0].geoms, key=lambda p: p.bounds[0])
        shapes[0] = shapely.geometry.MultiPolygon(polygons[1:])

    if not shapes:
        raise Exception(f"Pattern {str(pattern)} not found")
    merged_shape = functools.reduce(lambda r, s: r.union(s), shapes[1:], shapes[0])

    coordinates = [list(poly.exterior.coords) for poly in merged_shape.geoms]
    transformer = get_transformer(transformer_pref, "")
    return tuple(np.array([transformer.transform(lat, lon)[::-1] for lon, lat in c]) for c in coordinates)


def _collect_area_shapes(name: str, gj: dict[str, Any]) -> Iterable[shapely.geometry.shape]:
    def get_indices(name: str) -> list[int]:
        match name:
            case "北海道":
                return [0]
            case "本州":
                return list(range(1, 35))
            case "四国":
                return list(range(35, 39))
            case "九州":
                return list(range(39, 46))
            case "関東":
                return list(range(7, 14))
            case "沖縄本島":
                return [46]
            case _:
                raise ValueError(f"Not supported area name: {name}")

    return (shapely.geometry.shape(gj["features"][i]["geometry"]) for i in get_indices(name))
