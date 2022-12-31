import csv
import itertools
import json
import numpy as np
import numpy.typing as npt
import streamlit as st
from typing import Any, Iterable
from pathlib import Path
from pprint import pprint
from pyproj import Transformer
import shapely.geometry


def get_station_locations(file_name: str) -> npt.NDArray[np.float64]:
    with open(file_name, "r", encoding="utf-8-sig", newline="") as file:
        csv_reader = csv.reader(file)
        return np.array([(float(row[1]), float(row[2])) for row in csv_reader if not row[0].startswith("#")])


def get_all_station_locations(dir_name: str) -> npt.NDArray[np.float64]:
    return np.vstack(
        tuple(get_station_locations(str(csv_path)) for csv_path in Path(dir_name).glob("*.csv")))


def get_all_station_locations_transformed(dir_name: str, transformer: Transformer) -> npt.NDArray[np.float64]:
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in get_all_station_locations(dir_name)])


def get_pref_contour(pref_name: str) -> npt.NDArray[np.float64]:
    with open("geojson/pref.geojson", encoding="utf-8-sig") as file:
        geojson = json.load(file)

    feature = _first_true(geojson["features"], lambda f: f["properties"]["name"] == pref_name)
    if not feature:
        raise KeyError

    coordinates = list(itertools.chain.from_iterable(feature["geometry"]["coordinates"]))
    longest = sorted(coordinates, key=lambda c: len(c))[-1]
    return np.array(longest)


def get_pref_contour_transformed(pref_name: str, transformer: Transformer)-> npt.NDArray[np.float64]:
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in get_pref_contour(pref_name)])


def get_island_contour(name: str) -> npt.NDArray[np.float64]:
    with open("geojson/pref.geojson", encoding="utf-8-sig") as file:
        gj = json.load(file)

    shapes = _collect_island_shapes(name, gj)
    merged_shape = next(shapes)
    for s in shapes:
        merged_shape = merged_shape.union(s)

    coordinates = list(list(poly.exterior.coords) for poly in merged_shape.geoms)
    longest = sorted(coordinates, key=lambda c: len(c))[-1]

    return np.array(longest)


def get_main_islands_contours() -> npt.NDArray[np.float64]:
    shapes = (
        get_island_contour("北海道"),
        get_island_contour("本州"),
        get_island_contour("四国"),
        get_island_contour("九州"))    
    return np.vstack(shapes)


def get_main_islands_contours_transformed(transformer: Transformer)-> npt.NDArray[np.float64]:
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in get_main_islands_contours()])

def _collect_island_shapes(name: str, gj: dict[str, Any]) -> Iterable[shapely.geometry.shape]:
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
            case "沖縄本島":
                return [46]
            case _:
                raise ValueError(f"Not supported island name: {name}")

    return (shapely.geometry.shape(gj['features'][i]['geometry']) for i in get_indices(name))


def _first_true(iterable, pred):
    return next(filter(pred, iterable), None)