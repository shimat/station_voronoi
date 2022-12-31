import csv
import itertools
import json
import numpy as np
import numpy.typing as npt
import streamlit as st
from typing import Iterable
from pathlib import Path
from pprint import pprint
from pyproj import Transformer


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
    #pprint(longest)
    return np.array(longest)


def get_pref_contour_transformed(pref_name: str, transformer: Transformer)-> npt.NDArray[np.float64]:
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in get_pref_contour(pref_name)])


def _first_true(iterable, pred):
    return next(filter(pred, iterable), None)