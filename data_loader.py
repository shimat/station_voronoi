import csv
import numpy as np
import numpy.typing as npt
import streamlit as st
from pathlib import Path
from typing import Iterable


def get_station_locations(file_name: str) -> npt.NDArray[float]:
    with open(file_name, "r", encoding="utf-8-sig", newline="") as file:
        csv_reader = csv.reader(file)
        return np.array([(float(row[1]), float(row[2])) for row in csv_reader])


def get_all_station_locations(dir_name: str) -> npt.NDArray[float]:
    return np.vstack(
        (get_station_locations(str(csv_path)) for csv_path in Path(dir_name).glob("*.csv")))
