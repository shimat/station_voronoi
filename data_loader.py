import csv
import numpy as np
import streamlit as st


def get_station_locations(file_name: str) -> list[tuple[float, float]]:
    with open(file_name, "r", encoding="utf-8-sig", newline="") as file:
        csv_reader = csv.reader(file)
        return np.array([(float(row[2]), float(row[1])) for row in csv_reader])

