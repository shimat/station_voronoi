import csv
import numpy as np
import numpy.typing as npt
import re
from typing import Iterable
from pathlib import Path
from pyproj import Transformer


def get_station_locations_in_area(area_name: str, transformer: Transformer) -> npt.NDArray[np.float64]:
    paths = tuple(_find_csv_files_in_island(area_name))
    if not paths:
        raise ValueError(f"Not supported area_name name: {area_name}")
    target = np.vstack(
        tuple(_load_station_csv(str(csv_path)) for csv_path in paths))
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in target])


def _load_station_csv(file_name: str) -> npt.NDArray[np.float64]:
    with open(file_name, "r", encoding="utf-8-sig", newline="") as file:
        csv_reader = csv.reader(file)
        return np.array([(float(row[1]), float(row[2])) for row in csv_reader if not row[0].startswith("#")])


def _find_csv_files_in_island(island_name: str) -> Iterable[Path]:
    paths = Path("csv").glob("*.csv")
    match island_name:
        case "北海道":
            return filter(lambda p: re.search(r"^北海道*|^道南*", p.name), paths)
        case "本州":
            return filter(lambda p: re.search(r"^東日本*", p.name), paths)
        case "四国":
            return filter(lambda p: re.search(r"^四国*|^阿佐海岸*|^土佐*|^伊予*|^高松琴平*", p.name), paths)
        case "九州":
            return filter(lambda p: re.search(r"^九州*|^西日本鉄道*|^熊本*|^南阿蘇*|^平成筑豊*|^島原*|^肥薩*|^くま川*|^甘木*|^松浦*", p.name), paths)
        case "沖縄本島":
            raise ValueError(f"Not supported island name: {island_name}")
        case _:
            raise ValueError(f"Not supported island name: {island_name}")


if __name__ == "__main__":
    print(list(_find_csv_files_in_island("四国")))
