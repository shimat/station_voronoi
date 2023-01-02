import csv
import numpy as np
import numpy.typing as npt
import re
import cv2
from typing import Iterable
from pathlib import Path
from pyproj import Transformer


def get_station_locations(
    area_name: str,
    transformer: Transformer
) -> npt.NDArray[np.float64]:
    paths = tuple(_find_csv_files_in_island(area_name))
    if not paths:
        raise ValueError(f"Not supported area_name name: {area_name}")

    target = np.vstack(
        tuple(_load_station_csv(str(csv_path)) for csv_path in paths))
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in target])


def get_station_locations_in_area(
    area_name: str,
    transformer: Transformer,
    area_contour: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    transformed = get_station_locations(area_name, transformer)
    print(area_contour[0:2])
    inside_points = [cv2.pointPolygonTest(area_contour.astype(np.float32), p, measureDist=False) >= 0 for p in transformed]
    return np.array(inside_points)


def _load_station_csv(file_name: str) -> npt.NDArray[np.float64]:
    with open(file_name, "r", encoding="utf-8-sig", newline="") as file:
        csv_reader = csv.reader(file)
        return np.array([(float(row[1]), float(row[2])) for row in csv_reader if not row[0].startswith("#")])


def _find_csv_files_in_island(island_name: str) -> Iterable[Path]:
    paths = Path("csv").glob("*.csv")
    match island_name:
        case "全国":
            return paths
        case "北海道":
            return filter(lambda p: re.search(r"^北海道.+|^道南.+", p.name), paths)
        case "本州":
            ret = set(paths)
            for n in ("北海道", "四国", "九州"):
                ret -= set(_find_csv_files_in_island(n))
            return ret
        case "四国":
            return filter(lambda p: re.search(r"^四国.+|^阿佐海岸.+|^土佐.+|^伊予.+|^高松琴平.+", p.name), paths)
        case "九州":
            return filter(lambda p: re.search(r"^九州.*|^西日本鉄道.+|^熊本.+|^南阿蘇.+|^平成筑豊.+|^島原.+|^肥薩.+|^くま川.+|^甘木.+|^松浦.+", p.name), paths)
        case "関東":
            return filter(lambda p: re.search(r"^西武.+|^東武.+|^東日本旅客鉄道(山手線|東海道線|南武線|横浜線|相模線|横須賀線|鶴見線|総武本線|内房線|外房線|久留里線|成田線|中央本線|常磐線|東北本線|高崎線|両毛線|上越線|吾妻線|信越本線).+|御殿場線|^東急.+|^京急.+", p.name), paths)
        case "沖縄本島":
            raise ValueError(f"Not supported island name: {island_name}")
        case _:
            raise ValueError(f"Not supported island name: {island_name}")


if __name__ == "__main__":
    print(list(_find_csv_files_in_island("四国")))
