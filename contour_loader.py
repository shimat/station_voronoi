import itertools
import json
import re
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import shapely.geometry
from pyproj import Transformer


def get_pref_contour(pref_name: str, transformer: Transformer) -> npt.NDArray[np.float64]:
    with open("geojson/prefectures.geojson", encoding="utf-8-sig") as file:
        geojson = json.load(file)

    feature = _first_true(geojson["features"], lambda f: f["properties"]["name"] == pref_name)
    if not feature:
        raise KeyError(f"{pref_name} not found")

    coordinates = itertools.chain.from_iterable(feature["geometry"]["coordinates"])
    longest = sorted(coordinates, key=lambda c: len(c))[-1]
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in longest])


def get_area_contour(name: str, transformer: Transformer) -> npt.NDArray[np.float64]:
    with open("geojson/prefectures.geojson", encoding="utf-8-sig") as file:
        gj = json.load(file)

    shapes = _collect_area_shapes(name, gj)
    merged_shape = next(shapes)
    for s in shapes:
        merged_shape = merged_shape.union(s)

    coordinates = [list(poly.exterior.coords) for poly in merged_shape.geoms]
    longest = sorted(coordinates, key=lambda c: len(c))[-1]
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in longest])


def get_main_islands_contours(
    transformer: Transformer,
) -> tuple[npt.NDArray[np.float64], ...]:
    return (
        get_area_contour("北海道", transformer),
        get_area_contour("本州", transformer),
        get_area_contour("四国", transformer),
        get_area_contour("九州", transformer),
    )


def get_area_contours_from_prefecture(
    pref_name: str, pattern: re.Pattern, transformer: Transformer
) -> tuple[npt.NDArray[np.float64], ...]:
    with open(f"geojson/{pref_name}.geojson", encoding="utf-8-sig") as file:
        gj = json.load(file)

    shapes = [shapely.geometry.shape(f["geometry"]) for f in gj["features"] if pattern.search(f["properties"]["name"])]
    # print([s.geom_type for s in shapes])
    if pref_name == "大阪府":
        shapes = sorted(shapes, key=lambda s: s.bounds[0])
        print("##### shape0 #####")
        print(shapes[0])

    if not shapes:
        raise Exception(".*区 not found")
    merged_shape = shapes[0]
    for s in shapes[1:]:
        merged_shape = merged_shape.union(s)

    coordinates = [list(poly.exterior.coords) for poly in merged_shape.geoms]
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


def _first_true(iterable, pred):
    return next(filter(pred, iterable), None)
