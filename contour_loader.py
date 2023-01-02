import itertools
import json
import numpy as np
import numpy.typing as npt
from typing import Any, Iterable
from pyproj import Transformer
import shapely.geometry


def get_pref_contour(pref_name: str, transformer: Transformer) -> npt.NDArray[np.float64]:
    with open("geojson/pref.geojson", encoding="utf-8-sig") as file:
        geojson = json.load(file)

    feature = _first_true(geojson["features"], lambda f: f["properties"]["name"] == pref_name)
    if not feature:
        raise KeyError

    coordinates = itertools.chain.from_iterable(feature["geometry"]["coordinates"])
    longest = sorted(coordinates, key=lambda c: len(c))[-1]
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in longest])


def get_area_contour(name: str, transformer: Transformer) -> npt.NDArray[np.float64]:
    with open("geojson/pref.geojson", encoding="utf-8-sig") as file:
        gj = json.load(file)

    shapes = _collect_area_shapes(name, gj)
    merged_shape = next(shapes)
    for s in shapes:
        merged_shape = merged_shape.union(s)

    coordinates = [list(poly.exterior.coords) for poly in merged_shape.geoms]
    longest = sorted(coordinates, key=lambda c: len(c))[-1]
    return np.array([transformer.transform(lat, lon)[::-1] for lon, lat in longest])


def get_main_islands_contours(transformer: Transformer) -> tuple[npt.NDArray[np.float64], ...]:
    return (
        get_area_contour("北海道", transformer),
        get_area_contour("本州", transformer),
        get_area_contour("四国", transformer),
        get_area_contour("九州", transformer))


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

    return (shapely.geometry.shape(gj['features'][i]['geometry']) for i in get_indices(name))


def _first_true(iterable, pred):
    return next(filter(pred, iterable), None)
