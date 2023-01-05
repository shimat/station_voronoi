from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingParameters:
    scale: float
    xmin: int
    ymin: int


@dataclass(frozen=True)
class FarthestPoint:
    lonlat: tuple[float]
    utm_xy: tuple[float]
    image_xy: tuple[int]
