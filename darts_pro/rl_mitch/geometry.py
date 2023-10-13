from dataclasses import dataclass


@dataclass
class Size2d:
    """The size of an object in 2D space."""

    width_mm: int
    height_mm: int


@dataclass
class Point2d:
    """A point in 2D space."""

    x: float  # pylint: disable=invalid-name
    y: float  # pylint: disable=invalid-name

    def __add__(self, other: "Point2d") -> "Point2d":
        return Point2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2d") -> "Point2d":
        return Point2d(self.x - other.x, self.y - other.y)
