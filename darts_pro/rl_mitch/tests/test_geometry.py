from darts_pro.rl_mitch.geometry import Point2d, Size2d


class TestSize2d:
    @staticmethod
    def test_init() -> None:
        size = Size2d(1, 2)
        assert size.width_mm == 1
        assert size.height_mm == 2


class TestPoint2d:
    @staticmethod
    def test_init() -> None:
        point = Point2d(1, 2)
        assert point.x == 1
        assert point.y == 2

    @staticmethod
    def test_add() -> None:
        point1 = Point2d(1, 2)
        point2 = Point2d(3, 4)
        assert point1 + point2 == Point2d(4, 6)

    @staticmethod
    def test_subtract() -> None:
        point1 = Point2d(1, 2)
        point2 = Point2d(3, 4)
        assert point1 - point2 == Point2d(-2, -2)
