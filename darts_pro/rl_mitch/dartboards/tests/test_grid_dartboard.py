import numpy as np
import pytest
from darts_pro.rl_mitch.dartboards.grid_dartboard import (
    GridDartBoard,
    GridDartBoardScoringLogic,
    ScoreGrid,
)
from darts_pro.rl_mitch.geometry import Point2d, Size2d


@pytest.fixture(name="board_size_mm")
def fixture_board_size_mm() -> Size2d:
    return Size2d(100, 100)


@pytest.fixture(name="score_grid")
def fixture_score_grid() -> ScoreGrid:
    return ScoreGrid(np.array([[0, 1], [2, 3], [4, 5]]))


@pytest.fixture(name="grid_dartboard")
def fixture_grid_dartboard(
    board_size_mm: Size2d, score_grid: ScoreGrid
) -> GridDartBoard:
    return GridDartBoard(board_size_mm, score_grid)


class TestGridDartBoardScoringLogic:
    @staticmethod
    def test_init(score_grid: ScoreGrid, board_size_mm: Size2d) -> None:
        scoring_logic = GridDartBoardScoringLogic(score_grid, board_size_mm)
        assert scoring_logic.score_grid == score_grid
        assert scoring_logic.size_mm == board_size_mm

    @staticmethod
    def test_vertical_segment_height(
        score_grid: ScoreGrid, board_size_mm: Size2d
    ) -> None:
        scoring_logic = GridDartBoardScoringLogic(score_grid, board_size_mm)
        assert scoring_logic.vertical_segment_height == pytest.approx(100 / 3)

    @staticmethod
    def test_horizontal_segment_width(
        score_grid: ScoreGrid, board_size_mm: Size2d
    ) -> None:
        scoring_logic = GridDartBoardScoringLogic(score_grid, board_size_mm)
        assert scoring_logic.horizontal_segment_width == pytest.approx(50)

    @staticmethod
    def test_score(score_grid: ScoreGrid, board_size_mm: Size2d) -> None:

        scoring_logic = GridDartBoardScoringLogic(score_grid, board_size_mm)

        assert scoring_logic.score(Point2d(0, 0)) == 0
        assert scoring_logic.score(Point2d(0, 33.33)) == 0
        assert scoring_logic.score(Point2d(33.33, 0)) == 0

        assert scoring_logic.score(Point2d(0, 33.34)) == 2
        assert scoring_logic.score(Point2d(0, 66.6)) == 2
        assert scoring_logic.score(Point2d(49.9, 33.34)) == 2
        assert scoring_logic.score(Point2d(49.9, 66.6)) == 2

        assert scoring_logic.score(Point2d(0, 66.7)) == 4
        assert scoring_logic.score(Point2d(0, 99.9)) == 4
        assert scoring_logic.score(Point2d(49.9, 66.7)) == 4
        assert scoring_logic.score(Point2d(49.9, 99.9)) == 4

        assert scoring_logic.score(Point2d(50.1, 0)) == 1
        assert scoring_logic.score(Point2d(50.1, 33.33)) == 1
        assert scoring_logic.score(Point2d(99.9, 0)) == 1
        assert scoring_logic.score(Point2d(99.9, 33.33)) == 1

        assert scoring_logic.score(Point2d(50.1, 33.34)) == 3
        assert scoring_logic.score(Point2d(50.1, 66.6)) == 3
        assert scoring_logic.score(Point2d(99.9, 33.34)) == 3
        assert scoring_logic.score(Point2d(99.9, 66.6)) == 3

        assert scoring_logic.score(Point2d(50.1, 66.7)) == 5
        assert scoring_logic.score(Point2d(50.1, 99.9)) == 5
        assert scoring_logic.score(Point2d(99.9, 66.7)) == 5
        assert scoring_logic.score(Point2d(99.9, 99.9)) == 5


class TestGridDartBoard:
    @staticmethod
    def test_init(board_size_mm: Size2d, score_grid: ScoreGrid) -> None:
        dartboard = GridDartBoard(board_size_mm, score_grid)
        assert dartboard.size_mm == board_size_mm
        assert dartboard.score_grid == score_grid

    @staticmethod
    def test_centre(board_size_mm: Size2d, score_grid: ScoreGrid) -> None:
        dartboard = GridDartBoard(board_size_mm, score_grid)
        assert dartboard.centre == Point2d(50, 50)

    @staticmethod
    def test_score(grid_dartboard: GridDartBoard) -> None:
        assert grid_dartboard.score(Point2d(0, 0)) == 0
        assert grid_dartboard.score(Point2d(0, 33.33)) == 0
        assert grid_dartboard.score(Point2d(33.33, 0)) == 0

        assert grid_dartboard.score(Point2d(0, 33.34)) == 2
        assert grid_dartboard.score(Point2d(0, 66.6)) == 2
        assert grid_dartboard.score(Point2d(49.9, 33.34)) == 2
        assert grid_dartboard.score(Point2d(49.9, 66.6)) == 2

        assert grid_dartboard.score(Point2d(0, 66.7)) == 4
        assert grid_dartboard.score(Point2d(0, 99.9)) == 4
        assert grid_dartboard.score(Point2d(49.9, 66.7)) == 4
        assert grid_dartboard.score(Point2d(49.9, 99.9)) == 4

        assert grid_dartboard.score(Point2d(50.1, 0)) == 1
        assert grid_dartboard.score(Point2d(50.1, 33.33)) == 1
        assert grid_dartboard.score(Point2d(99.9, 0)) == 1
        assert grid_dartboard.score(Point2d(99.9, 33.33)) == 1

        assert grid_dartboard.score(Point2d(50.1, 33.34))
