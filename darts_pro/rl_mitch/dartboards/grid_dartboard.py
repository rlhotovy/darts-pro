import numpy as np
from darts_pro.rl_mitch.dartboards.base_dartboard import DartBoard

from darts_pro.rl_mitch.geometry import Point2d, Size2d


class ScoreGrid:
    def __init__(self, score_grid: np.ndarray):
        self.score_grid = score_grid

    @property
    def rows(self) -> int:
        return self.score_grid.shape[0]

    @property
    def columns(self) -> int:
        return self.score_grid.shape[1]


class GridDartBoard(DartBoard):
    """A dartboard with a rectilinear grid of scores."""

    def __init__(self, size_mm: Size2d, score_grid: ScoreGrid):
        self.size_mm = size_mm
        self.score_grid = score_grid
        self.scoring_logic = GridDartBoardScoringLogic(score_grid, size_mm)

    def unique_scores(self) -> np.ndarray:
        """The unique scores on the dartboard.

        Returns
        -------
        np.ndarray
            The unique scores on the dartboard.
        """
        return np.sort(np.unique(self.score_grid.score_grid))

    @property
    def centre(self) -> Point2d:
        """The centre of the dartboard in mm. This is measured from the top left corner
        of the smallest enclosing rectangle.

        Returns
        -------
        Point2d
            The centre of the dartboard in mm.
        """
        return Point2d(self.size_mm.width_mm / 2, self.size_mm.height_mm / 2)

    def score(self, shot_location: Point2d) -> int:
        """The score of a dart shot at a given location. The location is measured from
        the top left corner of the smallest enclosing rectangle.

        Parameters
        ----------
        shot_location : Point2d
            The location of the dart shot. This is measured from the top left corner of
            the smallest enclosing rectangle.

        Returns
        -------
        int
            The score of the shot.
        """
        return self.scoring_logic.score(shot_location)


class GridDartBoardScoringLogic:
    """The scoring logic for a dartboard with a rectilinear grid of scores."""

    def __init__(self, score_grid: ScoreGrid, size_mm: Size2d):
        self.score_grid = score_grid
        self.size_mm = size_mm

    @property
    def vertical_segment_height(self) -> float:
        """The height of a vertical segment in mm.

        Returns
        -------
        float
            The height of a vertical segment in mm.
        """
        return self.size_mm.height_mm / self.score_grid.rows

    @property
    def horizontal_segment_width(self) -> float:
        """The width of a horizontal segment in mm.

        Returns
        -------
        float
            The width of a horizontal segment in mm.
        """
        return self.size_mm.width_mm / self.score_grid.columns

    def score(self, shot_location: Point2d) -> int:
        """The score of a dart shot at a given location. The location is measured from
        the top left corner of the smallest enclosing rectangle.

        Parameters
        ----------
        shot_location : Point2d
            The location of the dart shot. This is measured from the top left corner of
            the smallest enclosing rectangle.

        Returns
        -------
        int
            The score of the shot.
        """
        row = int(shot_location.y // self.vertical_segment_height)
        column = int(shot_location.x // self.horizontal_segment_width)
        return self.score_grid.score_grid[row, column]
