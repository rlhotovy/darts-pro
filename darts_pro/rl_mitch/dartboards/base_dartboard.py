from abc import ABC, abstractmethod

from darts_pro.rl_mitch.geometry import Point2d


class DartBoard(ABC):
    @property
    @abstractmethod
    def centre(self) -> Point2d:
        """The centre of the dartboard in mm. This is measured from the top left corner
        of the smallest enclosing rectangle.

        Returns
        -------
        Point2d
            The centre of the dartboard in mm.
        """

    @abstractmethod
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

    def normalise_shot_location(self, shot_location: Point2d) -> Point2d:
        """Normalise a shot location to be measured from the centre of the dartboard.

        Parameters
        ----------
        shot_location : Point2d
            The location of the dart shot. This is measured from the top left corner of
            the smallest enclosing rectangle.

        Returns
        -------
        Point2d
            The location of the dart shot. This is measured from the centre of the
            dartboard.
        """
        return shot_location - self.centre
