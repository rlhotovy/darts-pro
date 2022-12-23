from dataclasses import dataclass

_DEFAULT_ODER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
_DEFAULT_BULLSEYE_VALUE = 50
_DEFAULT_HALF_BULLSEYE = 25


@dataclass
class Target:
    value: int
    multiplier: int = 1
    is_bullseye: bool = False


@dataclass
class DartBoard:
    """
    Roughly "ordered" (by "slice index") list of targets. Then inside each "slice", we
    have room for doubles and triples
    """

    radial_targets: dict[int, list[Target]]
    bullseye_targets: list[Target]
    radial_values_order: list[int]

    @classmethod
    def get_default_dartboard(cls, double_bulls: bool) -> "DartBoard":
        radial_targets = {}
        for idx, value in enumerate(_DEFAULT_ODER):
            slice_targets = [Target(value, mult) for mult in range(1, 4)]
            radial_targets[idx] = slice_targets
        if double_bulls:
            bullseye_targets = [
                Target(_DEFAULT_HALF_BULLSEYE, 1, True),
                Target(_DEFAULT_HALF_BULLSEYE, 2, True),
            ]
        else:
            bullseye_targets = [Target(_DEFAULT_BULLSEYE_VALUE, 1)]

        return DartBoard(
            radial_targets=radial_targets,
            bullseye_targets=bullseye_targets,
            radial_values_order=_DEFAULT_ODER,
        )
