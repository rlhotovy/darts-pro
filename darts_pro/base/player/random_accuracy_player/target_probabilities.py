from dataclasses import dataclass
from typing import Callable
from enum import Enum, auto

import numpy as np
from scipy.integrate import dblquad

# Derived from https://www.dimensions.com/element/dartboard
DIAMETER = 451
RADIUS = DIAMETER / 2
RING_WIDTH = 8
R_SINGLE_BULL = 32
R_DOUBLE_BULL = 12.7

R_CENTER = 0.0

R_OUTER_TRIPLE = 107
R_INNER_TRIPLE = R_OUTER_TRIPLE - RING_WIDTH
R_INNER_SINGLE = (R_INNER_TRIPLE + R_DOUBLE_BULL) / 2

R_TRIPLE_CENTER = (R_INNER_TRIPLE + R_OUTER_TRIPLE) / 2

R_OUTER_DOUBLE = 170
R_INNER_DOUBLE = 170 - RING_WIDTH

R_OUTER_SINGLE = (R_OUTER_TRIPLE + R_INNER_DOUBLE) / 2
R_DOUBLE_CENTER = (R_INNER_DOUBLE + R_OUTER_DOUBLE) / 2

R_DOUBLE_BULL_PCT = R_DOUBLE_BULL / RADIUS
R_SINGLE_BULL_PCT = R_SINGLE_BULL / RADIUS
R_INNER_TRIPLE_PCT = R_INNER_TRIPLE / RADIUS
R_OUTER_TRIPLE_PCT = R_OUTER_TRIPLE / RADIUS
R_INNER_DOUBLE_PCT = R_INNER_DOUBLE / RADIUS
R_OUTER_DOUBLE_PCT = R_OUTER_DOUBLE / RADIUS


def _get_integrand(mu_x, mu_y, sigma_x, sigma_y) -> Callable[[float, float], float]:
    def normal(radius, theta):
        normalizer = 1 / (2 * np.pi * sigma_x * sigma_y)
        x_hat = (radius * np.cos(theta) - mu_x) / sigma_x
        y_hat = (radius * np.sin(theta) - mu_y) / sigma_y
        return (
            normalizer
            * radius
            * np.exp((-1 / 2) * (np.square(x_hat) + np.square(y_hat)))
        )

    return normal


class AimPoints(Enum):
    BULLSEYE = auto()
    SINGLE = auto()
    DOUBLE = auto()
    TRIPLE = auto()


@dataclass
class RadialProbabiltyResult:
    inner_single_percentage: float
    triple_percentage: float
    outer_single_percentage: float
    double_percentage: float


@dataclass
class ProbabilityComputationResult:
    sigma_x: float
    sigma_y: float
    double_bull_percentage: float
    single_bull_percentage: float
    miss_percentage: float
    radial_target_percentages: dict[int, RadialProbabiltyResult]


def _compute_lookup_for_target(
    target: float, sigma_x: float, sigma_y: float, n_radial_targets: int
):
    sector_theta = 2 * np.pi / n_radial_targets
    p_total = 0.0
    integrand = _get_integrand(target, 0, sigma_x, sigma_y)
    p_double_bull, _ = dblquad(integrand, 0, 2 * np.pi, 0, R_DOUBLE_BULL_PCT)
    p_total += p_double_bull

    p_single_bull, _ = dblquad(
        integrand, 0, 2 * np.pi, R_DOUBLE_BULL_PCT, R_SINGLE_BULL_PCT
    )
    p_total += p_single_bull

    radial_probs = {}

    for offset in range(n_radial_targets):
        theta_start = -sector_theta / 2 + (offset * sector_theta)
        theta_end = sector_theta / 2 + (offset * sector_theta)

        inner_single_pct, _ = dblquad(
            integrand, theta_start, theta_end, R_DOUBLE_BULL_PCT, R_INNER_TRIPLE_PCT
        )
        triple_pct, _ = dblquad(
            integrand, theta_start, theta_end, R_INNER_TRIPLE_PCT, R_OUTER_TRIPLE_PCT
        )
        outer_single_pct, _ = dblquad(
            integrand, theta_start, theta_end, R_OUTER_TRIPLE_PCT, R_INNER_DOUBLE_PCT
        )
        double_pct, _ = dblquad(
            integrand, theta_start, theta_end, R_INNER_DOUBLE_PCT, R_OUTER_DOUBLE_PCT
        )

        radial_probs[offset] = RadialProbabiltyResult(
            inner_single_percentage=inner_single_pct,
            triple_percentage=triple_pct,
            outer_single_percentage=outer_single_pct,
            double_percentage=double_pct,
        )

        p_total += inner_single_pct + triple_pct + outer_single_pct + double_pct

    return ProbabilityComputationResult(
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        double_bull_percentage=p_double_bull,
        single_bull_percentage=p_single_bull,
        miss_percentage=1 - p_total,
        radial_target_percentages=radial_probs,
    )


AIM_POINTS_TO_TARGET_RS = {
    AimPoints.BULLSEYE: R_CENTER / RADIUS,
    AimPoints.SINGLE: R_OUTER_SINGLE / RADIUS,
    AimPoints.DOUBLE: R_DOUBLE_CENTER / RADIUS,
    AimPoints.TRIPLE: R_TRIPLE_CENTER / RADIUS,
}


def compute_probability_lookup(
    sigma_x: float, sigma_y: float, n_radial_targets: int = 20
) -> dict[AimPoints, ProbabilityComputationResult]:
    result = {}
    for aim_target in AimPoints:
        target_r = R_CENTER / RADIUS
        if aim_target == AimPoints.SINGLE:
            target_r = R_OUTER_SINGLE / RADIUS
        elif aim_target == AimPoints.DOUBLE:
            target_r = R_DOUBLE_CENTER / RADIUS
        elif aim_target == AimPoints.TRIPLE:
            target_r = R_TRIPLE_CENTER / RADIUS
        result[aim_target] = _compute_lookup_for_target(
            target_r, sigma_x, sigma_y, n_radial_targets
        )

    return result
