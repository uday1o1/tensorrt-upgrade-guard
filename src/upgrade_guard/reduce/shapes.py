"""Profile and concrete-shape reduction."""

from __future__ import annotations

from dataclasses import dataclass

from upgrade_guard.reduce.predicate import ProfilePredicate


@dataclass(frozen=True)
class ReducedProfileFailure:
    """Smallest one-dimension profile violation."""

    observed_shape: tuple[int, ...]
    minimum_shape: tuple[int, ...]
    optimum_shape: tuple[int, ...]
    maximum_shape: tuple[int, ...]
    violating_dimension: int
    direction: str


def reduce_profile_failure(predicate: ProfilePredicate) -> ReducedProfileFailure:
    """Keep one minimal out-of-profile dimension and collapse every other dimension."""

    violation = next(
        index
        for index, (observed, minimum, maximum) in enumerate(
            zip(
                predicate.observed_shape,
                predicate.minimum_shape,
                predicate.maximum_shape,
                strict=True,
            )
        )
        if observed < minimum or observed > maximum
    )
    reduced_minimum = list(predicate.minimum_shape)
    reduced_maximum = list(predicate.minimum_shape)
    reduced_observed = list(predicate.minimum_shape)
    observed = predicate.observed_shape[violation]
    minimum = predicate.minimum_shape[violation]
    maximum = predicate.maximum_shape[violation]
    if observed > maximum:
        reduced_maximum[violation] = maximum
        reduced_observed[violation] = maximum + 1
        direction = "above_maximum"
    else:
        reduced_maximum[violation] = maximum
        reduced_observed[violation] = max(1, minimum - 1)
        direction = "below_minimum"
    return ReducedProfileFailure(
        tuple(reduced_observed),
        tuple(reduced_minimum),
        tuple(reduced_minimum),
        tuple(reduced_maximum),
        violation,
        direction,
    )
