"""Benchmark-block environmental validity policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidityObservation:
    """Observed hardware state around one benchmark block."""

    gpu_uuid: str
    expected_gpu_uuid: str
    temperature_celsius: float | None
    maximum_temperature_celsius: float
    utilization_percent_before: float | None
    maximum_idle_utilization_percent: float
    competing_compute_processes: tuple[str, ...]
    graphics_clock_mhz: int | None
    minimum_graphics_clock_mhz: int | None


def rejection_reasons(observation: ValidityObservation) -> tuple[str, ...]:
    """Return every reason a measurement block must be discarded."""

    reasons: list[str] = []
    if observation.gpu_uuid != observation.expected_gpu_uuid:
        reasons.append("gpu_uuid_changed")
    if (
        observation.temperature_celsius is not None
        and observation.temperature_celsius > observation.maximum_temperature_celsius
    ):
        reasons.append("temperature_limit_exceeded")
    if (
        observation.utilization_percent_before is not None
        and observation.utilization_percent_before > observation.maximum_idle_utilization_percent
    ):
        reasons.append("gpu_not_idle_before_block")
    if observation.competing_compute_processes:
        reasons.append("competing_compute_process")
    if (
        observation.graphics_clock_mhz is not None
        and observation.minimum_graphics_clock_mhz is not None
        and observation.graphics_clock_mhz < observation.minimum_graphics_clock_mhz
    ):
        reasons.append("graphics_clock_below_policy")
    return tuple(reasons)
