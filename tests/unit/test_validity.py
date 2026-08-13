"""Environmental rejection policy tests."""

from upgrade_guard.compare.validity import ValidityObservation, rejection_reasons


def test_every_observed_validity_violation_is_retained() -> None:
    reasons = rejection_reasons(
        ValidityObservation(
            gpu_uuid="GPU-wrong",
            expected_gpu_uuid="GPU-expected",
            temperature_celsius=90,
            maximum_temperature_celsius=80,
            utilization_percent_before=50,
            maximum_idle_utilization_percent=5,
            competing_compute_processes=("pid=123",),
            graphics_clock_mhz=1000,
            minimum_graphics_clock_mhz=1500,
        )
    )
    assert reasons == (
        "gpu_uuid_changed",
        "temperature_limit_exceeded",
        "gpu_not_idle_before_block",
        "competing_compute_process",
        "graphics_clock_below_policy",
    )
