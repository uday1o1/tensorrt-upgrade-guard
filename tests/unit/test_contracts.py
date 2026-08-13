"""Strict contract and hashing tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.factories import available_tool
from upgrade_guard.contracts.base import canonical_json_bytes, model_sha256
from upgrade_guard.contracts.environment import ToolObservation
from upgrade_guard.contracts.matrix import MatrixSpec


def matrix_payload() -> dict[str, object]:
    return {
        "api_version": "upgradeguard.dev/v1alpha1",
        "kind": "EnvironmentMatrix",
        "gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111",
        "environments": [
            {
                "id": "baseline",
                "base_image": "registry.example/tensorrt/base:v1",
                "worker_image": "registry.example/upgrade/worker:v1",
            },
            {
                "id": "candidate",
                "base_image": "registry.example/tensorrt/base:v2",
                "worker_image": "registry.example/upgrade/worker:v2",
            },
        ],
    }


def test_matrix_rejects_unknown_fields() -> None:
    payload = matrix_payload()
    payload["surprise"] = True
    with pytest.raises(ValidationError, match="surprise"):
        MatrixSpec.model_validate(payload)


def test_matrix_requires_exact_ordered_pair() -> None:
    payload = matrix_payload()
    payload["environments"] = payload["environments"][:1]  # type: ignore[index]
    with pytest.raises(ValidationError, match="exactly two"):
        MatrixSpec.model_validate(payload)


def test_matrix_rejects_duplicate_environment_ids() -> None:
    payload = matrix_payload()
    environments = payload["environments"]
    assert isinstance(environments, list)
    environments[1]["id"] = "baseline"
    with pytest.raises(ValidationError, match="unique"):
        MatrixSpec.model_validate(payload)


def test_matrix_requires_distinct_final_worker() -> None:
    payload = matrix_payload()
    environments = payload["environments"]
    assert isinstance(environments, list)
    environments[0]["worker_image"] = environments[0]["base_image"]
    with pytest.raises(ValidationError, match="separately derived"):
        MatrixSpec.model_validate(payload)


def test_contracts_are_frozen() -> None:
    tool = available_tool()
    with pytest.raises(ValidationError, match="frozen"):
        tool.available = False


def test_tool_observation_fails_closed() -> None:
    with pytest.raises(ValidationError, match="path and version"):
        ToolObservation(available=True)
    with pytest.raises(ValidationError, match="cannot contain"):
        ToolObservation(available=False, path="/bin/tool")


def test_canonical_json_and_model_hash_are_stable() -> None:
    first = canonical_json_bytes({"b": 2, "a": 1})
    second = canonical_json_bytes({"a": 1, "b": 2})
    assert first == b'{"a":1,"b":2}'
    assert first == second
    assert model_sha256(available_tool()) == model_sha256(available_tool())
