"""Adversarial OCI descriptor and content-identity tests."""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Mapping
from io import BytesIO

import pytest

from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.errors import (
    InfrastructureError,
    InvalidInputError,
    UnsupportedEnvironmentError,
)
from upgrade_guard.matrix.digest import (
    CONFIG_MEDIA_TYPES,
    MANIFEST_ACCEPT,
    OCI_CONFIG,
    OCI_INDEX,
    OCI_MANIFEST,
    HttpResponse,
    RegistryClient,
    RegistryCredentials,
    ResolvedArtifact,
    UrllibRegistryTransport,
    _json_object,
    _read_bounded,
    _required_digest,
    _safe_token_realm,
    _select_linux_amd64_descriptor,
    _verified_digest,
    _verified_media_type,
    credentials_from_environment,
    parse_image_reference,
)


class QueueTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, Mapping[str, str], int]] = []

    def get(self, url: str, headers: Mapping[str, str], *, max_bytes: int) -> HttpResponse:
        self.requests.append((url, dict(headers), max_bytes))
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        return self.responses.pop(0)


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def response(body: bytes, media_type: str, *, digest: str | None = None) -> HttpResponse:
    return HttpResponse(
        200,
        {
            "Content-Type": media_type,
            "Docker-Content-Digest": digest or sha256_bytes(body),
        },
        body,
    )


def image_documents(
    *,
    duplicate_platform: bool = False,
    config_os: str = "linux",
    config_architecture: str = "amd64",
) -> tuple[bytes, bytes, bytes]:
    config = json_bytes(
        {
            "architecture": config_architecture,
            "os": config_os,
            "config": {
                "Labels": {
                    "com.udayarora.upgradeguard.base.manifest.digest": "sha256:" + ("a" * 64)
                }
            },
        }
    )
    manifest = json_bytes(
        {
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST,
            "config": {
                "mediaType": OCI_CONFIG,
                "digest": sha256_bytes(config),
                "size": len(config),
            },
            "layers": [],
        }
    )
    descriptor = {
        "mediaType": OCI_MANIFEST,
        "digest": sha256_bytes(manifest),
        "size": len(manifest),
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    manifests = [descriptor, {**descriptor, "platform": {"os": "linux", "architecture": "arm64"}}]
    if duplicate_platform:
        manifests.append(dict(descriptor))
    index = json_bytes(
        {
            "schemaVersion": 2,
            "mediaType": OCI_INDEX,
            "manifests": manifests,
        }
    )
    return index, manifest, config


def test_resolves_index_child_manifest_and_config_as_distinct_identities() -> None:
    index, manifest, config = image_documents()
    transport = QueueTransport(
        [
            response(index, OCI_INDEX),
            response(manifest, OCI_MANIFEST),
            response(config, OCI_CONFIG),
        ]
    )
    artifact = RegistryClient(transport).resolve_linux_amd64(
        "registry.example/nvidia/tensorrt:11.2"
    )
    assert artifact.image.index_digest == sha256_bytes(index)
    assert artifact.image.manifest_digest == sha256_bytes(manifest)
    assert artifact.image.config_digest == sha256_bytes(config)
    assert (
        len(
            {
                artifact.image.index_digest,
                artifact.image.manifest_digest,
                artifact.image.config_digest,
            }
        )
        == 3
    )
    assert artifact.image.canonical_reference.endswith(f"@{sha256_bytes(manifest)}")
    assert artifact.label("com.udayarora.upgradeguard.base.manifest.digest") == (
        "sha256:" + ("a" * 64)
    )
    assert transport.requests[0][1]["Accept"] == MANIFEST_ACCEPT


def test_single_manifest_has_no_invented_index_identity() -> None:
    _, manifest, config = image_documents()
    transport = QueueTransport(
        [
            response(manifest, OCI_MANIFEST),
            response(config, OCI_CONFIG),
        ]
    )
    artifact = RegistryClient(transport).resolve_linux_amd64(
        "registry.example/nvidia/tensorrt:single"
    )
    assert artifact.image.index_digest is None
    assert artifact.image.manifest_digest == sha256_bytes(manifest)


def test_duplicate_linux_amd64_descriptors_fail_closed() -> None:
    index, _, _ = image_documents(duplicate_platform=True)
    with pytest.raises(UnsupportedEnvironmentError, match="exactly one"):
        RegistryClient(QueueTransport([response(index, OCI_INDEX)])).resolve_linux_amd64(
            "registry.example/nvidia/tensorrt:duplicate"
        )


@pytest.mark.parametrize(
    ("operating_system", "architecture"),
    [("windows", "amd64"), ("linux", "arm64")],
)
def test_single_manifest_wrong_platform_fails_closed(
    operating_system: str,
    architecture: str,
) -> None:
    _, manifest, config = image_documents(
        config_os=operating_system,
        config_architecture=architecture,
    )
    transport = QueueTransport([response(manifest, OCI_MANIFEST), response(config, OCI_CONFIG)])
    with pytest.raises(UnsupportedEnvironmentError, match="not linux/amd64"):
        RegistryClient(transport).resolve_linux_amd64(
            "registry.example/nvidia/tensorrt:wrong-platform"
        )


def test_docker_content_digest_mismatch_is_rejected() -> None:
    index, _, _ = image_documents()
    bad_digest = "sha256:" + ("f" * 64)
    with pytest.raises(InvalidInputError, match="does not match"):
        RegistryClient(
            QueueTransport([response(index, OCI_INDEX, digest=bad_digest)])
        ).resolve_linux_amd64("registry.example/nvidia/tensorrt:tampered")


def test_requested_digest_mismatch_is_rejected() -> None:
    _, manifest, _ = image_documents()
    wrong = "sha256:" + ("f" * 64)
    with pytest.raises(InvalidInputError, match="requested digest"):
        RegistryClient(QueueTransport([response(manifest, OCI_MANIFEST)])).resolve_linux_amd64(
            f"registry.example/nvidia/tensorrt@{wrong}"
        )


def test_config_descriptor_digest_mismatch_is_rejected() -> None:
    _, manifest, config = image_documents()
    tampered = config + b" "
    transport = QueueTransport([response(manifest, OCI_MANIFEST), response(tampered, OCI_CONFIG)])
    with pytest.raises(InvalidInputError, match="requested digest"):
        RegistryClient(transport).resolve_linux_amd64("registry.example/nvidia/tensorrt:bad-config")


def test_descriptor_size_mismatch_is_rejected() -> None:
    _, manifest, config = image_documents()
    document = json.loads(manifest)
    document["config"]["size"] = len(config) + 1
    wrong_size_manifest = json_bytes(document)
    transport = QueueTransport(
        [
            response(wrong_size_manifest, OCI_MANIFEST),
            response(config, OCI_CONFIG),
        ]
    )
    with pytest.raises(InvalidInputError, match="descriptor size"):
        RegistryClient(transport).resolve_linux_amd64("registry.example/nvidia/tensorrt:bad-size")


def test_unsupported_media_type_is_rejected() -> None:
    body = json_bytes(
        {
            "schemaVersion": 1,
            "mediaType": "application/vnd.docker.distribution.manifest.v1+json",
        }
    )
    with pytest.raises(UnsupportedEnvironmentError, match="unsupported OCI media type"):
        RegistryClient(
            QueueTransport(
                [
                    response(
                        body,
                        "application/vnd.docker.distribution.manifest.v1+json",
                    )
                ]
            )
        ).resolve_linux_amd64("registry.example/nvidia/tensorrt:legacy")


def test_content_type_must_match_document_media_type() -> None:
    _, manifest, _ = image_documents()
    with pytest.raises(InvalidInputError, match="Content-Type"):
        RegistryClient(QueueTransport([response(manifest, OCI_INDEX)])).resolve_linux_amd64(
            "registry.example/nvidia/tensorrt:mismatch"
        )


def test_bearer_authentication_retries_without_serializing_secret() -> None:
    _, manifest, config = image_documents()
    challenge = HttpResponse(
        401,
        {
            "WWW-Authenticate": (
                'Bearer realm="https://auth.registry.example/token",'
                'service="registry.example",scope="repository:nvidia/tensorrt:pull"'
            )
        },
        b"",
    )
    token = HttpResponse(200, {"Content-Type": "application/json"}, b'{"token":"short-lived"}')
    transport = QueueTransport(
        [challenge, token, response(manifest, OCI_MANIFEST), response(config, OCI_CONFIG)]
    )
    client = RegistryClient(
        transport,
        credentials=RegistryCredentials(
            "registry.example",
            "user",
            "secret",
            "auth.registry.example",
        ),
    )
    artifact = client.resolve_linux_amd64("registry.example/nvidia/tensorrt:private")
    assert artifact.image.authored_reference.endswith(":private")
    assert transport.requests[1][1]["Authorization"].startswith("Basic ")
    assert transport.requests[2][1]["Authorization"] == "Bearer short-lived"
    assert "secret" not in artifact.image.model_dump_json()


@pytest.mark.parametrize(
    ("authored", "registry", "repository", "tag"),
    [
        ("alpine", "docker.io", "library/alpine", "latest"),
        ("nvidia/cuda:13.0", "docker.io", "nvidia/cuda", "13.0"),
        ("nvcr.io/nvidia/tensorrt:26.07", "nvcr.io", "nvidia/tensorrt", "26.07"),
        ("localhost:5000/team/image:v1", "localhost:5000", "team/image", "v1"),
    ],
)
def test_reference_parsing(
    authored: str,
    registry: str,
    repository: str,
    tag: str,
) -> None:
    parsed = parse_image_reference(authored)
    assert (parsed.registry, parsed.repository, parsed.tag) == (registry, repository, tag)


def test_local_registry_uses_http_only_for_loopback() -> None:
    _, manifest, config = image_documents()
    transport = QueueTransport([response(manifest, OCI_MANIFEST), response(config, OCI_CONFIG)])
    RegistryClient(transport).resolve_linux_amd64("127.0.0.1:5000/team/worker:v1")
    assert transport.requests[0][0].startswith("http://127.0.0.1:5000/")


def test_reference_parser_rejects_scheme_whitespace_and_bad_digest() -> None:
    with pytest.raises(InvalidInputError):
        parse_image_reference("https://registry.example/team/image:v1")
    with pytest.raises(InvalidInputError):
        parse_image_reference("registry.example/team/image:bad tag")
    with pytest.raises(InvalidInputError):
        parse_image_reference("registry.example/team/image@sha256:ABC")
    with pytest.raises(InvalidInputError):
        parse_image_reference("registry.example/team/image?redirect=evil:v1")
    with pytest.raises(InvalidInputError):
        parse_image_reference("registry.example:99999/team/image:v1")
    with pytest.raises(InvalidInputError):
        parse_image_reference("registry.example/team/../image:v1")


def test_reference_with_tag_and_digest_preserves_both() -> None:
    requested = "sha256:" + ("a" * 64)
    parsed = parse_image_reference(f"registry.example/team/image:v1@{requested}")
    assert parsed.tag == "v1"
    assert parsed.digest == requested
    assert parsed.selector == requested


def test_partial_environment_credentials_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPGRADE_GUARD_REGISTRY_HOST", "registry.example")
    monkeypatch.delenv("UPGRADE_GUARD_REGISTRY_USERNAME", raising=False)
    monkeypatch.delenv("UPGRADE_GUARD_REGISTRY_TOKEN", raising=False)
    with pytest.raises(InvalidInputError, match="complete set"):
        credentials_from_environment("registry.example")


def test_config_media_type_inventory_is_narrow() -> None:
    assert CONFIG_MEDIA_TYPES == {
        "application/vnd.oci.image.config.v1+json",
        "application/vnd.docker.container.image.v1+json",
    }


def test_artifact_label_missing_shapes_return_none() -> None:
    image = (
        RegistryClient(
            QueueTransport(
                [
                    response(image_documents()[1], OCI_MANIFEST),
                    response(image_documents()[2], OCI_CONFIG),
                ]
            )
        )
        .resolve_linux_amd64("registry.example/team/image:v1")
        .image
    )
    assert ResolvedArtifact(image, {}).label("missing") is None
    assert ResolvedArtifact(image, {"config": []}).label("missing") is None
    assert ResolvedArtifact(image, {"config": {"Labels": []}}).label("missing") is None


def test_registry_status_and_authentication_failures() -> None:
    with pytest.raises(InfrastructureError, match="HTTP 500"):
        RegistryClient(QueueTransport([HttpResponse(500, {}, b"failure")])).resolve_linux_amd64(
            "registry.example/team/image:v1"
        )
    with pytest.raises(InfrastructureError, match="Bearer"):
        RegistryClient(QueueTransport([HttpResponse(401, {}, b"")])).resolve_linux_amd64(
            "registry.example/team/image:v1"
        )
    unsafe = HttpResponse(
        401,
        {"WWW-Authenticate": 'Bearer realm="http://evil.example/token"'},
        b"",
    )
    with pytest.raises(InfrastructureError, match="unsafe token realm"):
        RegistryClient(QueueTransport([unsafe])).resolve_linux_amd64(
            "registry.example/team/image:v1"
        )


def test_token_response_and_credential_scope_failures() -> None:
    challenge = HttpResponse(
        401,
        {"WWW-Authenticate": 'Bearer realm="https://auth.registry.example/token"'},
        b"",
    )
    with pytest.raises(InvalidInputError, match="do not match"):
        RegistryClient(
            QueueTransport([challenge]),
            credentials=RegistryCredentials("other.example", "user", "secret"),
        ).resolve_linux_amd64("registry.example/team/image:v1")

    with pytest.raises(InfrastructureError, match="token request failed"):
        RegistryClient(QueueTransport([challenge, HttpResponse(403, {}, b"")])).resolve_linux_amd64(
            "registry.example/team/image:v1"
        )

    with pytest.raises(InfrastructureError, match="did not contain"):
        RegistryClient(
            QueueTransport([challenge, HttpResponse(200, {}, b"{}")])
        ).resolve_linux_amd64("registry.example/team/image:v1")

    with pytest.raises(InfrastructureError, match="not explicitly trusted"):
        RegistryClient(
            QueueTransport([challenge]),
            credentials=RegistryCredentials("registry.example", "user", "secret"),
        ).resolve_linux_amd64("registry.example/team/image:v1")


def test_second_authentication_rejection_is_explicit() -> None:
    challenge = HttpResponse(
        401,
        {"WWW-Authenticate": 'Bearer realm="https://auth.registry.example/token"'},
        b"",
    )
    transport = QueueTransport(
        [
            challenge,
            HttpResponse(200, {}, b'{"access_token":"token"}'),
            HttpResponse(401, {}, b""),
        ]
    )
    with pytest.raises(InfrastructureError, match="authentication is required"):
        RegistryClient(transport).resolve_linux_amd64("registry.example/team/image:v1")


def test_manifest_shape_failures() -> None:
    _, manifest, config = image_documents()
    missing_config = json_bytes({"schemaVersion": 2, "mediaType": OCI_MANIFEST, "layers": []})
    with pytest.raises(InvalidInputError, match="config descriptor"):
        RegistryClient(
            QueueTransport([response(missing_config, OCI_MANIFEST)])
        ).resolve_linux_amd64("registry.example/team/image:v1")

    document = json.loads(manifest)
    document["config"]["mediaType"] = "application/unknown"
    bad_media = json_bytes(document)
    with pytest.raises(UnsupportedEnvironmentError, match="config media type"):
        RegistryClient(QueueTransport([response(bad_media, OCI_MANIFEST)])).resolve_linux_amd64(
            "registry.example/team/image:v1"
        )

    child_index = json_bytes({"schemaVersion": 2, "mediaType": OCI_INDEX, "manifests": []})
    index = json_bytes(
        {
            "schemaVersion": 2,
            "mediaType": OCI_INDEX,
            "manifests": [
                {
                    "mediaType": OCI_INDEX,
                    "digest": sha256_bytes(child_index),
                    "size": len(child_index),
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        }
    )
    transport = QueueTransport(
        [
            response(index, OCI_INDEX),
            response(child_index, OCI_INDEX),
        ]
    )
    with pytest.raises(InvalidInputError, match="not an image manifest"):
        RegistryClient(transport).resolve_linux_amd64("registry.example/team/image:v1")
    assert config


def test_low_level_descriptor_validation_failures() -> None:
    with pytest.raises(InvalidInputError, match="manifests array"):
        _select_linux_amd64_descriptor({})
    with pytest.raises(InvalidInputError, match="non-object"):
        _select_linux_amd64_descriptor({"manifests": [1]})
    with pytest.raises(UnsupportedEnvironmentError, match="exactly one"):
        _select_linux_amd64_descriptor(
            {"manifests": [{"platform": None, "digest": "sha256:" + ("a" * 64)}]}
        )
    with pytest.raises(InvalidInputError, match="invalid digest"):
        _required_digest({"digest": "bad"}, "descriptor")
    with pytest.raises(InvalidInputError, match="not valid JSON"):
        _json_object(b"{", "document")
    with pytest.raises(InvalidInputError, match="JSON object"):
        _json_object(b"[]", "document")
    with pytest.raises(InvalidInputError, match="missing mediaType"):
        _verified_media_type(HttpResponse(200, {}, b"{}"), {})
    with pytest.raises(InvalidInputError, match="malformed"):
        _verified_digest(
            HttpResponse(200, {"Docker-Content-Digest": "bad"}, b"body"),
            requested=None,
        )


def test_bounded_reader_and_url_safety() -> None:
    assert _read_bounded(BytesIO(b"ok"), 2) == b"ok"
    with pytest.raises(InfrastructureError, match="size limit"):
        _read_bounded(BytesIO(b"too large"), 2)
    assert _safe_token_realm("https://auth.example/token", "registry.example")
    assert _safe_token_realm("http://localhost/token", "localhost:5000")
    assert not _safe_token_realm("http://auth.example/token", "registry.example")


def test_urllib_transport_success_and_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, amount: int = -1) -> bytes:
            del amount
            return b"{}"

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())
    result = UrllibRegistryTransport().get(
        "https://registry.example/v2/",
        {},
        max_bytes=100,
    )
    assert result.status == 200

    def network_error(request: object, timeout: float) -> object:
        del request, timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", network_error)
    with pytest.raises(InfrastructureError, match="request failed"):
        UrllibRegistryTransport().get(
            "https://registry.example/v2/?secret=no",
            {},
            max_bytes=100,
        )


def test_complete_environment_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPGRADE_GUARD_REGISTRY_HOST", "registry.example")
    monkeypatch.setenv("UPGRADE_GUARD_REGISTRY_USERNAME", "user")
    monkeypatch.setenv("UPGRADE_GUARD_REGISTRY_TOKEN", "secret")
    credentials = credentials_from_environment("registry.example")
    assert credentials is not None
    assert credentials.username == "user"
    with pytest.raises(InvalidInputError, match="do not match"):
        credentials_from_environment("other.example")
