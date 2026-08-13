"""Content-verified OCI Distribution image resolution."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from upgrade_guard.contracts.base import sha256_bytes
from upgrade_guard.contracts.environment import PlatformIdentity, ResolvedImage
from upgrade_guard.errors import InfrastructureError, InvalidInputError, UnsupportedEnvironmentError

OCI_INDEX = "application/vnd.oci.image.index.v1+json"
DOCKER_INDEX = "application/vnd.docker.distribution.manifest.list.v2+json"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
INDEX_MEDIA_TYPES = frozenset({OCI_INDEX, DOCKER_INDEX})
MANIFEST_MEDIA_TYPES = frozenset({OCI_MANIFEST, DOCKER_MANIFEST})
CONFIG_MEDIA_TYPES = frozenset({OCI_CONFIG, DOCKER_CONFIG})
MANIFEST_ACCEPT = ", ".join((*INDEX_MEDIA_TYPES, *MANIFEST_MEDIA_TYPES))
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHALLENGE_FIELD = re.compile(r'([A-Za-z][A-Za-z0-9_-]*)="([^"]*)"')
_REGISTRY_PATTERN = re.compile(r"^(?:localhost|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?$")
_REPOSITORY_COMPONENT_PATTERN = re.compile(r"^[a-z0-9]+(?:(?:[._]|__|[-]+)[a-z0-9]+)*$")
_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ImageReferenceParts:
    """Parsed Docker-compatible image reference."""

    authored: str
    registry: str
    repository: str
    selector: str
    tag: str | None
    digest: str | None

    @property
    def endpoint_registry(self) -> str:
        if self.registry in {"docker.io", "index.docker.io"}:
            return "registry-1.docker.io"
        return self.registry


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Bounded registry response."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class RegistryTransport(Protocol):
    """Injectable HTTPS transport."""

    def get(self, url: str, headers: Mapping[str, str], *, max_bytes: int) -> HttpResponse: ...


class ReadableResponse(Protocol):
    """Minimal response surface used by the bounded reader."""

    def read(self, amount: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class RegistryCredentials:
    """In-memory credentials that are never serialized into a lock."""

    registry: str
    username: str
    secret: str
    auth_host: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """Public lock identity plus verified OCI configuration metadata."""

    image: ResolvedImage
    config: Mapping[str, object]

    def label(self, name: str) -> str | None:
        """Read one string label from the verified image configuration."""

        config_value = self.config.get("config")
        if not isinstance(config_value, dict):
            return None
        labels = config_value.get("Labels")
        if not isinstance(labels, dict):
            return None
        value = labels.get(name)
        return value if isinstance(value, str) else None


class UrllibRegistryTransport:
    """Small HTTPS transport with strict response-size limits."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def get(self, url: str, headers: Mapping[str, str], *, max_bytes: int) -> HttpResponse:
        request = urllib.request.Request(  # noqa: S310
            url, headers=dict(headers), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                body = _read_bounded(response, max_bytes)
                return HttpResponse(response.status, dict(response.headers.items()), body)
        except urllib.error.HTTPError as error:
            body = _read_bounded(error, max_bytes)
            return HttpResponse(error.code, dict(error.headers.items()), body)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise InfrastructureError(
                "OCI registry request failed",
                details={"registry_url": _redact_url(url), "reason": str(error)},
            ) from error


class RegistryClient:
    """Resolve one reference to exact raw OCI descriptors and bytes."""

    def __init__(
        self,
        transport: RegistryTransport | None = None,
        *,
        credentials: RegistryCredentials | None = None,
    ) -> None:
        self.transport = transport or UrllibRegistryTransport()
        self.credentials = credentials

    def resolve_linux_amd64(self, authored_reference: str) -> ResolvedArtifact:
        """Resolve and byte-verify exactly one linux/amd64 image."""

        reference = parse_image_reference(authored_reference)
        response = self._get_registry_object(
            reference,
            kind="manifests",
            selector=reference.selector,
            accept=MANIFEST_ACCEPT,
            max_bytes=16 * 1024 * 1024,
        )
        root_digest = _verified_digest(response, requested=reference.digest)
        document = _json_object(response.body, "image manifest")
        media_type = _verified_media_type(response, document)

        index_digest: str | None = None
        if media_type in INDEX_MEDIA_TYPES:
            index_digest = root_digest
            descriptor = _select_linux_amd64_descriptor(document)
            child_digest = _required_digest(descriptor, "selected manifest descriptor")
            child_size = _required_size(descriptor, "selected manifest descriptor")
            child = self._get_registry_object(
                reference,
                kind="manifests",
                selector=child_digest,
                accept=", ".join(MANIFEST_MEDIA_TYPES),
                max_bytes=16 * 1024 * 1024,
            )
            manifest_digest = _verified_digest(child, requested=child_digest)
            _verify_size(child.body, child_size, "selected manifest")
            manifest = _json_object(child.body, "selected image manifest")
            manifest_media_type = _verified_media_type(child, manifest)
            if manifest_media_type not in MANIFEST_MEDIA_TYPES:
                raise InvalidInputError("selected index descriptor is not an image manifest")
        elif media_type in MANIFEST_MEDIA_TYPES:
            manifest_digest = root_digest
            manifest = document
            manifest_media_type = media_type
        else:
            raise UnsupportedEnvironmentError(
                f"unsupported OCI root media type: {media_type}",
                details={"image": authored_reference},
            )

        config_descriptor = manifest.get("config")
        if not isinstance(config_descriptor, dict):
            raise InvalidInputError("image manifest is missing its config descriptor")
        config_digest = _required_digest(config_descriptor, "image config descriptor")
        config_size = _required_size(config_descriptor, "image config descriptor")
        config_media_type = config_descriptor.get("mediaType")
        if config_media_type not in CONFIG_MEDIA_TYPES:
            raise UnsupportedEnvironmentError(
                f"unsupported OCI config media type: {config_media_type}",
                details={"image": authored_reference},
            )
        config_response = self._get_registry_object(
            reference,
            kind="blobs",
            selector=config_digest,
            accept=str(config_media_type),
            max_bytes=8 * 1024 * 1024,
        )
        _verified_digest(config_response, requested=config_digest)
        _verify_size(config_response.body, config_size, "image configuration")
        config = _json_object(config_response.body, "image configuration")
        _validate_config_platform(config)

        image = ResolvedImage(
            authored_reference=authored_reference,
            registry=reference.registry,
            repository=reference.repository,
            authored_tag=reference.tag,
            requested_digest=reference.digest,
            index_digest=index_digest,
            manifest_digest=manifest_digest,
            config_digest=config_digest,
            manifest_media_type=manifest_media_type,
            config_media_type=str(config_media_type),
            platform=PlatformIdentity(os="linux", architecture="amd64"),
        )
        return ResolvedArtifact(image=image, config=config)

    def _get_registry_object(
        self,
        reference: ImageReferenceParts,
        *,
        kind: str,
        selector: str,
        accept: str,
        max_bytes: int,
    ) -> HttpResponse:
        quoted_selector = urllib.parse.quote(selector, safe=":")
        scheme = _registry_scheme(reference.endpoint_registry)
        url = (
            f"{scheme}://{reference.endpoint_registry}/v2/{reference.repository}/"
            f"{kind}/{quoted_selector}"
        )
        headers = {"Accept": accept, "User-Agent": "tensorrt-upgrade-guard/0.1"}
        response = self.transport.get(url, headers, max_bytes=max_bytes)
        if response.status == 401:
            token = self._bearer_token(reference, response)
            headers["Authorization"] = f"Bearer {token}"
            response = self.transport.get(url, headers, max_bytes=max_bytes)
        if response.status == 401:
            raise InfrastructureError(
                "OCI registry authentication is required",
                details={"registry": reference.registry, "repository": reference.repository},
            )
        if response.status != 200:
            raise InfrastructureError(
                f"OCI registry returned HTTP {response.status}",
                details={"registry_url": _redact_url(url)},
            )
        return response

    def _bearer_token(self, reference: ImageReferenceParts, unauthorized: HttpResponse) -> str:
        challenge = _header(unauthorized.headers, "www-authenticate")
        if challenge is None or not challenge.lower().startswith("bearer "):
            raise InfrastructureError(
                "OCI registry did not provide a Bearer authentication challenge"
            )
        parameters = {key.lower(): value for key, value in _CHALLENGE_FIELD.findall(challenge)}
        realm = parameters.get("realm")
        if realm is None or not _safe_token_realm(realm, reference.endpoint_registry):
            raise InfrastructureError("OCI registry provided an unsafe token realm")
        query: dict[str, str] = {}
        if service := parameters.get("service"):
            query["service"] = service
        query["scope"] = parameters.get("scope", f"repository:{reference.repository}:pull")
        token_url = f"{realm}?{urllib.parse.urlencode(query)}"
        headers = {"Accept": "application/json", "User-Agent": "tensorrt-upgrade-guard/0.1"}
        credentials = self.credentials
        if credentials is not None:
            if credentials.registry not in {reference.registry, reference.endpoint_registry}:
                raise InvalidInputError("registry credentials do not match the requested registry")
            realm_host = urllib.parse.urlparse(realm).hostname
            allowed_auth_hosts = {
                reference.endpoint_registry.split(":", maxsplit=1)[0],
                credentials.auth_host,
            }
            if realm_host not in allowed_auth_hosts:
                raise InfrastructureError(
                    "OCI credential token realm is not explicitly trusted",
                    details={"realm_host": realm_host},
                )
            encoded = base64.b64encode(
                f"{credentials.username}:{credentials.secret}".encode()
            ).decode("ascii")
            headers["Authorization"] = f"Basic {encoded}"
        response = self.transport.get(token_url, headers, max_bytes=1024 * 1024)
        if response.status != 200:
            raise InfrastructureError(
                "OCI registry token request failed",
                details={"registry": reference.registry, "status": response.status},
            )
        payload = _json_object(response.body, "registry token")
        token = payload.get("token", payload.get("access_token"))
        if not isinstance(token, str) or not token:
            raise InfrastructureError("OCI registry token response did not contain a token")
        return token


def credentials_from_environment(registry: str) -> RegistryCredentials | None:
    """Load explicitly scoped credentials without logging or serializing them."""

    configured_registry = os.environ.get("UPGRADE_GUARD_REGISTRY_HOST")
    username = os.environ.get("UPGRADE_GUARD_REGISTRY_USERNAME")
    secret = os.environ.get("UPGRADE_GUARD_REGISTRY_TOKEN")
    auth_host = os.environ.get("UPGRADE_GUARD_REGISTRY_AUTH_HOST")
    if not any((configured_registry, username, secret, auth_host)):
        return None
    if not configured_registry or not username or not secret:
        raise InvalidInputError(
            "registry credential environment variables must be provided as a complete set"
        )
    if configured_registry != registry:
        raise InvalidInputError("configured registry credentials do not match the image registry")
    return RegistryCredentials(registry, username, secret, auth_host)


def parse_image_reference(authored: str) -> ImageReferenceParts:
    """Parse a registry reference without treating a mutable tag as identity."""

    if not authored or any(character.isspace() for character in authored):
        raise InvalidInputError("image references must be nonempty and contain no whitespace")
    if "://" in authored:
        raise InvalidInputError("image references must not include a URL scheme")
    name, separator, digest_part = authored.partition("@")
    last_slash = name.rfind("/")
    last_colon = name.rfind(":")
    authored_tag = name[last_colon + 1 :] if last_colon > last_slash else None
    if authored_tag is not None:
        name = name[:last_colon]
        if not _TAG_PATTERN.fullmatch(authored_tag):
            raise InvalidInputError("image tag has invalid syntax")
    if separator:
        if "@" in digest_part or not _DIGEST_PATTERN.fullmatch(digest_part):
            raise InvalidInputError("image digest must be a lowercase SHA-256 digest")
        digest: str | None = digest_part
        tag = authored_tag
        selector = digest_part
    else:
        tag = authored_tag or "latest"
        digest = None
        selector = tag

    components = name.split("/")
    if any(not component for component in components):
        raise InvalidInputError("image reference contains an empty path component")
    first = components[0]
    if len(components) > 1 and ("." in first or ":" in first or first == "localhost"):
        registry = first
        repository_components = components[1:]
    else:
        registry = "docker.io"
        repository_components = components
    if registry == "docker.io" and len(repository_components) == 1:
        repository_components.insert(0, "library")
    if not _REGISTRY_PATTERN.fullmatch(registry):
        raise InvalidInputError("image registry has invalid syntax")
    if ":" in registry:
        port = int(registry.rpartition(":")[2])
        if port > 65535:
            raise InvalidInputError("image registry port is out of range")
    if not all(_REPOSITORY_COMPONENT_PATTERN.fullmatch(part) for part in repository_components):
        raise InvalidInputError("image repository has invalid syntax")
    repository = "/".join(repository_components)
    if not repository:
        raise InvalidInputError("image repository cannot be empty")
    return ImageReferenceParts(authored, registry, repository, selector, tag, digest)


def _select_linux_amd64_descriptor(document: Mapping[str, object]) -> Mapping[str, object]:
    manifests = document.get("manifests")
    if not isinstance(manifests, list):
        raise InvalidInputError("OCI index is missing its manifests array")
    matches: list[Mapping[str, object]] = []
    for descriptor in manifests:
        if not isinstance(descriptor, dict):
            raise InvalidInputError("OCI index contains a non-object descriptor")
        platform = descriptor.get("platform")
        if not isinstance(platform, dict):
            continue
        if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
            matches.append(descriptor)
    if len(matches) != 1:
        raise UnsupportedEnvironmentError(
            "OCI index must contain exactly one linux/amd64 manifest",
            details={"matching_descriptors": len(matches)},
        )
    return matches[0]


def _verified_digest(response: HttpResponse, *, requested: str | None) -> str:
    computed = sha256_bytes(response.body)
    header_digest = _header(response.headers, "docker-content-digest")
    if header_digest is not None:
        if not _DIGEST_PATTERN.fullmatch(header_digest):
            raise InvalidInputError("registry returned a malformed Docker-Content-Digest")
        if header_digest != computed:
            raise InvalidInputError("registry response does not match Docker-Content-Digest")
    if requested is not None and requested != computed:
        raise InvalidInputError("registry response bytes do not match the requested digest")
    return computed


def _verified_media_type(response: HttpResponse, document: Mapping[str, object]) -> str:
    content_type_header = _header(response.headers, "content-type")
    content_type = (
        content_type_header.split(";", maxsplit=1)[0].strip() if content_type_header else None
    )
    declared = document.get("mediaType")
    if not isinstance(declared, str):
        raise InvalidInputError("OCI document is missing mediaType")
    if content_type is not None and content_type != declared:
        raise InvalidInputError("OCI Content-Type does not match the document mediaType")
    if declared not in INDEX_MEDIA_TYPES | MANIFEST_MEDIA_TYPES:
        raise UnsupportedEnvironmentError(f"unsupported OCI media type: {declared}")
    return declared


def _validate_config_platform(config: Mapping[str, object]) -> None:
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise UnsupportedEnvironmentError(
            "selected image configuration is not linux/amd64",
            details={"os": config.get("os"), "architecture": config.get("architecture")},
        )


def _required_digest(descriptor: Mapping[str, object], context: str) -> str:
    digest = descriptor.get("digest")
    if not isinstance(digest, str) or not _DIGEST_PATTERN.fullmatch(digest):
        raise InvalidInputError(f"{context} has an invalid digest")
    return digest


def _required_size(descriptor: Mapping[str, object], context: str) -> int:
    size = descriptor.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise InvalidInputError(f"{context} has an invalid size")
    return size


def _verify_size(body: bytes, expected: int, context: str) -> None:
    if len(body) != expected:
        raise InvalidInputError(
            f"{context} byte length does not match its descriptor size",
            details={"expected": expected, "observed": len(body)},
        )


def _json_object(body: bytes, context: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidInputError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict):
        raise InvalidInputError(f"{context} must be a JSON object")
    return value


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def _read_bounded(response: ReadableResponse, max_bytes: int) -> bytes:
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise InfrastructureError("OCI registry response exceeded the configured size limit")
    return body


def _safe_token_realm(realm: str, registry: str) -> bool:
    parsed = urllib.parse.urlparse(realm)
    if parsed.scheme == "https" and parsed.hostname:
        return True
    registry_host = registry.split(":", maxsplit=1)[0]
    return parsed.scheme == "http" and registry_host in {"localhost", "127.0.0.1"}


def _registry_scheme(registry: str) -> str:
    host = registry.split(":", maxsplit=1)[0]
    return "http" if host in {"localhost", "127.0.0.1"} else "https"


def _redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
