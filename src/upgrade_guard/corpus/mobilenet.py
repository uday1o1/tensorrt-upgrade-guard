"""Pinned external MobileNet materialization and dynamic-spatial derivation."""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx

from upgrade_guard.contracts.base import sha256_bytes, sha256_file
from upgrade_guard.errors import InvalidInputError

SOURCE_REVISION = "4f43949841cb55a0b98dc8fcd045431ccafd9f96"
SOURCE_PATH = (
    "Computer_Vision/mobilenetv3_small_075_Opset17_timm/mobilenetv3_small_075_Opset17.onnx"
)
SOURCE_URL = (
    f"https://media.githubusercontent.com/media/onnx/models/{SOURCE_REVISION}/{SOURCE_PATH}"
)
SOURCE_SHA256 = "sha256:ef7b5191b3e2586c409ddcfbfef42a6434a9ac885608b8d16e9c767c518f1c31"
SOURCE_BYTES = 8_179_614
IMAGE_PREPROCESSING = "imagenet-rgb-nearest-v1"
IMAGE_FIXTURES = {
    "image-gradient": (
        Path("models/assets/gradient.ppm"),
        "sha256:4dd63cb1a6810501580d1e3a1f1e559d3b253d66298e67d7d53eee7b9466e8fb",
    ),
    "image-checkerboard": (
        Path("models/assets/checkerboard.ppm"),
        "sha256:7025a995b94ea8a2f7b9bd3a181965751eac157ac663edb1b18aae9cb44144aa",
    ),
}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class DerivedMobileNet:
    """Source and derived immutable identities."""

    source_sha256: str
    derived_sha256: str
    derived_bytes: int
    input_name: str
    output_names: tuple[str, ...]
    opset: int
    ir_version: int


@dataclass(frozen=True)
class PreprocessedImage:
    """One hash-bound PPM fixture and its deterministic model tensor."""

    source_sha256: str
    tensor_sha256: str
    preprocessing: str
    values: np.ndarray


class _PinnedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, request, file_pointer, code, message, headers, new_url
    ):
        _validate_download_url(new_url)
        return super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )


def download_source(destination: Path) -> None:
    """Download only the pinned official object and verify it before publication."""

    _validate_download_url(SOURCE_URL)
    request = urllib.request.Request(  # noqa: S310
        SOURCE_URL,
        headers={"User-Agent": "UpgradeGuard/0.1"},
    )
    opener = urllib.request.build_opener(_PinnedRedirectHandler())
    digest = hashlib.sha256()
    observed_bytes = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    try:
        with opener.open(request, timeout=120) as response, temporary.open("xb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                observed_bytes += len(chunk)
                if observed_bytes > SOURCE_BYTES:
                    raise InvalidInputError("MobileNet download exceeds pinned object size")
                digest.update(chunk)
                output.write(chunk)
        observed_hash = f"sha256:{digest.hexdigest()}"
        if observed_bytes != SOURCE_BYTES or observed_hash != SOURCE_SHA256:
            raise InvalidInputError(
                "MobileNet source identity differs from pinned Git LFS object",
                details={
                    "expected_sha256": SOURCE_SHA256,
                    "observed_sha256": observed_hash,
                    "expected_bytes": SOURCE_BYTES,
                    "observed_bytes": observed_bytes,
                },
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def derive_dynamic_mobilenet(source: Path, destination: Path) -> DerivedMobileNet:
    """Derive one frozen dynamic-batch and dynamic-spatial ONNX artifact."""

    if source.stat().st_size != SOURCE_BYTES or sha256_file(source) != SOURCE_SHA256:
        raise InvalidInputError("MobileNet source does not match the pinned object")
    model = onnx.load(source, load_external_data=False)
    if len(model.graph.input) != 1 or len(model.graph.output) < 1:
        raise InvalidInputError("MobileNet source graph has an unexpected input or output count")
    input_value = model.graph.input[0]
    input_dimensions = input_value.type.tensor_type.shape.dim
    if len(input_dimensions) != 4:
        raise InvalidInputError("MobileNet source input is not NCHW rank four")
    for index, name in ((0, "batch"), (2, "height"), (3, "width")):
        input_dimensions[index].ClearField("dim_value")
        input_dimensions[index].dim_param = name
    for output in model.graph.output:
        dimensions = output.type.tensor_type.shape.dim
        if dimensions:
            dimensions[0].ClearField("dim_value")
            dimensions[0].dim_param = "batch"
    model.producer_name = "tensorrt-upgrade-guard"
    model.producer_version = "0.1.0"
    model.doc_string = (
        "Dynamic-spatial derivative of pinned ONNX Models MobileNetV3 Small 0.75. "
        f"Source revision {SOURCE_REVISION}; source object {SOURCE_SHA256}."
    )
    onnx.checker.check_model(model, full_check=True)
    default_opsets = [item.version for item in model.opset_import if not item.domain]
    if len(default_opsets) != 1:
        raise InvalidInputError("MobileNet graph must declare one default ONNX opset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    return DerivedMobileNet(
        source_sha256=SOURCE_SHA256,
        derived_sha256=sha256_file(destination),
        derived_bytes=destination.stat().st_size,
        input_name=input_value.name,
        output_names=tuple(output.name for output in model.graph.output),
        opset=int(default_opsets[0]),
        ir_version=int(model.ir_version),
    )


def deterministic_image_input(batch: int, height: int, width: int) -> np.ndarray:
    """Create a reproducible numeric NCHW input without external image licensing."""

    if not (1 <= batch <= 16 and 160 <= height <= 320 and 160 <= width <= 320):
        raise InvalidInputError("MobileNet input shape is outside the locked profile")
    generator = np.random.Generator(
        np.random.PCG64(20260813 + batch * 1_000_000 + height * 1_000 + width)
    )
    return generator.uniform(-1.0, 1.0, size=(batch, 3, height, width)).astype(np.float32)


def preprocess_ppm_fixture(
    source: Path,
    expected_sha256: str,
    *,
    height: int = 224,
    width: int = 224,
) -> PreprocessedImage:
    """Decode, nearest-resize, and ImageNet-normalize one locked P3 PPM."""

    if height <= 0 or width <= 0:
        raise InvalidInputError("MobileNet image target dimensions must be positive")
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise InvalidInputError(
            "MobileNet image fixture hash changed",
            details={"expected_sha256": expected_sha256, "observed_sha256": observed_sha256},
        )
    pixels = _decode_p3_ppm(source)
    source_height, source_width, _ = pixels.shape
    y_indexes = (np.arange(height, dtype=np.int64) * source_height) // height
    x_indexes = (np.arange(width, dtype=np.int64) * source_width) // width
    resized = pixels[y_indexes[:, None], x_indexes[None, :], :].astype(np.float32)
    scaled = resized / np.float32(255.0)
    normalized = (scaled - IMAGENET_MEAN) / IMAGENET_STD
    values = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None, ...])
    return PreprocessedImage(
        source_sha256=observed_sha256,
        tensor_sha256=sha256_bytes(values.tobytes(order="C")),
        preprocessing=IMAGE_PREPROCESSING,
        values=values,
    )


def _decode_p3_ppm(path: Path) -> np.ndarray:
    try:
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise InvalidInputError("MobileNet PPM fixture exceeds the size limit")
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise InvalidInputError("MobileNet PPM fixture is not readable ASCII") from error
    tokens = []
    for line in text.splitlines():
        tokens.extend(line.partition("#")[0].split())
    if len(tokens) < 4 or tokens[0] != "P3":
        raise InvalidInputError("MobileNet image fixture must be an ASCII P3 PPM")
    try:
        width, height, maximum = (int(item) for item in tokens[1:4])
        samples = np.asarray([int(item) for item in tokens[4:]], dtype=np.int64)
    except ValueError as error:
        raise InvalidInputError("MobileNet PPM fixture contains invalid integers") from error
    if width <= 0 or height <= 0 or maximum != 255:
        raise InvalidInputError("MobileNet PPM header is outside the locked format")
    if samples.size != width * height * 3 or np.any((samples < 0) | (samples > maximum)):
        raise InvalidInputError("MobileNet PPM pixel inventory is invalid")
    return samples.astype(np.uint8).reshape(height, width, 3)


def _validate_download_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "media.githubusercontent.com":
        raise InvalidInputError("MobileNet download URL is outside the pinned official host")
