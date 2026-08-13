"""Deterministic project-owned ONNX model generators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from upgrade_guard.contracts.base import sha256_file

HIDDEN = 256
HEADS = 8
HEAD_WIDTH = HIDDEN // HEADS
FEED_FORWARD = 1024
BLOCKS = 4
OPSET = 17
SEED = 20260813


@dataclass(frozen=True)
class GeneratedModel:
    """Identity of a generated ONNX artifact."""

    path: Path
    sha256: str
    bytes: int
    precision: Literal["fp32", "fp16"]
    opset: int
    ir_version: int


class _GraphBuilder:
    def __init__(self, precision: Literal["fp32", "fp16"]) -> None:
        self.precision = precision
        self.numpy_dtype = np.float32 if precision == "fp32" else np.float16
        self.tensor_type = TensorProto.FLOAT if precision == "fp32" else TensorProto.FLOAT16
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.rng = np.random.Generator(np.random.PCG64(SEED))
        self._constant("reshape_qkv", np.asarray([0, 0, 3, HEADS, HEAD_WIDTH], np.int64))
        self._constant("reshape_context", np.asarray([0, 0, HIDDEN], np.int64))
        self._constant("split_qkv", np.asarray([1, 1, 1], np.int64))
        self._constant("axes_zero", np.asarray([0], np.int64))
        self._constant("sqrt_two", np.asarray(np.sqrt(2.0), self.numpy_dtype))
        self._constant("one", np.asarray(1.0, self.numpy_dtype))
        self._constant("half", np.asarray(0.5, self.numpy_dtype))
        self._constant("attention_scale", np.asarray(HEAD_WIDTH**-0.5, self.numpy_dtype))

    def _constant(self, name: str, value: np.ndarray) -> None:
        self.initializers.append(numpy_helper.from_array(value, name=name))

    def _weight(self, name: str, shape: tuple[int, ...], scale: float = 0.02) -> str:
        values = self.rng.normal(0.0, scale, size=shape).astype(self.numpy_dtype)
        self._constant(name, values)
        return name

    def _zeros(self, name: str, size: int) -> str:
        self._constant(name, np.zeros((size,), dtype=self.numpy_dtype))
        return name

    def _ones(self, name: str, size: int) -> str:
        self._constant(name, np.ones((size,), dtype=self.numpy_dtype))
        return name

    def _node(
        self,
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        *,
        name: str,
        **attributes: Any,
    ) -> None:
        self.nodes.append(helper.make_node(op_type, inputs, outputs, name=name, **attributes))

    def layer_norm(self, value: str, prefix: str) -> str:
        mean = f"{prefix}.mean"
        centered = f"{prefix}.centered"
        squared = f"{prefix}.squared"
        variance = f"{prefix}.variance"
        adjusted = f"{prefix}.adjusted"
        root = f"{prefix}.root"
        normalized = f"{prefix}.normalized"
        scaled = f"{prefix}.scaled"
        output = f"{prefix}.output"
        epsilon = f"{prefix}.epsilon"
        self._constant(epsilon, np.asarray(1e-5, self.numpy_dtype))
        gamma = self._ones(f"{prefix}.gamma", HIDDEN)
        beta = self._zeros(f"{prefix}.beta", HIDDEN)
        self._node("ReduceMean", [value], [mean], name=f"{prefix}.mean_node", axes=[-1], keepdims=1)
        self._node("Sub", [value, mean], [centered], name=f"{prefix}.center")
        self._node("Mul", [centered, centered], [squared], name=f"{prefix}.square")
        self._node(
            "ReduceMean",
            [squared],
            [variance],
            name=f"{prefix}.variance_node",
            axes=[-1],
            keepdims=1,
        )
        self._node("Add", [variance, epsilon], [adjusted], name=f"{prefix}.epsilon_node")
        self._node("Sqrt", [adjusted], [root], name=f"{prefix}.sqrt")
        self._node("Div", [centered, root], [normalized], name=f"{prefix}.normalize")
        self._node("Mul", [normalized, gamma], [scaled], name=f"{prefix}.scale")
        self._node("Add", [scaled, beta], [output], name=f"{prefix}.shift")
        return output

    def linear(self, value: str, prefix: str, input_width: int, output_width: int) -> str:
        multiplied = f"{prefix}.matmul"
        output = f"{prefix}.output"
        weight = self._weight(f"{prefix}.weight", (input_width, output_width))
        bias = self._zeros(f"{prefix}.bias", output_width)
        self._node("MatMul", [value, weight], [multiplied], name=f"{prefix}.matmul_node")
        self._node("Add", [multiplied, bias], [output], name=f"{prefix}.bias_node")
        return output

    def gelu(self, value: str, prefix: str) -> str:
        divided = f"{prefix}.divided"
        erf = f"{prefix}.erf"
        shifted = f"{prefix}.shifted"
        multiplied = f"{prefix}.multiplied"
        output = f"{prefix}.output"
        self._node("Div", [value, "sqrt_two"], [divided], name=f"{prefix}.divide")
        self._node("Erf", [divided], [erf], name=f"{prefix}.erf_node")
        self._node("Add", [erf, "one"], [shifted], name=f"{prefix}.add_one")
        self._node("Mul", [value, shifted], [multiplied], name=f"{prefix}.multiply")
        self._node("Mul", [multiplied, "half"], [output], name=f"{prefix}.half_node")
        return output

    def transformer_block(self, value: str, block: int) -> str:
        prefix = f"block{block}"
        normalized = self.layer_norm(value, f"{prefix}.attention_norm")
        qkv = self.linear(normalized, f"{prefix}.qkv", HIDDEN, HIDDEN * 3)
        reshaped = f"{prefix}.qkv_reshaped"
        transposed = f"{prefix}.qkv_transposed"
        split = [f"{prefix}.{name}_split" for name in ("q", "k", "v")]
        q_name, k_name, v_name = [f"{prefix}.{name}" for name in ("q", "k", "v")]
        self._node("Reshape", [qkv, "reshape_qkv"], [reshaped], name=f"{prefix}.reshape_qkv")
        self._node(
            "Transpose",
            [reshaped],
            [transposed],
            name=f"{prefix}.transpose_qkv",
            perm=[2, 0, 3, 1, 4],
        )
        self._node(
            "Split",
            [transposed, "split_qkv"],
            split,
            name=f"{prefix}.split_qkv",
            axis=0,
        )
        for source, target, label in zip(
            split, (q_name, k_name, v_name), ("q", "k", "v"), strict=True
        ):
            self._node(
                "Squeeze",
                [source, "axes_zero"],
                [target],
                name=f"{prefix}.squeeze_{label}",
            )
        k_transposed = f"{prefix}.k_transposed"
        scores = f"{prefix}.scores"
        scaled_scores = f"{prefix}.scaled_scores"
        masked_scores = f"{prefix}.masked_scores"
        probabilities = f"{prefix}.probabilities"
        context = f"{prefix}.context"
        context_transposed = f"{prefix}.context_transposed"
        context_reshaped = f"{prefix}.context_reshaped"
        self._node(
            "Transpose",
            [k_name],
            [k_transposed],
            name=f"{prefix}.transpose_k",
            perm=[0, 1, 3, 2],
        )
        self._node("MatMul", [q_name, k_transposed], [scores], name=f"{prefix}.attention_scores")
        self._node(
            "Mul",
            [scores, "attention_scale"],
            [scaled_scores],
            name=f"{prefix}.scale_attention",
        )
        self._node("Add", [scaled_scores, "mask"], [masked_scores], name=f"{prefix}.mask")
        self._node("Softmax", [masked_scores], [probabilities], name=f"{prefix}.softmax", axis=-1)
        self._node("MatMul", [probabilities, v_name], [context], name=f"{prefix}.context_node")
        self._node(
            "Transpose",
            [context],
            [context_transposed],
            name=f"{prefix}.transpose_context",
            perm=[0, 2, 1, 3],
        )
        self._node(
            "Reshape",
            [context_transposed, "reshape_context"],
            [context_reshaped],
            name=f"{prefix}.reshape_context",
        )
        projected = self.linear(context_reshaped, f"{prefix}.attention_output", HIDDEN, HIDDEN)
        attention_residual = f"{prefix}.attention_residual"
        self._node("Add", [value, projected], [attention_residual], name=f"{prefix}.attention_add")
        ff_normalized = self.layer_norm(attention_residual, f"{prefix}.ff_norm")
        expanded = self.linear(ff_normalized, f"{prefix}.ff_expand", HIDDEN, FEED_FORWARD)
        activated = self.gelu(expanded, f"{prefix}.gelu")
        contracted = self.linear(activated, f"{prefix}.ff_contract", FEED_FORWARD, HIDDEN)
        output = f"{prefix}.output"
        self._node("Add", [attention_residual, contracted], [output], name=f"{prefix}.ff_add")
        return output


def generate_tiny_transformer(
    destination: Path,
    *,
    precision: Literal["fp32", "fp16"] = "fp32",
) -> GeneratedModel:
    """Generate the locked four-block dynamic mini transformer."""

    builder = _GraphBuilder(precision)
    current = "tokens"
    for block in range(BLOCKS):
        current = builder.transformer_block(current, block)
    builder._node("Identity", [current], ["output"], name="output_identity")
    inputs = [
        helper.make_tensor_value_info(
            "tokens",
            builder.tensor_type,
            ["batch", "sequence", HIDDEN],
        ),
        helper.make_tensor_value_info(
            "mask",
            builder.tensor_type,
            ["batch", 1, 1, "sequence"],
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "output",
            builder.tensor_type,
            ["batch", "sequence", HIDDEN],
        )
    ]
    graph = helper.make_graph(
        builder.nodes,
        "upgradeguard_tiny_transformer",
        inputs,
        outputs,
        initializer=builder.initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="tensorrt-upgrade-guard",
        producer_version="0.1.0",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = (
        "Project-owned deterministic four-block dynamic mini transformer. "
        f"precision={precision}; seed={SEED}."
    )
    onnx.checker.check_model(model, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    return GeneratedModel(
        path=destination,
        sha256=sha256_file(destination),
        bytes=destination.stat().st_size,
        precision=precision,
        opset=OPSET,
        ir_version=model.ir_version,
    )


def generate_plugin_micrograph(
    destination: Path,
    *,
    precision: Literal["fp32", "fp16"] = "fp32",
) -> GeneratedModel:
    """Generate the dynamic ONNX graph containing the project custom operator."""

    tensor_type = TensorProto.FLOAT if precision == "fp32" else TensorProto.FLOAT16
    inputs = [
        helper.make_tensor_value_info("x", tensor_type, ["batch", "tokens", "hidden"]),
        helper.make_tensor_value_info("residual", tensor_type, ["batch", "tokens", "hidden"]),
        helper.make_tensor_value_info("gamma", TensorProto.FLOAT, ["hidden"]),
    ]
    output = helper.make_tensor_value_info("output", tensor_type, ["batch", "tokens", "hidden"])
    node = helper.make_node(
        "ResidualRMSNorm",
        ["x", "residual", "gamma"],
        ["output"],
        name="residual_rmsnorm",
        domain="com.udayarora.upgradeguard",
        epsilon=1e-5,
    )
    graph = helper.make_graph([node], "upgradeguard_plugin_micrograph", inputs, [output])
    model = helper.make_model(
        graph,
        producer_name="tensorrt-upgrade-guard",
        producer_version="0.1.0",
        opset_imports=[
            helper.make_opsetid("", OPSET),
            helper.make_opsetid("com.udayarora.upgradeguard", 1),
        ],
    )
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    return GeneratedModel(
        path=destination,
        sha256=sha256_file(destination),
        bytes=destination.stat().st_size,
        precision=precision,
        opset=OPSET,
        ir_version=model.ir_version,
    )
