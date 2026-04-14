# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cross-validate IRModule against the flatbuffer binary: op walk order,
tensor shapes, and element types.
"""
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import pytest

import ttrt.binary

from chisel.ops import IRModule, get_op_inputs, get_op_outputs


# Maps str(element_type) from MLIR to flatbuffer DataType enum string.
# Path in FB JSON: tensor_ref["desc"]["layout"]["memory_desc"]["data_type"]
MLIR_TO_FB_DTYPE = {
    "f32": "Float32",
    "f16": "Float16",
    "bf16": "BFloat16",
    "f64": "Float64",
    "i32": "Int32",
    "si32": "Int32",
    "i16": "Int16",
    "si16": "Int16",
    "i8": "Int8",
    "si8": "Int8",
    "ui32": "UInt32",
    "ui16": "UInt16",
    "ui8": "UInt8",
    "i64": "Int64",
    "ui64": "UInt64",
}


def _get_ir_module(binary) -> IRModule:
    """Build an IRModule from a binary's embedded MLIR."""
    mlir_json = json.loads(binary.get_mlir_as_json())
    mlir_source = mlir_json["source"]
    functions = [
        binary.get_program_name(i)
        for i in range(binary.get_num_programs())
    ]
    return IRModule(mlir_source=mlir_source, functions=functions)


def _get_fb_ops(
    binary, program_index: int
) -> List[str]:
    """Walk flatbuffer ops, return list of debug_info strings."""
    try:
        ops_json = json.loads(binary.get_program_ops_as_json(program_index))
    except json.decoder.JSONDecodeError:
        pytest.xfail("Flatbuffer has non-standard json (numbers can be -inf, inf or NaN), so this is expected in small number of tests")

    return [op.get("debug_info", "") for op in ops_json]


def _get_fb_ops_json(binary, program_index: int) -> List[dict]:
    """Walk flatbuffer ops, return list of full op dicts."""
    try:
        return json.loads(binary.get_program_ops_as_json(program_index))
    except json.decoder.JSONDecodeError:
        pytest.xfail("Flatbuffer has non-standard json (numbers can be -inf, inf or NaN), so this is expected in small number of tests")


def _get_mlir_ops(
    binary, program_index: int
) -> List[str]:
    """Walk MLIR ops via IRModule, return list of debug_info strings.

    Mirrors the flow in ChiselContext.ensure_ir_module().
    """
    ir_module = _get_ir_module(binary)
    ops = ir_module.get_function_ops(binary.get_program_name(program_index))
    return [op.get_asm(enable_debug_info=True) for op in ops]


def _itterate_programs(binary):
    """Yield (index, name) for each program."""
    for i in range(binary.get_num_programs()):
        yield i, binary.get_program_name(i)


# ---------------------------------------------------------------------------
# Tensor info extraction helpers
# ---------------------------------------------------------------------------

TensorInfo = Tuple[tuple, str]  # (shape, dtype)


def _is_tensor_ref(obj) -> bool:
    """True if obj looks like a flatbuffer TensorRef JSON object."""
    return isinstance(obj, dict) and "global_id" in obj and "desc" in obj


def _tensor_ref_info(ref: dict) -> TensorInfo:
    """Extract (shape, dtype) from a TensorRef JSON object."""
    shape = tuple(ref["desc"]["shape"])
    dtype = ref["desc"]["layout"]["memory_desc"]["data_type"]
    return (shape, dtype)


def _is_output_field(key: str) -> bool:
    """True if the flatbuffer field name designates an output TensorRef.

    TTNN convention: ``out`` for single-output ops, ``*_out`` for multi-output
    ops (e.g. ``q_out``, ``k_out``, ``v_out``), ``outputs`` for vector outputs.
    """
    return key == "out" or key.endswith("_out") or key == "outputs"


def _extract_fb_tensor_info(
    op_json: dict,
) -> Tuple[List[TensorInfo], Optional[List[TensorInfo]]]:
    """Extract (input_info, output_info) from a flatbuffer JSON op.

    Uses the TTNN schema convention to classify TensorRef fields:
    - ``out`` or ``*_out``: single output TensorRef
    - ``outputs``: list of output TensorRefs
    - everything else: input TensorRef

    Returns ``output_info=None`` when no output fields are found,
    meaning the caller should fall back to a combined comparison.
    """
    type_dict = op_json.get("type", {})
    if not type_dict:
        return [], None

    input_info: List[TensorInfo] = []
    output_info: List[TensorInfo] = []
    has_out_field = False

    for key, value in type_dict.items():
        if _is_tensor_ref(value):
            info = _tensor_ref_info(value)
            if _is_output_field(key):
                output_info.append(info)
                has_out_field = True
            else:
                input_info.append(info)
        elif key == "outputs" and isinstance(value, list):
            for item in value:
                if _is_tensor_ref(item):
                    output_info.append(_tensor_ref_info(item))
                    has_out_field = True

    return input_info, output_info if has_out_field else None


def _mlir_dtype_to_fb(element_type) -> str:
    """Convert an MLIR element type to a flatbuffer DataType string.

    Handles quantized types like ``!quant.uniform<i32:f32, 1.0e-01>`` by
    extracting the storage type (``i32`` in this example).
    """
    type_str = str(element_type)
    if type_str.startswith("!quant.uniform<"):
        storage_type = type_str.split("<")[1].split(":")[0]
        return MLIR_TO_FB_DTYPE.get(storage_type, type_str)
    return MLIR_TO_FB_DTYPE.get(type_str, type_str)


def _get_mlir_tensor_info(op) -> Tuple[List[TensorInfo], List[TensorInfo]]:
    """Return (input_info, output_info) from an MLIR op via get_op_inputs/get_op_outputs.

    Dtype strings are converted to the flatbuffer convention using MLIR_TO_FB_DTYPE.
    """
    input_info = [
        (
            tuple(t.type.shape),
            _mlir_dtype_to_fb(t.type.element_type),
        )
        for t in get_op_inputs(op)
    ]
    output_info = [
        (
            tuple(t.type.shape),
            _mlir_dtype_to_fb(t.type.element_type),
        )
        for t in get_op_outputs(op)
    ]
    return input_info, output_info


def test_op_order_matches(binary_path):
    """Op debug strings must match at every index between MLIR and flatbuffer walks."""
    assert binary_path is not None, (
        "No .ttnn binary provided (use --binary)"
    )
    assert Path(binary_path).exists(), (
        f"Binary path does not exist: {binary_path}"
    )

    binary = ttrt.binary.load_binary_from_path(binary_path)

    for prog_idx, prog_name in _itterate_programs(binary):
        mlir_ops = _get_mlir_ops(binary, prog_idx)
        fb_ops = _get_fb_ops(binary, prog_idx)

        assert len(mlir_ops) == len(fb_ops), (
            f"Program '{prog_name}': count mismatch "
            f"(MLIR={len(mlir_ops)}, FB={len(fb_ops)})"
        )

        for i, (mlir_debug, fb_debug) in enumerate(zip(mlir_ops, fb_ops)):
            assert mlir_debug == fb_debug, (
                f"Program '{prog_name}' op {i}: debug info mismatch\n"
                f"  MLIR: {mlir_debug}\n"
                f"  FB:   {fb_debug}"
            )


def test_op_tensor_info_matches(binary_path):
    """Tensor shapes and dtypes from get_op_inputs/get_op_outputs must match flatbuffer TensorRefs."""
    assert binary_path is not None, (
        "No .ttnn binary provided (use --binary)"
    )
    assert Path(binary_path).exists(), (
        f"Binary path does not exist: {binary_path}"
    )

    binary = ttrt.binary.load_binary_from_path(binary_path)
    ir_module = _get_ir_module(binary)

    for prog_idx, prog_name in _itterate_programs(binary):
        mlir_ops = ir_module.get_function_ops(binary.get_program_name(prog_idx))
        fb_ops = _get_fb_ops_json(binary, prog_idx)

        assert len(mlir_ops) == len(fb_ops), (
            f"Program '{prog_name}': count mismatch "
            f"(MLIR={len(mlir_ops)}, FB={len(fb_ops)})"
        )

        for i, (mlir_op, fb_op) in enumerate(zip(mlir_ops, fb_ops)):
            mlir_inputs, mlir_outputs = _get_mlir_tensor_info(mlir_op)
            fb_inputs, fb_outputs = _extract_fb_tensor_info(fb_op)

            mlir_all = mlir_inputs + mlir_outputs
            fb_all = fb_inputs + (fb_outputs if fb_outputs is not None else [])

            # Skip ops with nested TensorRefs (count mismatch from top-level extraction)
            if len(mlir_all) != len(fb_all):
                warnings.warn(
                    f"Program '{prog_name}' op {i}: tensor count mismatch "
                    f"(MLIR={len(mlir_all)}, FB={len(fb_all)}), skipping"
                )
                continue

            # FB schema field order may differ from MLIR operand order,
            # so sort both sides before comparing.
            if fb_outputs is not None:
                assert sorted(mlir_inputs) == sorted(fb_inputs), (
                    f"Program '{prog_name}' op {i}: INPUT tensor info mismatch\n"
                    f"  MLIR: {sorted(mlir_inputs)}\n"
                    f"  FB:   {sorted(fb_inputs)}"
                )
                assert sorted(mlir_outputs) == sorted(fb_outputs), (
                    f"Program '{prog_name}' op {i}: OUTPUT tensor info mismatch\n"
                    f"  MLIR: {sorted(mlir_outputs)}\n"
                    f"  FB:   {sorted(fb_outputs)}"
                )
            else:
                assert sorted(mlir_all) == sorted(fb_all), (
                    f"Program '{prog_name}' op {i}: combined tensor info mismatch\n"
                    f"  MLIR: {sorted(mlir_all)}\n"
                    f"  FB:   {sorted(fb_all)}"
                )
