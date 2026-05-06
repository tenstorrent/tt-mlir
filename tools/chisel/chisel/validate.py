# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pre-flight binary validation: single walk_binary pass per program.

Runs four sub-checks in one pass through each program's ops before any
op-level hooks fire:

  1. debug_info          — op count + debug-info string match (MLIR vs flatbuffer)
  2. shapes              — TensorRef input/output shapes match IR types
  3. golden_meta_iso     — golden functions produce expected shape/dtype given
                           fresh meta tensors built from IR types (isolation)
  4. golden_meta_accum   — same, but inputs come from a running pool seeded from
                           TensorRef metadata; outputs are stored back (accumulation)

Returns a list of failure dicts. Each entry has keys:
  check, program, op, detail

An empty list means all checks passed. Failures are non-fatal.
"""
import json
import logging
import re

import torch
from _ttmlir_runtime import runtime as rt

from golden import get_chisel_golden_function, is_non_executable_op
from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype

from .executor import execute_golden
from .op_configs import get_skipped_op_names
from .ops import IRModule, get_op_inputs, get_op_outputs

logger = logging.getLogger("chisel")

_SKIP_OPS = get_skipped_op_names()


def _json_as_list(json_string: str) -> list:
    if not json_string:
        return []
    json_string = re.sub(r"\bnan\b", "NaN", json_string)
    json_string = re.sub(r"\binf\b", "Infinity", json_string)
    parsed = json.loads(json_string)
    return parsed if isinstance(parsed, list) else []


def _meta_from_ir(operand) -> torch.Tensor:
    """Build a meta tensor from an MLIR IR operand's type."""
    shape = list(operand.type.shape)
    try:
        dtype = mlir_type_to_torch_dtype(operand.type.element_type)
    except TypeError:
        dtype = torch.float32
    return torch.empty(shape, dtype=dtype, device="meta")


def _meta_from_ref(ref) -> torch.Tensor:
    """Build a meta tensor from a TensorRef's shape/dtype metadata."""
    try:
        dtype = mlir_datatype_to_torch_dtype(ref.get_dtype())
    except Exception:
        dtype = torch.float32
    return torch.empty(ref.get_shape(), dtype=dtype, device="meta")


def _check_golden_meta(
    mlir_op, ir_module: IRModule, asm_state,
    pool: dict | None = None,
    input_refs: list | None = None,
) -> str | None:
    """Run golden on meta tensors; return a failure detail string or None.

    pool=None        → isolation: build fresh meta inputs from IR types.
    pool=dict        → accumulation: pull inputs from pool when present;
                       otherwise seed from input_refs[i] (TensorRef metadata)
                       and store in pool. Store golden outputs back to pool.
    input_refs       → TensorRefs already fetched for sub-check 2 (no extra
                       runtime calls needed).
    """
    if is_non_executable_op(type(mlir_op.opview)):
        return None
    if get_chisel_golden_function(type(mlir_op.opview)) is None:
        return None

    op_inputs = get_op_inputs(mlir_op)
    inputs = {}
    for i, operand in enumerate(op_inputs):
        name = operand.get_name(asm_state)
        if pool is None:
            inputs[name] = _meta_from_ir(operand)
        else:
            if name not in pool:
                ref = input_refs[i] if input_refs and i < len(input_refs) else None
                pool[name] = _meta_from_ref(ref) if ref is not None else _meta_from_ir(operand)
            inputs[name] = pool[name]

    try:
        golden_result = execute_golden(mlir_op.opview, ir_module, inputs)
    except Exception as exc:
        logger.debug("preflight golden %s: execution error: %s", mlir_op.name, exc)
        return None

    if golden_result is None:
        return None

    # Store accumulation outputs back to pool.
    if pool is not None:
        for out_val, tensor in zip(get_op_outputs(mlir_op), golden_result):
            pool[out_val.get_name(asm_state)] = tensor

    op_outputs = get_op_outputs(mlir_op)
    if not op_outputs:
        return None

    expected_shape = list(op_outputs[0].type.shape)
    try:
        expected_dtype = mlir_type_to_torch_dtype(op_outputs[0].type.element_type)
    except TypeError:
        expected_dtype = torch.float32

    first = golden_result[0]
    if list(first.shape) != expected_shape:
        return f"shape: got={list(first.shape)} expected={expected_shape}"
    if first.dtype != expected_dtype:
        return f"dtype: got={first.dtype} expected={expected_dtype}"
    return None


def validate_binary(binary, ir_module: IRModule) -> list[dict]:
    """Walk each program once, running all four pre-flight sub-checks per op.

    Returns a list of failure dicts with keys: check, program, op, detail.
    An empty list means all checks passed.
    """
    failures = []

    def _fail(check, program, op_name, detail):
        failures.append({"check": check, "program": program, "op": op_name, "detail": detail})
        logger.warning("preflight[%s] %s %s: %s", check, program, op_name or "", detail)

    for prog_idx in range(binary.get_num_programs()):
        prog_name = binary.get_program_name(prog_idx)
        mlir_ops = ir_module.get_function_ops(prog_name)
        fb_ops = _json_as_list(binary.get_program_ops_as_json(prog_idx))
        asm_state = ir_module.get_asm_state()
        accum_pool = {}

        if len(mlir_ops) != len(fb_ops):
            _fail("op_count", prog_name, None, f"MLIR={len(mlir_ops)} FB={len(fb_ops)}")

        mlir_iter = iter(mlir_ops)
        fb_iter = iter(fb_ops)

        def _cb(_bin, prog_ctx, op_ctx,
                _mlir_iter=mlir_iter, _fb_iter=fb_iter,
                _prog=prog_name, _asm=asm_state,
                _ir_module=ir_module, _pool=accum_pool):
            mlir_op = next(_mlir_iter, None)
            if mlir_op is None:
                return

            # Sub-check 1: debug info
            fb_op = next(_fb_iter, None)
            if fb_op is not None:
                mlir_debug = mlir_op.get_asm(enable_debug_info=True)
                fb_debug = fb_op.get("debug_info", "")
                if mlir_debug != fb_debug:
                    _fail("debug_info", _prog, mlir_op.name,
                          f"MLIR={mlir_debug!r} FB={fb_debug!r}")

            # Sub-check 2: TensorRef shapes (fetch refs once; reused for accum below)
            input_refs = rt.get_op_input_refs(op_ctx, prog_ctx)
            output_refs = rt.get_op_output_refs(op_ctx, prog_ctx)
            if mlir_op.name not in _SKIP_OPS:
                for kind, get_mlir_vals, refs in (
                    ("input", get_op_inputs, input_refs),
                    ("output", get_op_outputs, output_refs),
                ):
                    mlir_vals = get_mlir_vals(mlir_op)
                    if len(mlir_vals) != len(refs):
                        _fail(f"{kind}_count", _prog, mlir_op.name,
                              f"IR={len(mlir_vals)} FB={len(refs)}")
                    else:
                        for val, ref in zip(mlir_vals, refs):
                            ir_shape = list(val.type.shape)
                            fb_shape = ref.get_shape()
                            if ir_shape != fb_shape:
                                _fail(f"{kind}_shape", _prog, mlir_op.name,
                                      f"IR={ir_shape} FB={fb_shape}")

            # Sub-check 3: golden isolation (fresh meta inputs from IR types)
            try:
                detail = _check_golden_meta(mlir_op, _ir_module, _asm)
                if detail is not None:
                    _fail("golden_meta_iso", _prog, mlir_op.name, detail)
            except Exception as exc:
                logger.debug("preflight golden_meta_iso %s crashed: %s", mlir_op.name, exc)

            # Sub-check 4: golden accumulation (pool seeded from TensorRef metadata)
            try:
                detail = _check_golden_meta(
                    mlir_op, _ir_module, _asm,
                    pool=_pool, input_refs=input_refs,
                )
                if detail is not None:
                    _fail("golden_meta_accum", _prog, mlir_op.name, detail)
            except Exception as exc:
                logger.debug("preflight golden_meta_accum %s crashed: %s", mlir_op.name, exc)

        try:
            rt.walk_binary(binary, prog_idx, _cb)
        except Exception as exc:
            logger.error("preflight walk_binary prog=%s crashed: %s", prog_name, exc)

    if failures:
        logger.warning("preflight: %d failure(s) for binary %d", len(failures), binary.id)
    else:
        logger.debug("preflight: all checks passed for binary %d", binary.id)

    return failures
