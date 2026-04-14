# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Validate that IRModule's op iterator produces the same op sequence as the
flatbuffer's operation list.  This is the invariant Chisel relies on: the
MLIR walk must match the runtime callback order.
"""
import json
import re
from typing import List, Optional, Tuple

import pytest

import ttrt.binary

from chisel.ops import IRModule


def _extract_op_name_from_debug_info(debug_info: str) -> Optional[str]:
    """Extract the MLIR op name (e.g. 'ttnn.add') from a flatbuffer debug_info string.

    The debug_info field contains the full op->print() output, e.g.:
        %3 = "ttnn.add"(%1, %2) <{...}> : (...) -> ...
    """
    m = re.search(r'"([a-z_]+\.[a-z_0-9]+)"', debug_info)
    return m.group(1) if m else None


def _get_fb_ops(
    binary, program_index: int
) -> List[Tuple[Optional[str], str]]:
    """Walk flatbuffer ops, return list of (op_name, loc_info)."""
    ops_json = json.loads(binary.get_program_ops_as_json(program_index))
    result = []
    for op in ops_json:
        name = _extract_op_name_from_debug_info(op.get("debug_info", ""))
        loc = op.get("loc_info", "")
        result.append((name, loc))
    return result


def _get_mlir_ops(
    binary, program_index: int
) -> List[Tuple[str, str]]:
    """Walk MLIR ops via IRModule, return list of (op_name, loc_str).

    Mirrors the flow in ChiselContext.ensure_ir_module().
    """
    mlir_json = json.loads(binary.get_mlir_as_json())
    mlir_source = mlir_json["source"]
    functions = [
        binary.get_program_name(i)
        for i in range(binary.get_num_programs())
    ]
    ir_module = IRModule(mlir_source=mlir_source, functions=functions)
    ir_module.current_function_name = binary.get_program_name(program_index)
    ops = ir_module.get_function_ops()
    return [(op.name, str(op.location)) for op in ops]


def _non_private_programs(binary):
    """Yield (index, name) for each non-private program."""
    for i in range(binary.get_num_programs()):
        if not binary.is_program_private(i):
            yield i, binary.get_program_name(i)


def test_op_count_matches(binary_path):
    """Op count from MLIR walk must equal op count from flatbuffer per program."""
    if binary_path is None:
        pytest.skip(
            "No .ttnn binary provided "
            "(set CHISEL_TEST_BINARY_PATHS or use --binary-path)"
        )

    binary = ttrt.binary.load_binary_from_path(binary_path)

    for prog_idx, prog_name in _non_private_programs(binary):
        mlir_ops = _get_mlir_ops(binary, prog_idx)
        fb_ops = _get_fb_ops(binary, prog_idx)
        assert len(mlir_ops) == len(fb_ops), (
            f"Program '{prog_name}' (idx {prog_idx}): "
            f"MLIR has {len(mlir_ops)} ops, flatbuffer has {len(fb_ops)}"
        )


def test_op_order_matches(binary_path):
    """Op locations and names must match at every index between MLIR and flatbuffer walks."""
    if binary_path is None:
        pytest.skip(
            "No .ttnn binary provided "
            "(set CHISEL_TEST_BINARY_PATHS or use --binary-path)"
        )

    binary = ttrt.binary.load_binary_from_path(binary_path)

    for prog_idx, prog_name in _non_private_programs(binary):
        mlir_ops = _get_mlir_ops(binary, prog_idx)
        fb_ops = _get_fb_ops(binary, prog_idx)

        assert len(mlir_ops) == len(fb_ops), (
            f"Program '{prog_name}': count mismatch "
            f"(MLIR={len(mlir_ops)}, FB={len(fb_ops)})"
        )

        for i, ((mlir_name, mlir_loc), (fb_name, fb_loc)) in enumerate(
            zip(mlir_ops, fb_ops)
        ):
            assert mlir_loc == fb_loc, (
                f"Program '{prog_name}' op {i}: location mismatch\n"
                f"  MLIR: {mlir_name} @ {mlir_loc}\n"
                f"  FB:   {fb_name} @ {fb_loc}"
            )

            if fb_name is not None:
                assert mlir_name == fb_name, (
                    f"Program '{prog_name}' op {i}: name mismatch\n"
                    f"  MLIR: {mlir_name}\n"
                    f"  FB:   {fb_name}"
                )
