# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cross-validate IRModule against the flatbuffer binary: op walk order,
tensor shapes, and element types.
"""
import json
import re

import pytest
from _ttmlir_runtime import binary as rt_binary


def _json_as_dict(json_string):
    if json_string == "":
        return {}
    json_string = re.sub(r"\bnan\b", "NaN", json_string)
    json_string = re.sub(r"\binf\b", "Infinity", json_string)
    return json.loads(json_string)


def _iterate_programs(binary):
    """Yield (index, name) for each program."""
    for i in range(binary.get_num_programs()):
        yield i, binary.get_program_name(i)


@pytest.mark.no_device
def test_ops(ir_module, binary, mlir_source_path):
    """Cross-validate walk order, debug info, and tensor shapes against the flatbuffer."""
    for prog_idx, prog_name in _iterate_programs(binary):
        mlir_ops = ir_module.get_function_ops(prog_name)
        fb_ops = _json_as_dict(binary.get_program_ops_as_json(prog_idx))

        assert len(mlir_ops) == len(fb_ops), (
            f"Program '{prog_name}': count mismatch "
            f"(MLIR={len(mlir_ops)}, FB={len(fb_ops)})"
            f"\nMLIR source: {mlir_source_path}"
        )

        for i, (mlir_op, fb_op) in enumerate(zip(mlir_ops, fb_ops, strict=True)):
            mlir_debug = mlir_op.get_asm(enable_debug_info=True)
            fb_debug = fb_op.get("debug_info", "")
            assert mlir_debug == fb_debug, (
                f"Program '{prog_name}' op {i}: debug info mismatch\n"
                f"  MLIR: {mlir_debug}\n"
                f"  FB:   {fb_debug}"
                f"\nMLIR source: {mlir_source_path}"
            )
