#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Print in-place ops found in a .ttnn flatbuffer and their mutated TensorRefs.

Usage:
    python show_inplace_ops.py <path-to.ttnn>
"""
import sys

import ttrt.binary
import ttrt.runtime as rt

from chisel.ops import IRModule, get_op_outputs


def find_inplace_ops(binary_path: str):
    binary = ttrt.binary.load_binary_from_path(binary_path)
    mlir_json = ttrt.binary.mlir_as_dict(binary)
    functions = [binary.get_program_name(i) for i in range(binary.get_num_programs())]
    ir_module = IRModule(mlir_source=mlir_json["source"], functions=functions)

    results = []

    for prog_idx in range(binary.get_num_programs()):
        prog_name = binary.get_program_name(prog_idx)
        mlir_ops = iter(ir_module.get_function_ops(prog_name))

        def _walk(_bin, prog_ctx, op_ctx, _mlir_ops=mlir_ops):
            mlir_op = next(_mlir_ops)

            # In-place: no MLIR tensor results, but runtime returns the mutated ref
            if get_op_outputs(mlir_op):
                return

            output_refs = rt.get_op_output_refs(op_ctx, prog_ctx)
            if not output_refs:
                return

            inplace_refs = tuple(output_refs)
            results.append((prog_name, mlir_op.name, inplace_refs))
            shapes = [ref.get_shape() for ref in inplace_refs]
            dtypes = [ref.get_dtype() for ref in inplace_refs]
            print(f"  [{prog_name}] {mlir_op.name}")
            for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
                print(f"    mutated[{i}]: shape={shape}  dtype={dtype}")

        rt.walk_binary(binary, prog_idx, _walk)

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to.ttnn>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Scanning: {path}\n")
    found = find_inplace_ops(path)
    print(f"\nTotal in-place ops found: {len(found)}")
