# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttnn_jit._src.tracing_compiler import TracingCompiler


def print_and_verify_ir(ir, method_name, debug):
    if debug:
        print("---- IR Dump after " + method_name + " ----")
        print(ir)
    ir.operation.verify()


def generate_ir(f, debug, memory_config, *args, **kwargs):
    """Generate IR from tracing compilation."""
    compiler = TracingCompiler(f, *args, memory_config=memory_config, **kwargs)
    ir = compiler.compile()
    print_and_verify_ir(ir, "TracingCompiler (Tracing-based)", debug)
    return ir
