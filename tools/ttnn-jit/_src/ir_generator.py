# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
from ttnn_jit._src.ttir_ast import TTIRCompiler
from ttnn_jit._src.graph_trace_compiler import GraphToIRCompiler
from ttnn._ttnn.graph import RunMode, begin_graph_capture, end_graph_capture
from ttnn.graph import visualize


def print_and_verify_ir(ir, method_name, debug):
    if debug:
        print("---- After " + method_name + " ----")
        print(ir)
    ir.operation.verify()


def generate_ir_from_ast(source_code, debug, *args, **kwargs):
    # Parse and compile
    m = ast.parse(source_code)
    if debug:
        print(ast.dump(m, indent=2) + "\n")

    # TODO (#5043): emit ttnn IR instead of TTIR, TTIR should be fallback.
    b = TTIRCompiler(None, *args, **kwargs)
    b.visit(m)
    ir = b.module

    print_and_verify_ir(ir, "TTIRCompiler (AST-based)", debug)

    return ir


def generate_ir_from_graph(f, debug, *args, **kwargs):
    begin_graph_capture(RunMode.NO_DISPATCH)
    f(*args)
    captured_graph = end_graph_capture()
    # visualize(captured_graph, file_name=f.__name__ + "_graph.svg")

    # Extract tensor args for the compiler
    tensor_args = kwargs.get("_tensor_args", {})
    max_grid = kwargs.get("_max_grid", (0, 0))

    graph_compiler = GraphToIRCompiler(
        captured_graph, f.__name__, tensor_args, max_grid
    )
    ir = graph_compiler.compile()

    print_and_verify_ir(ir, "GraphToIRCompiler (Graph-based)", debug)

    return ir


# This utility function, though not used in production code, can help in debugging whether both
# compilers (AST based and Graph based) are generating the same IR or not.
def compare_ir(ir_graph, ir_ast):
    ir_str_graph = str(ir_graph)
    ir_str_ast = str(ir_ast)
    if ir_str_graph == ir_str_ast:
        print("✅ IRs are IDENTICAL!")
    else:
        print("⚠️  IRs are DIFFERENT:")
        print(f"\nTTIRCompiler length: {len(ir_str_ast)} chars")
        print(f"GraphTraceCompiler length: {len(ir_str_graph)} chars")
        # Show first difference
        for i, (c1, c2) in enumerate(zip(ir_str_graph, ir_str_ast)):
            if c1 != c2:
                print(f"First difference at position {i}:")
                print(f"  TTIRCompiler: ...{ir_str_ast[max(0,i-20):i+20]}...")
                print(f"  GraphTraceCompiler: ...{ir_str_graph[max(0,i-20):i+20]}...")
                break

    assert ir_str_graph == ir_str_ast, "IRs are different"


def generate_ir(graph_capture, source_code, f, debug, *args, **kwargs):
    if graph_capture:
        return generate_ir_from_graph(f, debug, *args, **kwargs)
    else:
        return generate_ir_from_ast(source_code, debug, *args, **kwargs)
