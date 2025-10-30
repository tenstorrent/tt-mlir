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
    backend = kwargs.get("_backend", "ttnn")

    graph_compiler = GraphToIRCompiler(captured_graph, f.__name__, tensor_args, max_grid, backend)
    ir = graph_compiler.compile()
    
    print_and_verify_ir(ir, "GraphToIRCompiler (Graph-based)", debug)

    return ir

def generate_ir(use_ttir_compiler, source_code, f, debug, *args, **kwargs):
    if use_ttir_compiler:
        return generate_ir_from_ast(source_code, debug, *args, **kwargs)
    else:
        return generate_ir_from_graph(f, debug, *args, **kwargs)