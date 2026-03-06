# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import ast
import os
from typing import Dict, Any

from ttmlir.ir import *
from ttmlir.dialects import func, scf
from ttmlir.dialects import _d2m_ops_gen as d2m

from d2m_jit._src.mlir_generator import generate_ir


def promote_scratch_allocs_to_cbs(module):
    """Replace memref.alloc() scratch buffers in compute kernel functions with
    d2m.wait on new CB function arguments, so convert-d2m-to-ttkernel can
    convert them via D2MCBOpRewriter (avoiding unresolved materializations).

    NOTE: This correctly handles the stage 10 materialization problem, but
    leaves a semantic gap at stage 11: the new CB arg (arg index N) gets
    ArgAttr(CBPort, N) in the kernel's ArgSpec, but d2m.generic only has N
    operands so ttnn.generic's CB descriptors array is missing the scratch CB
    at index N.  For compile_only=True this is benign; before execution is
    implemented, fix D2MToTTNN to synthesize scratch CB descriptors from
    extra CBPort entries in the compute kernel's ArgSpec, or switch the
    frontend to emit d2m.scratch_allocate + run d2m-add-scratch-inputs /
    d2m-lower-scratch-allocate properly.
    """
    ctx = module.context
    with ctx:
        for op in module.body.operations:
            attr_dict = {na.name: na.attr for na in op.attributes}
            thread_attr = attr_dict.get("d2m.thread")
            if thread_attr is None or "compute" not in str(thread_attr):
                continue
            block = op.regions[0].blocks[0]
            scratch_allocs = []
            for inner_op in block.operations:
                if inner_op.name != "memref.alloc":
                    continue
                result_type = inner_op.results[0].type
                if not MemRefType.isinstance(result_type):
                    continue
                mr = MemRefType(result_type)
                if mr.memory_space is not None:
                    continue
                if "tile" not in str(mr.element_type).lower():
                    continue
                scratch_allocs.append(inner_op)
            if not scratch_allocs:
                continue
            old_func_type = FunctionType(TypeAttr(attr_dict["function_type"]).value)
            new_arg_types = list(old_func_type.inputs)
            for alloc_op in scratch_allocs:
                alloc_type = alloc_op.results[0].type
                alloc_loc = alloc_op.location
                cb_type = Type.parse(f"!d2m.cb<{str(alloc_type)}>", context=ctx)
                new_arg = block.add_argument(cb_type, alloc_loc)
                with InsertionPoint(alloc_op):
                    wait_op = d2m.WaitOp(alloc_type, new_arg, loc=alloc_loc)
                alloc_op.results[0].replace_all_uses_with(wait_op.result)
                alloc_op.operation.erase()
                new_arg_types.append(cb_type)
            new_func_type = FunctionType.get(new_arg_types, list(old_func_type.results))
            op.attributes["function_type"] = TypeAttr.get(new_func_type)

class AstCompiler:
    def __init__(self, func, grid, compile_only, debug, math_fidelity):
        self.func = func
        self.grid = grid
        self.compile_only = compile_only
        self.debug = debug
        self.math_fidelity = math_fidelity
        self.out_dir = os.path.join("generated", "d2m_jit", func.__name__)
        os.makedirs(self.out_dir, exist_ok=True)
        
    def _get_source_and_parse(self):
        source = inspect.getsource(self.func)
        
        # Strip decorators
        lines = source.split('\n')
        while lines and lines[0].strip().startswith('@'):
            lines.pop(0)
            
        source = '\n'.join(lines)
        
        # Python ast doesn't like the pseudo-code "Tensor a"
        # We replace it with "a: Tensor"
        import re
        source = re.sub(r'Tensor\s+([a-zA-Z_]\w*)', r'\1: Tensor', source)
        
        tree = ast.parse(source)
        return tree, source
        
    def compile_and_run(self, *args, **kwargs):
        tree, source = self._get_source_and_parse()
        
        if self.debug:
            print(f"--- Parsed Source for {self.func.__name__} ---")
            print(source)
            if hasattr(ast, "unparse"):
                print("--- Unparsed AST ---")
                print(ast.unparse(tree))
            else:
                print(ast.dump(tree, indent=4))
            
        # Parse tensor shapes and dtypes from args
        sig = inspect.signature(self.func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        tensor_metadata = {}
        for name, value in bound_args.arguments.items():
            if hasattr(value, "shape"): # Relaxed check since we're using a stub class in tests
                tensor_metadata[name] = {
                    "shape": list(value.shape),
                    "dtype": value.dtype,
                    "is_output": name == "out" # Hack for now: assume param named 'out' is the output
                }
                
        # Generate MLIR
        module = generate_ir(tree.body[0], self.grid, tensor_metadata, self.debug)

        print("--- Frontend Generated IR (Pre-Pipeline) ---")
        print(module)
        
        pass_pipeline = [
            # Stage 4 — existing: bufferize, register device, linalg→affine, DST access
            "ttir-bufferization-pipeline{ttnn-mode=true}",
            "ttcore-register-device",
            "d2m-linalg-to-affine{use-tile-matmul=true}",
            "d2m-insert-dst-register-access",
            # Stage 5 — affine loop finalization
            "d2m-sfpu-tile-loop-fission",
            "canonicalize",
            "lower-affine",
            "fold-memref-alias-ops",
            "lower-affine",
            # Stage 6 — explicit CB form
            # Note: d2m-convert-local-load-store-ops-to-aliased-cbs is for the TTIR→D2M path;
            # it fails on d2m-jit's explicit form (block_factors=[]), so skip it.
            # Note: d2m-lower-load-store-ops-to-explicit-cb-form must run BEFORE
            # d2m-generic-linearize-memref to avoid a type mismatch: linearize generates
            # collapse_shape on local allocs (no memory space), but explicit-cb-form
            # replaces the alloc with d2m.wait result (#l1 memory space), invalidating the type.
            "d2m-lower-load-store-ops-to-explicit-cb-form",
            # Stage 5b — linearize memref indices (after CBs are set up with correct types)
            "d2m-generic-linearize-memref",
            "lower-affine",
            # Stage 7 — thread split
            "d2m-split-unified-thread",
            # Stage 8 — DMA lowering
            "d2m-preallocate-mcast-semaphores",
            "d2m-schedule-dma",
            "d2m-lower-load-store-ops-to-dma",
            # Stage 9 — region extraction
            "d2m-generic-regions-to-funcs",
            # Stage 10 — TTKernel conversion (ttnn-mode=true so ttir ops stay for convert-d2m-to-ttnn)
            "convert-d2m-to-ttkernel{ttnn-mode=true}",
            "canonicalize",
            "ttkernel-control-dst-section",
            # Stage 11 — final TTNN conversion
            "convert-d2m-to-ttnn" if self.math_fidelity is None
                else f"convert-d2m-to-ttnn{{math-fidelity={self.math_fidelity}}}",
            # ttkernel-hoist-inits runs after convert-d2m-to-ttnn (matching reference pipeline order)
            "ttkernel-hoist-inits",
        ]

        try:
            from ttmlir.passmanager import PassManager

            for pass_name in pass_pipeline:
                single_pass_pipeline = f"builtin.module({pass_name})"
                try:
                    pm = PassManager.parse(single_pass_pipeline, module.context)
                    pm.run(module.operation)
                    print(f"--- IR After Pass: {pass_name} ---")
                    print(module)
                except Exception as pass_error:
                    print(f"--- IR After Pass (failed): {pass_name} ---")
                    print(module)
                    raise pass_error

                # Between stage 9 and stage 10: promote scratch allocs to CBs
                # so that convert-d2m-to-ttkernel can lower them without
                # unresolved materializations.
                if pass_name == "d2m-generic-regions-to-funcs":
                    promote_scratch_allocs_to_cbs(module)
                    if self.debug:
                        print("--- IR After promote_scratch_allocs_to_cbs ---")
                        print(module)
        except Exception as e:
            if self.debug:
                print(f"--- Lowering Pass Failed ---")
                print(e)
        
        if self.debug:
            print("--- Generated MLIR ---")
            print(module)
            
        if self.compile_only:
            # TODO: Add lowering to flatbuffer here
            return module
            
        # TODO: Add runtime execution
        raise NotImplementedError("Execution not yet implemented")