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


def _walk_ops(op, name):
    """Recursively walk an operation tree yielding ops matching `name`."""
    for region in op.regions:
        for block in region.blocks:
            for inner_op in block.operations:
                if inner_op.name == name:
                    yield inner_op
                yield from _walk_ops(inner_op, name)


def convert_scratch_allocs(module):
    """Replace memref.alloc() scratch buffers inside d2m.generic regions
    with d2m.scratch_allocate ops.

    Must run AFTER d2m-lower-load-store-ops-to-explicit-cb-form so that
    the only remaining memref.alloc (no memory space, tile element type)
    inside d2m.generic are true scratch buffers.  The d2m.scratch_allocate
    ops survive through the remaining middle passes and are lowered later
    by lower_scratch_allocates().
    """
    ctx = module.context
    slot = 0
    with ctx:
        for generic_op in _walk_ops(module.operation, "d2m.generic"):
            for region in generic_op.regions:
                for block in region.blocks:
                    allocs_to_replace = []
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
                        allocs_to_replace.append(inner_op)
                    for alloc_op in allocs_to_replace:
                        with InsertionPoint(alloc_op):
                            scratch = d2m.ScratchAllocateOp(
                                alloc_op.results[0].type, slot,
                                loc=alloc_op.location,
                            )
                        alloc_op.results[0].replace_all_uses_with(scratch.result)
                        alloc_op.operation.erase()
                        slot += 1


def lower_scratch_allocates(module):
    """Lower d2m.scratch_allocate ops in compute kernel functions to
    subviews of a single consolidated scratch CB (mainline pattern).

    Runs after d2m-generic-regions-to-funcs has extracted regions into
    standalone functions.  All scratch allocations in a compute kernel
    are consolidated into one CB.

    After linearize-memref, each scratch_allocate has a collapse_shape
    user that flattens the result to 1D.  We replace the collapse_shape
    output directly with a flat 1D subview of the consolidated buffer,
    avoiding memref.expand_shape (which D2MToTTKernel cannot lower).
    """
    ctx = module.context
    with ctx, Location.unknown(ctx):
        for func_op in module.body.operations:
            attr_dict = {na.name: na.attr for na in func_op.attributes}
            thread_attr = attr_dict.get("d2m.thread")
            if thread_attr is None or "compute" not in str(thread_attr):
                continue
            block = func_op.regions[0].blocks[0]

            # Find max existing CB port in this function.
            max_port = -1
            for inner_op in block.operations:
                if inner_op.name == "d2m.get_cb":
                    port_attr = inner_op.attributes["port"]
                    max_port = max(max_port, IntegerAttr(port_attr).value)

            # Collect scratch_allocate ops with their slot IDs and sizes.
            scratch_ops = []
            for inner_op in block.operations:
                if inner_op.name != "d2m.scratch_allocate":
                    continue
                slot_id = IntegerAttr(inner_op.attributes["slot"]).value
                mr = MemRefType(inner_op.results[0].type)
                num_elements = 1
                for d in mr.shape:
                    num_elements *= d
                scratch_ops.append((inner_op, slot_id, num_elements))
            if not scratch_ops:
                continue

            # Sort by slot for deterministic layout.
            scratch_ops.sort(key=lambda x: x[1])

            # Compute sequential element offsets.
            offsets = []
            current_offset = 0
            for _, _, num_elem in scratch_ops:
                offsets.append(current_offset)
                current_offset += num_elem
            total_tiles = current_offset

            # Element type from first scratch alloc.
            elem_type = MemRefType(scratch_ops[0][0].results[0].type).element_type

            # Create single consolidated flat scratch CB: memref<total_tiles x elem>
            scratch_memref_type = MemRefType.get([total_tiles], elem_type)
            cb_type = Type.parse(
                f"!d2m.cb<{scratch_memref_type}>", context=ctx
            )
            next_port = max_port + 1

            # Insert get_cb + wait at the first scratch_allocate location.
            first_scratch = scratch_ops[0][0]
            with InsertionPoint(first_scratch):
                get_cb = d2m.GetCBOp(
                    cb_type, next_port, loc=first_scratch.location
                )
                wait_op = d2m.WaitOp(
                    scratch_memref_type, get_cb.result,
                    loc=first_scratch.location,
                )
            scratch_base = wait_op.result

            # Replace each scratch_allocate with a flat subview.
            for i, (scratch_op, slot_id, num_elem) in enumerate(scratch_ops):
                loc = scratch_op.location

                # Build a flat 1D slice of the consolidated buffer.
                flat_type = MemRefType.get([num_elem], elem_type)

                # Find collapse_shape users (from linearize-memref) and
                # replace their results directly with the flat subview.
                collapse_users = []
                direct_users = []
                for use in scratch_op.results[0].uses:
                    user = use.owner
                    if user.name == "memref.collapse_shape":
                        collapse_users.append(user)
                    else:
                        direct_users.append(use)

                if total_tiles == num_elem and not collapse_users:
                    # Single scratch fills the whole buffer, no collapse —
                    # just use the wait result directly (same shape).
                    scratch_op.results[0].replace_all_uses_with(scratch_base)
                    scratch_op.operation.erase()
                    continue

                # Create the subview (only if we actually need a slice
                # or need to match the flat type for collapse users).
                with InsertionPoint(scratch_op):
                    if total_tiles == num_elem:
                        # Whole buffer, but collapse users need flat type.
                        subview_result = scratch_base
                    else:
                        subview = Operation.create(
                            "memref.subview",
                            results=[flat_type],
                            operands=[scratch_base],
                            attributes={
                                "static_offsets": DenseI64ArrayAttr.get(
                                    [offsets[i]]
                                ),
                                "static_sizes": DenseI64ArrayAttr.get(
                                    [num_elem]
                                ),
                                "static_strides": DenseI64ArrayAttr.get(
                                    [1]
                                ),
                                "operandSegmentSizes": DenseI32ArrayAttr.get(
                                    [1, 0, 0, 0]
                                ),
                            },
                            loc=loc,
                        )
                        subview_result = subview.results[0]

                # Replace collapse_shape users with flat subview directly.
                for collapse_op in collapse_users:
                    collapse_op.results[0].replace_all_uses_with(subview_result)
                    collapse_op.operation.erase()

                # Replace any remaining direct users of scratch_allocate.
                if direct_users:
                    scratch_op.results[0].replace_all_uses_with(subview_result)

                scratch_op.operation.erase()

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
        
        # Passes before scratch marking (bufferize through explicit-cb-form).
        # After explicit-cb-form, remote_load/store allocs are replaced with
        # CB waits, leaving only true scratch allocs as bare memref.alloc.
        pre_scratch_passes = [
            # Stage 4 — bufferize, register device
            "ttir-bufferization-pipeline{ttnn-mode=true}",
            "ttcore-register-device",
            # DST-aware tiling of linalg.generic ops (must run BEFORE linalg-to-affine).
            # Requires unified form with linalg.generic ops that have indexing_maps.
            "d2m-generic-tile-compute-loops",
            "d2m-linalg-to-affine{use-tile-matmul=true}",
            "d2m-insert-dst-register-access",
            # Stage 5 — affine loop finalization
            "d2m-sfpu-tile-loop-fission",
            "canonicalize",
            "lower-affine",
            "fold-memref-alias-ops",
            "lower-affine",
            # Stage 6 — explicit CB form (must run BEFORE linearize-memref to avoid
            # type mismatch: linearize creates collapse_shape on local allocs without
            # memory space, but explicit-cb-form replaces allocs with d2m.wait #l1 results).
            # d2m-convert-local-load-store-ops-to-aliased-cbs is skipped — it fails
            # on explicit datamovement form (block_factors=[]).
            "d2m-lower-load-store-ops-to-explicit-cb-form",
        ]
        # Passes between scratch marking and scratch lowering.
        middle_passes = [
            "d2m-generic-linearize-memref",
            "lower-affine",
            # Stage 7 — thread split
            "d2m-split-unified-thread",
            # Stage 8 — DMA lowering
            "d2m-preallocate-mcast-semaphores",
            "d2m-schedule-dma",
            "d2m-lower-load-store-ops-to-dma",
            "d2m-lower-dma-to-fully-indexed-form",
            # Optimization passes (matching reference pipeline)
            "canonicalize",
            # Stage 9 — region extraction
            "d2m-generic-regions-to-funcs",
        ]
        # Passes after scratch lowering.
        post_scratch_passes = [
            # Stage 10 — TTKernel conversion
            "convert-d2m-to-ttkernel{ttnn-mode=true}",
            "canonicalize",
            "ttkernel-control-dst-section",
            # Optimization passes
            "canonicalize",
            # Stage 11 — final TTNN conversion
            "convert-d2m-to-ttnn" if self.math_fidelity is None
                else f"convert-d2m-to-ttnn{{math-fidelity={self.math_fidelity}}}",
            "ttkernel-hoist-inits",
        ]

        def run_pass(pass_name):
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

        try:
            from ttmlir.passmanager import PassManager

            # Phase 1: bufferize + register device
            for pass_name in pre_scratch_passes:
                run_pass(pass_name)

            # Mark scratch allocs as d2m.scratch_allocate inside d2m.generic.
            convert_scratch_allocs(module)
            if self.debug:
                print("--- IR After convert_scratch_allocs ---")
                print(module)

            # Phase 2: middle passes (scratch_allocate ops pass through).
            for pass_name in middle_passes:
                run_pass(pass_name)

            # Lower scratch_allocate ops to single CB + subviews.
            lower_scratch_allocates(module)
            if self.debug:
                print("--- IR After lower_scratch_allocates ---")
                print(module)

            # Phase 3: TTKernel + TTNN conversion.
            for pass_name in post_scratch_passes:
                run_pass(pass_name)
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