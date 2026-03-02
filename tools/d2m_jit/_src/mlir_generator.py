# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
from ttmlir.ir import *
from ttmlir.dialects import func, scf, arith
from ttmlir.dialects import _d2m_ops_gen as d2m

class D2MASTVisitor(ast.NodeVisitor):
    def __init__(self, ctx, block, tensor_metadata, grid):
        self.ctx = ctx
        self.block = block
        self.tensor_metadata = tensor_metadata
        self.grid = grid
        
        self.symbol_table = {}
        self.core_indices = {} # Tracks core_y, core_x
        self.scratch_slot = 0 # Tracks slot index for scratch_allocate
        self.outputs = [] # Tracks output values to yield
        
        # Populate symbol table with arguments
        input_names = [name for name, meta in tensor_metadata.items() if not meta["is_output"]]
        output_names = [name for name, meta in tensor_metadata.items() if meta["is_output"]]
        self.output_names = output_names
        
        all_names = input_names + output_names
        for idx, name in enumerate(all_names):
            self.symbol_table[name] = self.block.arguments[idx]
            if name in output_names:
                self.outputs.append(self.block.arguments[idx])
            
    def _resolve_slice(self, slice_node):
        """Resolves a python slice [start:end] to (offset, size)"""
        if isinstance(slice_node, ast.Slice):
            # Evaluate start
            start = self._eval_expr(slice_node.lower) if slice_node.lower else 0
            # Evaluate end
            end = self._eval_expr(slice_node.upper) if slice_node.upper else None
            
            # Simple symbolic subtraction for size (e.g. core_y + 128 - core_y = 128)
            size = None
            if end is not None:
                if isinstance(start, int) and isinstance(end, int):
                    size = end - start
                elif isinstance(end, tuple) and end[1] == 'Add' and end[0] == start:
                    size = end[2]
                else:
                    size = f"({end}) - ({start})"
            return start, size
        return None, None
        
    def _eval_expr(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            # For MLIR generation we need actual constants if calculating sizes,
            # or we generate affine maps. For now return a symbolic representation or string.
            if node.id in self.core_indices:
                return f"core_{node.id}" # Hacky symbolic rep
            return node.id
        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            if isinstance(left, int) and isinstance(right, int):
                if isinstance(node.op, ast.Add): return left + right
                if isinstance(node.op, ast.Sub): return left - right
                if isinstance(node.op, ast.Mult): return left * right
            # If it's symbolic, return a tuple representing the expression
            return (left, type(node.op).__name__, right)
        return 0 # Fallback

    def _infer_add_result_type(self, lhs_type, rhs_type):
        # With memrefs, we keep the same type if they match
        lhs = MemRefType(lhs_type)
        rhs = MemRefType(rhs_type)
        if lhs.shape == rhs.shape and lhs.element_type == rhs.element_type:
            return lhs
        return lhs

    def _infer_matmul_result_type(self, lhs_type, rhs_type):
        lhs = MemRefType(lhs_type)
        rhs = MemRefType(rhs_type)
        if len(lhs.shape) >= 2 and len(rhs.shape) >= 2 and lhs.element_type == rhs.element_type:
            prefix = list(lhs.shape[:-2])
            m_dim = lhs.shape[-2]
            n_dim = rhs.shape[-1]
            return MemRefType.get(prefix + [m_dim, n_dim], lhs.element_type, memory_space=lhs.memory_space)
        return lhs

    def _emit_linalg_generic_tile_op(self, op_name, lhs, rhs, out_type):
        from ttmlir.dialects import memref

        out_memref_type = MemRefType(out_type)
        # Allocate the output buffer in L1
        init = memref.AllocOp(out_memref_type, [], [])

        rank = len(out_memref_type.shape)
        identity_map = AffineMap.get_identity(rank)
        indexing_maps = ArrayAttr.get([
            AffineMapAttr.get(identity_map),
            AffineMapAttr.get(identity_map),
            AffineMapAttr.get(identity_map),
        ])
        iterator_types = ArrayAttr.get([StringAttr.get("parallel") for _ in range(rank)])

        linalg_generic = Operation.create(
            "linalg.generic",
            results=[],
            operands=[lhs, rhs, init.result],
            attributes={
                "indexing_maps": indexing_maps,
                "iterator_types": iterator_types,
                "operandSegmentSizes": DenseI32ArrayAttr.get([2, 1])
            },
            regions=1,
        )

        region = linalg_generic.regions[0]
        block = region.blocks.append(
            out_memref_type.element_type,
            out_memref_type.element_type,
            out_memref_type.element_type,
        )
        with InsertionPoint(block):
            if op_name == "add":
                tile_op = d2m.TileAddOp(
                    out_memref_type.element_type,
                    block.arguments[0],
                    block.arguments[1],
                )
                Operation.create("linalg.yield", operands=[tile_op.result])
            elif op_name == "matmul":
                tile_op = d2m.TileMatmulOp(
                    out_memref_type.element_type,
                    block.arguments[0],
                    block.arguments[1],
                    block.arguments[2],
                )
                Operation.create("linalg.yield", operands=[tile_op.result])
            else:
                # Keep unsupported ops structurally valid for now.
                Operation.create("linalg.yield", operands=[block.arguments[2]])

        # The result of the operation is the initialized and mutated output buffer
        return init.result
        
    def visit_Assign(self, node):
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return
            
        name = target.id
        val = node.value
        
        if isinstance(val, ast.Call):
            if isinstance(val.func, ast.Attribute) and val.func.value.id == "d2m" and val.func.attr == "core_idx":
                idx = val.args[0].value
                # d2m.block_index
                with Location.unknown(self.ctx), InsertionPoint(self.block):
                    idx_type = IndexType.get(self.ctx)
                    op = d2m.BlockIndexOp(IntegerAttr.get(IntegerType.get_signless(32), idx), results=[idx_type])
                    self.core_indices[name] = op.result
                    
            elif isinstance(val.func, ast.Name) and val.func.id == "alloc":
                # Handle alloc() -> memref.alloc
                with Location.unknown(self.ctx), InsertionPoint(self.block):
                    # Hardcode tile size for now
                    # TODO: Infer from type
                    f32 = F32Type.get()
                    tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=self.ctx)
                    l1_space = Attribute.parse("#ttcore.memory_space<l1>", context=self.ctx)
                    # Create a 1x1 buffer by default for eltwise?
                    memref_type = MemRefType.get([1, 1], tile_type, memory_space=l1_space)
                    from ttmlir.dialects import memref
                    op = memref.AllocOp(memref_type, [], [])
                    self.symbol_table[name] = op.result
                    
    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call) or node.iter.func.id != "range":
            raise NotImplementedError("Only range() loops are supported")
            
        args = node.iter.args
        start_val = 0
        step_val = 1
        
        if len(args) == 1:
            end_val = self._eval_expr(args[0])
        elif len(args) == 2:
            start_val = self._eval_expr(args[0])
            end_val = self._eval_expr(args[1])
        elif len(args) == 3:
            start_val = self._eval_expr(args[0])
            end_val = self._eval_expr(args[1])
            step_val = self._eval_expr(args[2])
            
        from ttmlir.dialects import affine
        from ttmlir.ir import IntegerAttr, IntegerType, IndexType
        
        with Location.unknown(self.ctx), InsertionPoint(self.block):
            # AffineForOp takes integer bounds directly
            start_int = int(start_val)
            end_int = int(end_val) if isinstance(end_val, int) else 128
            step_int = int(step_val)
            
            loop = affine.AffineForOp(start_int, end_int, step_int)
            
            # The induction variable is named node.target.id
            if isinstance(node.target, ast.Name):
                self.symbol_table[node.target.id] = loop.induction_variable
                
            # Visit the body inside the loop
            prev_block = self.block
            self.block = loop.body
            with InsertionPoint(self.block):
                for stmt in node.body:
                    self.visit(stmt)
            self.block = prev_block
            
    def visit_Call(self, node):
        if not isinstance(node.func, ast.Attribute):
            return
            
        if node.func.value.id != "d2m":
            return
            
        op_name = node.func.attr
        
        with Location.unknown(self.ctx), InsertionPoint(self.block):
            if op_name == "remote_load":
                dest_name = node.args[0].id
                src_node = node.args[1]
                
                # Parse slices: src[y_slice][x_slice]
                # In AST: Subscript(value=Subscript(value=Name, slice), slice)
                x_slice = src_node.slice
                y_slice = src_node.value.slice
                src_name = src_node.value.value.id
                
                y_start, y_size = self._resolve_slice(y_slice)
                x_start, x_size = self._resolve_slice(x_slice)
                
                # We need to map AST name node src_name to the MLIR variable
                src_val = self.symbol_table[src_name]
                dest_val = self.symbol_table[dest_name]
                
                # Mock up memref type for now based on slice sizes
                f32 = F32Type.get()
                tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=self.ctx)
                l1_space = Attribute.parse("#ttcore.memory_space<l1>", context=self.ctx)
                
                # Ensure sizes are evaluated if they are constants
                y_dim = 1
                x_dim = 1
                if isinstance(y_size, int): y_dim = max(1, y_size // 32)
                if isinstance(x_size, int): x_dim = max(1, x_size // 32)
                
                res_type = MemRefType.get([1, 1, y_dim, x_dim], tile_type, memory_space=l1_space)
                
                # Generate d2m.remote_load
                import ttmlir.dialects.d2m as d2m_dialect
                
                # RemoteLoadOp signature: result, memref, indices, mcastStartIndex, mcastShape, mcastDims, localBuffer, cb
                mcast_shape_attr = ArrayAttr.get([]) # Empty array for no mcast
                mcast_dims_attr = ArrayAttr.get([]) # Empty array for no mcast
                mcast_start_index = ArrayAttr.get([])
                
                op = d2m_dialect.RemoteLoadOp(res_type, src_val, [self.core_indices["core_y"], self.core_indices["core_x"]], mcast_start_index, mcast_shape_attr, mcast_dims_attr, localBuffer=dest_val)
                
                # Store the loaded value in symbol table for destination variable
                self.symbol_table[dest_name] = op.result
                
            elif op_name == "remote_store":
                dest_node = node.args[0]
                src_name = node.args[1].id
                
                # Parse slices: out[y_slice][x_slice]
                x_slice = dest_node.slice
                y_slice = dest_node.value.slice
                dest_name = dest_node.value.value.id
                
                y_start, y_size = self._resolve_slice(y_slice)
                x_start, x_size = self._resolve_slice(x_slice)
                
                src_val = self.symbol_table[src_name]
                dest_val = self.symbol_table[dest_name]
                import ttmlir.dialects.d2m as d2m_dialect

                store_op = d2m_dialect.RemoteStoreOp(
                    dest_val.type,
                    dest_val,
                    [self.core_indices["core_y"], self.core_indices["core_x"]],
                    localBuffer=src_val,
                )

                # Track the updated global tensor value for subsequent uses/yield.
                self.symbol_table[dest_name] = store_op.result
                if dest_name in self.output_names:
                    out_idx = self.output_names.index(dest_name)
                    self.outputs[out_idx] = store_op.result
                
            elif op_name in ["add", "mul", "max", "matmul"]:
                in1 = node.args[0].id
                in2 = node.args[1].id
                out = node.args[2].id
                
                val1 = self.symbol_table[in1]
                val2 = self.symbol_table[in2]
                if op_name == "add":
                    res_type = self._infer_add_result_type(val1.type, val2.type)
                    self.symbol_table[out] = self._emit_linalg_generic_tile_op("add", val1, val2, res_type)
                elif op_name == "matmul":
                    res_type = self._infer_matmul_result_type(val1.type, val2.type)
                    self.symbol_table[out] = self._emit_linalg_generic_tile_op("matmul", val1, val2, res_type)
                else:
                    # Keep non-priority ops as pass-through placeholders for now.
                    self.symbol_table[out] = val1

def generate_ir(func_ast, grid, tensor_metadata, debug=False):
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                # Create function signature
                input_types = []
                output_types = []
                
                inputs = []
                outputs = []
                
                for name, meta in tensor_metadata.items():
                    shape = meta["shape"]
                    # Hardcode types for now to get skeletal structure
                    f32 = F32Type.get()
                    tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=ctx)
                    dram_space = Attribute.parse("#ttcore.memory_space<dram>", context=ctx)
                    h = shape[2] if len(shape) > 2 else shape[0]
                    w = shape[3] if len(shape) > 3 else shape[1] if len(shape) > 1 else shape[0]
                    t_type = MemRefType.get([1, 1, h // 32, w // 32], tile_type, memory_space=dram_space)
                    
                    if meta["is_output"]:
                        output_types.append(t_type)
                    else:
                        input_types.append(t_type)
                        
                func_type = FunctionType.get(input_types + output_types, [])
                func_op = func.FuncOp(func_ast.name, func_type, loc=Location.unknown(ctx))
                func_bb = func_op.add_entry_block()
                
                with InsertionPoint(func_bb):
                    for i in range(len(input_types)):
                        inputs.append(func_bb.arguments[i])
                    for i in range(len(output_types)):
                        outputs.append(func_bb.arguments[len(input_types) + i])
                        
                    grid_attr = Attribute.parse(f"#ttcore.grid<{grid[0]}x{grid[1]}>", context=ctx)
                    block_factors = ArrayAttr.get([])
                    indexing_maps = ArrayAttr.get([])
                    iterator_types = ArrayAttr.get([])
                    threads = ArrayAttr.get([Attribute.parse("#d2m.thread<unified>", context=ctx)])
                    
                    generic_op = d2m.GenericOp(
                        results_=[],
                        inputs=inputs,
                        outputs=outputs,
                        grid=grid_attr,
                        block_factors=block_factors,
                        indexing_maps=indexing_maps,
                        iterator_types=iterator_types,
                        threads=threads,
                        num_regions=1
                    )
                    
                    region = generic_op.regions[0]
                    block_arg_types = [arg.type for arg in inputs] + [arg.type for arg in outputs]
                    generic_bb = region.blocks.append(*block_arg_types)
                    
                    visitor = D2MASTVisitor(ctx, generic_bb, tensor_metadata, grid)
                    visitor.visit(func_ast)
                    
                    with InsertionPoint(generic_bb):
                        pass
                        
                    func.ReturnOp([])
                
    return module
