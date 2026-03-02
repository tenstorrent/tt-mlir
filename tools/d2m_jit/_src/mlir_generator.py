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
        self.output_names = [name for name, meta in tensor_metadata.items() if meta["is_output"]]
        self.outputs = [None] * len(self.output_names) # Tracks output values to yield
            
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
        from ttmlir.ir import RankedTensorType, MemRefType
        try:
            lhs = RankedTensorType(lhs_type)
        except ValueError:
            lhs = MemRefType(lhs_type)
        try:
            rhs = RankedTensorType(rhs_type)
        except ValueError:
            rhs = MemRefType(rhs_type)
            
        if lhs.shape == rhs.shape and lhs.element_type == rhs.element_type:
            return lhs
        return lhs

    def _infer_matmul_result_type(self, lhs_type, rhs_type):
        from ttmlir.ir import RankedTensorType, MemRefType
        try:
            lhs = RankedTensorType(lhs_type)
            is_tensor = True
        except ValueError:
            lhs = MemRefType(lhs_type)
            is_tensor = False
        try:
            rhs = RankedTensorType(rhs_type)
        except ValueError:
            rhs = MemRefType(rhs_type)
            
        if len(lhs.shape) >= 2 and len(rhs.shape) >= 2 and lhs.element_type == rhs.element_type:
            prefix = list(lhs.shape[:-2])
            m_dim = lhs.shape[-2]
            n_dim = rhs.shape[-1]
            if is_tensor:
                return RankedTensorType.get(prefix + [m_dim, n_dim], lhs.element_type)
            else:
                return MemRefType.get(prefix + [m_dim, n_dim], lhs.element_type, memory_space=lhs.memory_space)
        return lhs

    def _emit_linalg_generic_tile_op(self, op_name, lhs, rhs, out_val, out_type):
        from ttmlir.ir import RankedTensorType, MemRefType

        is_tensor = True
        try:
            out_type = RankedTensorType(out_type)
        except ValueError:
            out_type = MemRefType(out_type)
            is_tensor = False

        # out_type is the res_type inferred. We just use out_val
        rank = len(lhs.type.shape)
        identity_map = AffineMap.get_identity(rank)
        indexing_maps = ArrayAttr.get([
            AffineMapAttr.get(identity_map),
            AffineMapAttr.get(identity_map),
            AffineMapAttr.get(identity_map),
        ])
        iterator_types = ArrayAttr.get([StringAttr.get("parallel") for _ in range(rank)])

        # Tensor type requires creating the operation differently?
        import ttmlir.dialects.linalg as linalg
        
        if is_tensor:
            op = linalg.GenericOp([out_type], [lhs, rhs], [out_val], indexing_maps, iterator_types)
        else:
            op = linalg.GenericOp([], [lhs, rhs], [out_val], indexing_maps, iterator_types)
            
        op.attributes["indexing_maps"] = indexing_maps
        op.attributes["iterator_types"] = iterator_types
        op.attributes["operandSegmentSizes"] = DenseI32ArrayAttr.get([2, 1])

        region = op.regions[0]
        block = region.blocks.append(
            out_type.element_type,
            out_type.element_type,
            out_type.element_type
        )
        with InsertionPoint(block):
            import ttmlir.dialects.d2m as d2m
            if op_name == "add":
                tile_op = d2m.TileAddOp(
                    out_type.element_type,
                    block.arguments[0],
                    block.arguments[1],
                )
                Operation.create("linalg.yield", operands=[tile_op.result])
            elif op_name == "matmul":
                tile_op = d2m.TileMatmulOp(
                    out_type.element_type,
                    block.arguments[0],
                    block.arguments[1],
                    block.arguments[2],
                )
                Operation.create("linalg.yield", operands=[tile_op.result])
            else:
                # Keep unsupported ops structurally valid for now.
                Operation.create("linalg.yield", operands=[block.arguments[2]])

        if is_tensor:
            return op.result
        else:
            return None
        
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
                    from ttmlir.ir import IntegerAttr, IntegerType, IndexType
                    idx_type = IndexType.get(self.ctx)
                    import ttmlir.dialects.d2m as d2m
                    op = d2m.BlockIndexOp(IntegerAttr.get(IntegerType.get_signless(64), idx), results=[idx_type])
                    self.core_indices[name] = op.result
                    
            elif isinstance(val.func, ast.Name) and val.func.id == "alloc":
                # Handle alloc() -> memref.alloc
                with Location.unknown(self.ctx), InsertionPoint(self.block):
                    # Let's keep `memref` for local buffers for now.
                    from ttmlir.ir import Type, F32Type, Attribute, MemRefType
                    f32 = F32Type.get()
                    tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=self.ctx)
                    l1_space = Attribute.parse("#ttcore.memory_space<l1>", context=self.ctx)
                    res_type = MemRefType.get([1, 1, 1, 1, 1, 1], tile_type, memory_space=l1_space)
                    
                    # Create memref.alloc operation directly
                    import ttmlir.dialects.memref as memref
                    op = memref.AllocOp(res_type, [], [])
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
                
                from ttmlir.ir import Type, F32Type, Attribute, MemRefType
                # Mock up memref type for now based on slice sizes
                f32 = F32Type.get()
                tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=self.ctx)
                l1_space = Attribute.parse("#ttcore.memory_space<l1>", context=self.ctx)
                
                # Ensure sizes are evaluated if they are constants
                y_dim = 1
                x_dim = 1
                if isinstance(y_size, int): y_dim = max(1, y_size // 32)
                if isinstance(x_size, int): x_dim = max(1, x_size // 32)
                
                # Get the memref logical shape to match the shard shape!
                # Since physical tensor is 6D, we make local buffer 6D. Wait, getDeviceLayout failed!
                # Actually, the error is: `d2m.remote_load' op failed to get device layout from memref/tensor
                # This means it couldn't find a DeviceLayoutInterface on the `memref`.
                # We need to make sure the memory space or layout has it.
                # Oh! Earlier I saw `DeviceLayoutInterface` is the `layout` attribute, but since `#ttcore.metal_layout` is in `memory_space`, it didn't find it!
                # Wait, if I put it in `layout`, python bindings crash because `layout` doesn't accept it.
                # How does `getDeviceLayout(MemRefType)` find it?
                # It calls `memref.getLayout()`. Wait, if it checks `memref.getLayout()`, then the layout MUST be set on the memref type as the layout, not memory space.
                # BUT when I printed a real dump, `#layout = ... : ... memref<2x2x... , #layout>` ! The `#layout` is the memory space! Wait!
                # MemRefType format is `memref<shapexelement_type, layout, memory_space>`
                # If there's only one attribute after element type, is it layout or memory space?
                # If it's `#layout`, it's an attribute! In MLIR, if the attribute is a MemRefLayoutAttrInterface, it's layout.
                # Is `#ttcore.metal_layout` a layout or a memory space?
                # Let's check `include/ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td`.
                
                # Let's keep `memref` for local buffers for now.
                # Let's keep `memref` for local buffers for now.
                from ttmlir.ir import Type, F32Type, Attribute, MemRefType
                import ttmlir.dialects.memref as memref
                l1_space = Attribute.parse("#ttcore.memory_space<l1>", context=self.ctx)
                # Ensure the tile_type is defined
                f32 = F32Type.get()
                tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=self.ctx)
                res_type = MemRefType.get([1, 1, 1, 1, y_dim, x_dim], tile_type, memory_space=l1_space)
                op = memref.AllocOp(res_type, [], [])
                
                # Generate d2m.remote_load
                import ttmlir.dialects.d2m as d2m_dialect
                # We need to unwrap the CB type and use wait/reserve ops to get the memref for d2m.generic
                # Actually wait, `d2m.remote_load` DOES have an explicit CB form where the source is the generic op's argument.
                # Look at `d2m.remote_load` usage in explicit CB form:
                # `d2m.remote_load %generic_operand[%i, %j] into %cb`
                # But wait, in `test_eltwise_add_chain.py`, the user does `d2m.remote_load(local_a, a[...])`.
                # If `a` is a block argument with type `!d2m.cb<memref<...>>`, we can't pass it to `remote_load` as `$memref`.
                # Wait, if `a` is the generic op operand (which is a global dram memref), it SHOULD NOT be a CB.
                # The arguments to `d2m.generic` region are CBs *for local L1 usage*, but what if they are used for remote ops?
                # Ah! In `test_eltwise_add_chain.mlir` previous dump:
                # `^bb0(%arg4: memref<...dram...>, %arg5: memref<...dram...>)`
                # If the generic op has `memref` region block arguments, `remote_load` works.
                # BUT the compiler gave the error:
                # `'d2m.generic' op all regions must either cb or semaphore block argument type`
                # So we changed them to `!d2m.cb`.
                # But if they are `!d2m.cb<memref<...dram...>>`, `remote_load` expects `memref<...dram...>` as its `memref` argument.
                # Wait, how does `remote_load` work when block arguments MUST be CBs?
                # Does `remote_load` take the CB as the `$cb` argument and NO `memref` argument?
                # Let's look at `RemoteLoadOp` builders:
                # `OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "Value":$cb)`
                # It requires BOTH `memref` AND `cb`?
                # Wait, look at `getODSOperandIndexAndLength`: $localBuffer is optional.
                # So `d2m.remote_load` is: `localBuffer`, `memref`, `indices`, `cb`, etc.
                # If `$memref` MUST be a `memref`, where does this `memref` come from?
                # In MLIR, if the block argument is `!d2m.cb`, you can't just pass it as `memref`.
                # Wait, look at the error again:
                # `'d2m.remote_load' op operand #1 must be ranked tensor of any type values or non-0-ranked.memref of any type values, but got '!d2m.cb<memref<...>>'`
                # So it's passing `!d2m.cb` as operand #1 (`$memref`).
                # If `a` is the global operand outside the `d2m.generic`, we could use it... but we're inside the region, so we can't use outer values.
                # Wait! "Loads an _entire shard_ from remote or local GenericOp operand into a local L1 buffer. The memref/tensor argument _must_ be an operand of the GenericOp."
                # Does it mean we pass the *outer* generic op's operand?
                # YES. "The memref/tensor argument _must_ be an operand of the GenericOp."
                # So inside the `d2m.generic` region, we should just use the outer operands directly!
                # Wait, if we use the outer operands, they are implicitly captured in the region. Does `d2m.generic` allow implicitly captured operands?
                # In standard MLIR, regions can implicitly capture values from the outer scope if `IsolatedFromAbove` is not specified.
                # Let's check `D2M_GenericOp` definition. It has `OpTrait::VariadicRegions`, `OpTrait::VariadicResults`, `OpTrait::ZeroSuccessors`, `OpTrait::VariadicOperands`... no `IsolatedFromAbove`!
                # So YES! We should just use the original outer operands instead of the block arguments!
                # But wait, why did `d2m.generic` have block arguments at all then?
                # In `d2m_insert_dst_dumps/insert_dst_register_access_eltwise.after.mlir`, the generic op region arguments are:
                # `^unified0(%cb0: !d2m.cb<...l1...>, %cb1: !d2m.cb<...l1...>, %cb2: !d2m.cb<...l1...>):`
                # And it does NOT do `remote_load`. It does `d2m.wait %cb0`.
                # This is an explicit CB form where the data is ALREADY in L1 (managed by the pipeline)!
                # But in our `d2m.jit` script, we are writing explicit data movement!
                # So the `d2m.generic` is in "Unified Explicit form".
                # In Unified Explicit form, the generic op has NO block arguments?
                # Wait, if we don't put block arguments, the number of block arguments won't match the number of operands?
                # Actually, does `d2m.generic` require block arguments to match operands?
                # The error was: `'d2m.generic' op all regions must either cb or semaphore block argument type`
                # If we have 0 block arguments, does it fail? "all regions must..." -> trivially true for 0 block arguments!
                mcast_shape_attr = ArrayAttr.get([]) # Empty array for no mcast
                mcast_dims_attr = ArrayAttr.get([]) # Empty array for no mcast
                mcast_start_index = ArrayAttr.get([])
                
                # Unwrap the cb type for the memref type
                import ttmlir.dialects.d2m as d2m_dialect
                # Note: src_val is a block argument with type !d2m.cb<memref<...>>
                # The remote_load should have the signature:
                # d2m.remote_load localBuffer src_val[y, x] : memref<...>, !d2m.cb<memref<...>> -> memref<...>
                # Wait, looking at the dump, remote_load doesn't have CB argument in tensor form, but in memref form it might.
                # Actually, the error says:
                # 'd2m.remote_load' op operand #1 must be ranked tensor of any type values or non-0-ranked.memref of any type values, but got '!d2m.cb<memref<...>>'
                # This implies the operand #1 (which is `memref` argument) should be the raw memref, and `cb` is a separate optional argument?
                # No, if the generic op's arguments are CBs, we might need to extract the underlying type for the result, or pass the CB directly.
                # Ah! Wait. D2M_RemoteLoadOp allows `memref` to be `AnyRankedTensorOrMemRef`.
                # If we pass CB, we must pass it to the `$cb` argument, NOT the `$memref` argument!
                # Wait, if we pass it to `$cb`, what do we pass to `$memref`? 
                # "The explicit CB form takes a CB block arg as additional input; the load is produced into the CB... In explicit CB form, no local buffer is required."
                # Wait, that's if we are loading *into* a CB!
                # "d2m.remote_load %generic_operand[%i, %j] into %cb" -> Here %generic_operand is the source, %cb is the destination.
                # But in our case, we are loading FROM a generic operand (which is a block argument) INTO a local buffer!
                # Wait, if the generic op's block argument is a CB, does that mean the generic op's operand is passed as a CB?
                # Yes! In our generated MLIR:
                # `^bb0(%arg4: !d2m.cb<memref<...>>, ...):`
                # So the source is `%arg4`.
                # But `remote_load` expects the source to be a memref. So how do we read from a CB?
                # Wait, if `%arg4` is the block argument of `d2m.generic`, should it be a memref instead of CB?
                # Let's check `test_eltwise_add_chain.mlir` dump:
                # In the real dump, generic region block arguments are:
                # `^bb0(%in: !ttcore.tile<32x32, f32>, ...)` -> Wait, that's `linalg.generic`!
                # We don't have `d2m.generic` in that dump because `d2m_matmul_ir_pipeline_dump.mlir` doesn't use `d2m.generic`?
                op = d2m_dialect.RemoteLoadOp(res_type, src_val, [self.core_indices["core_y"], self.core_indices["core_x"]], mcast_start_index, mcast_shape_attr, mcast_dims_attr, localBuffer=dest_val)
                
                # Store the loaded value in symbol table for destination variable
                
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
                    # For tensor semantics, we pass the out buffer as init tensor and update symbol table!
                    out_val = self.symbol_table[out]
                    op_result = self._emit_linalg_generic_tile_op("add", val1, val2, out_val, res_type)
                    if op_result is not None:
                        self.symbol_table[out] = op_result
                elif op_name == "matmul":
                    res_type = self._infer_matmul_result_type(val1.type, val2.type)
                    out_val = self.symbol_table[out]
                    op_result = self._emit_linalg_generic_tile_op("matmul", val1, val2, out_val, res_type)
                    if op_result is not None:
                        self.symbol_table[out] = op_result
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
                input_metal_types = []
                output_metal_types = []
                
                inputs = []
                outputs = []
                
                for name, meta in tensor_metadata.items():
                    shape = meta["shape"]
                    # Hardcode types for now to get skeletal structure
                    f32 = F32Type.get()
                    h = shape[2] if len(shape) > 2 else shape[0]
                    w = shape[3] if len(shape) > 3 else shape[1] if len(shape) > 1 else shape[0]
                    
                    grid_attr = Attribute.parse(f"#ttcore.grid<{grid[0]}x{grid[1]}>", context=ctx)
                    
                    # Create ttnn.ttnn_layout tensor type for function signature
                    rank = len(shape)
                    dims = ", ".join([f"d{i}" for i in range(rank)])
                    affine_map_str = f"({dims}) -> ({dims})"
                    
                    memref_shape = list(shape)
                    memref_shape[-2] = memref_shape[-2] // 32
                    memref_shape[-1] = memref_shape[-1] // 32
                    memref_shape_str = "x".join(map(str, memref_shape))
                    
                    ttnn_layout_str = f"#ttnn.ttnn_layout<{affine_map_str}, <1x1>, memref<{memref_shape_str}x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>>"
                    shape_str = "x".join(map(str, shape))
                    func_arg_type_str = f"tensor<{shape_str}xf32, {ttnn_layout_str}>"
                    func_arg_type = Type.parse(func_arg_type_str, context=ctx)
                    
                    # Prepare ttcore.metal_layout memref type for d2m.generic
                    t_shape = [1, 1, h // 32, w // 32]
                    logical_shape_str = f"logical_shape={grid[0]}x{grid[1]}x{t_shape[0]}x{t_shape[1]}x{h}x{w}"
                    dim_alignments_str = f"dim_alignments=1x1x1x1x32x32"
                    
                    # Create tensor with metal layout
                    metal_type_str = f"tensor<{grid[0]}x{grid[1]}x1x1x{h // 32}x{w // 32}x!ttcore.tile<32x32, f32>, #ttcore.metal_layout<{logical_shape_str}, {dim_alignments_str}, collapsed_intervals=dense<> : tensor<0x2xi64>, undef, dram, sharded, index_map = (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>>"
                    metal_type = Type.parse(metal_type_str, context=ctx)
                    
                    if meta["is_output"]:
                        output_types.append(func_arg_type)
                        output_metal_types.append(metal_type)
                    else:
                        input_types.append(func_arg_type)
                        input_metal_types.append(metal_type)
                        
                func_type = FunctionType.get(input_types + output_types, [])
                func_op = func.FuncOp(func_ast.name, func_type, loc=Location.unknown(ctx))
                func_bb = func_op.add_entry_block()
                
                with InsertionPoint(func_bb):
                    # Insert ttir.ttnn_metal_layout_cast for inputs and outputs
                    all_metal_types = input_metal_types + output_metal_types
                    for i in range(len(input_types) + len(output_types)):
                        arg = func_bb.arguments[i]
                        metal_type = all_metal_types[i]
                        cast_op = Operation.create(
                            "ttir.ttnn_metal_layout_cast",
                            results=[metal_type],
                            operands=[arg],
                            loc=Location.unknown(ctx)
                        )
                        if i < len(input_types):
                            inputs.append(cast_op.result)
                        else:
                            outputs.append(cast_op.result)
                        
                        
                    grid_attr = Attribute.parse(f"#ttcore.grid<{grid[0]}x{grid[1]}>", context=ctx)
                    block_factors = ArrayAttr.get([])
                    indexing_maps = ArrayAttr.get([])
                    iterator_types = ArrayAttr.get([])
                    threads = ArrayAttr.get([Attribute.parse("#d2m.thread<unified>", context=ctx)])
                    
                    generic_op = Operation.create(
                        "d2m.generic",
                        results=[],
                        operands=inputs + outputs,
                        attributes={
                            "grid": grid_attr,
                            "block_factors": block_factors,
                            "indexing_maps": indexing_maps,
                            "iterator_types": iterator_types,
                            "threads": threads,
                            "operandSegmentSizes": DenseI32ArrayAttr.get([len(inputs), len(outputs)])
                        },
                        regions=1
                    )
                    
                    region = generic_op.regions[0]
                    
                    block_arg_types = []
                    for arg in inputs + outputs:
                        # Oh wait, `!d2m.cb` might need to wrap `memref`, but if we use `tensor`, it wraps `tensor`?
                        # Let's just wrap `arg.type` which is now a `tensor`.
                        cb_type = Type.parse(f"!d2m.cb<{arg.type}>", context=ctx)
                        block_arg_types.append(cb_type)
                    generic_bb = region.blocks.append(*block_arg_types)
                    
                    visitor = D2MASTVisitor(ctx, generic_bb, tensor_metadata, grid)
                    # We need to map AST name node src_name to the MLIR variable.
                    # Since remote_load expects memref, not cb, and the arguments to the block are cbs,
                    # we must use the outer operands of the d2m.generic op for the memref argument!
                    for i, arg in enumerate(inputs + outputs):
                        if i < len(inputs):
                            name = list(tensor_metadata.keys())[i]
                            visitor.symbol_table[name] = arg
                        else:
                            name = list(tensor_metadata.keys())[len(inputs) + (i - len(inputs))]
                            visitor.symbol_table[name] = arg
                            
                    visitor.visit(func_ast)
                    
                    with InsertionPoint(generic_bb):
                        pass
                        
                    func.ReturnOp([])
                
    return module
