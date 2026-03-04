# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
from numbers import Integral
from ttmlir.ir import *
from ttmlir.dialects import func, scf, arith
from ttmlir.dialects import _d2m_ops_gen as d2m

TILE_SIZE = 32

_DTYPE_MAP = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float32": "f32",   "f32": "f32",
    "float16": "f16",   "f16": "f16",
}


def _dtype_to_mlir_str(dtype: str) -> str:
    return _DTYPE_MAP.get(str(dtype), "f32")


# ── AST pre-scan helpers ─────────────────────────────────────────────────────

def _slice_size_from_ast(slice_node):
    """Return the static element count of a slice node, or None.

    Handles patterns like ``k:k+128`` or ``(off+y):(off+y+64)``.
    """
    if not isinstance(slice_node, ast.Slice) or slice_node.upper is None:
        return None
    upper = slice_node.upper
    # Pattern: upper = lower + N  (BinOp whose right or left child is a literal)
    if isinstance(upper, ast.BinOp) and isinstance(upper.op, ast.Add):
        if isinstance(upper.right, ast.Constant) and isinstance(upper.right.value, int):
            return upper.right.value
        if isinstance(upper.left, ast.Constant) and isinstance(upper.left.value, int):
            return upper.left.value
    # Pattern: constant upper (possibly with constant lower)
    if isinstance(upper, ast.Constant) and isinstance(upper.value, int):
        lower = slice_node.lower
        if lower is None:
            return upper.value
        if isinstance(lower, ast.Constant) and isinstance(lower.value, int):
            return upper.value - lower.value
    return None


def _extract_remote_op_slice_sizes(subscript_node, sizes):
    """Update *sizes* with the (y_tiles, x_tiles) for the tensor in *subscript_node*.

    *subscript_node* is the AST node ``src[y_slice][x_slice]``.  We take the
    minimum observed slice size per tensor (handles mixed access patterns).
    """
    if not (isinstance(subscript_node, ast.Subscript) and
            isinstance(subscript_node.value, ast.Subscript)):
        return
    x_slice = subscript_node.slice
    y_slice = subscript_node.value.slice
    name_node = subscript_node.value.value
    if not isinstance(name_node, ast.Name):
        return
    name = name_node.id
    y_sz = _slice_size_from_ast(y_slice)
    x_sz = _slice_size_from_ast(x_slice)
    if y_sz is None or x_sz is None:
        return
    y_tiles = y_sz // TILE_SIZE
    x_tiles = x_sz // TILE_SIZE
    if name in sizes:
        prev_y, prev_x = sizes[name]
        sizes[name] = (min(prev_y, y_tiles), min(prev_x, x_tiles))
    else:
        sizes[name] = (y_tiles, x_tiles)


def _scan_stmts_for_slices(stmts, sizes):
    """Recursively walk *stmts* collecting remote_load/store slice sizes."""
    for stmt in stmts:
        if isinstance(stmt, ast.For):
            _scan_stmts_for_slices(stmt.body, sizes)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (isinstance(call.func, ast.Attribute) and
                    isinstance(call.func.value, ast.Name) and
                    call.func.value.id == "d2m"):
                op = call.func.attr
                if op == "remote_load" and len(call.args) >= 2:
                    _extract_remote_op_slice_sizes(call.args[1], sizes)
                elif op == "remote_store" and len(call.args) >= 1:
                    _extract_remote_op_slice_sizes(call.args[0], sizes)


def _scan_required_slice_sizes(func_body):
    """Return ``{tensor_name: (y_tiles, x_tiles)}`` for every remotely-accessed tensor."""
    sizes = {}
    _scan_stmts_for_slices(func_body, sizes)
    return sizes


# ── View-layout emission helper ───────────────────────────────────────────────

def _emit_view_layout(ctx, metal_val, f_y, f_x, sub_h, sub_w, tile_type_str,
                      logical_shape_str, dim_alignments_str,
                      memory_space_str="dram"):
    """Emit a ``d2m.view_layout`` that reblocks a metal tensor to finer virtual shards.

    Args:
        metal_val:           MLIR Value of type ``tensor<GY x GX x SH x SW x tile>``
        f_y, f_x:            Reblock factors (GY*f_y, GX*f_x is the new virtual grid)
        sub_h, sub_w:        New shard tile dimensions (SH/f_y, SW/f_x)
        tile_type_str:       e.g. ``!ttcore.tile<32x32, bf16>``
        logical_shape_str:   e.g. ``1x1x256x256``
        dim_alignments_str:  e.g. ``1x1x32x32``
        memory_space_str:    e.g. ``dram``

    The result type uses a MetalLayoutAttr (required by ViewLayoutOp::verify for
    tensor operands) with the same logical_shape, oob_val, and memory_space as the
    input.  Bufferization converts this to ``memref<...#ttcore.view<rank>>`` automatically.
    """
    metal_ranked = RankedTensorType(metal_val.type)
    orig_shape = list(metal_ranked.shape)   # [GY, GX, SH, SW]
    GY, GX = orig_shape[0], orig_shape[1]
    SH, SW = orig_shape[2], orig_shape[3]

    new_shape_str = f"{GY * f_y}x{GX * f_x}x{sub_h}x{sub_w}"

    # Remapping map: virtual (d0,d1,d2,d3) → physical coords
    map_str = (
        f"affine_map<(d0, d1, d2, d3) -> "
        f"((d0 * {sub_h} + d2) floordiv {SH}, "
        f"(d1 * {sub_w} + d3) floordiv {SW}, "
        f"(d0 * {sub_h} + d2) mod {SH}, "
        f"(d1 * {sub_w} + d3) mod {SW})>"
    )
    remapping_attr = Attribute.parse(map_str, context=ctx)

    # Result type: MetalLayoutAttr with same logical_shape, oob_val, memory_space.
    # ViewLayoutOp::verify() hard-casts result.getEncoding() to MetalLayoutAttr
    # for tensor operands, so we must use MetalLayoutAttr (not ViewLayoutAttr).
    # Bufferization later converts this to memref<...#ttcore.view<rank>> via
    # getBufferType(result, isView=true).
    result_type_str = (
        f"tensor<{new_shape_str}x{tile_type_str}, "
        f"#ttcore.metal_layout<logical_shape={logical_shape_str}, "
        f"dim_alignments={dim_alignments_str}, "
        f"collapsed_intervals=dense<[[0, -1]]> : tensor<1x2xi64>, "
        f"undef, {memory_space_str}, sharded>>"
    )
    result_type = Type.parse(result_type_str, context=ctx)

    import ttmlir.dialects.d2m as d2m_dialect
    return d2m_dialect.ViewLayoutOp(
        result=result_type,
        input=metal_val,
        remapping=remapping_attr,
    ).result


def _normalize_int_dim(value, tensor_name, axis):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"Tensor '{tensor_name}' has non-integer dimension at axis {axis}: {value!r}"
        )
    dim = int(value)
    if dim <= 0:
        raise ValueError(
            f"Tensor '{tensor_name}' has non-positive dimension at axis {axis}: {dim}"
        )
    return dim


def _get_validated_layout_dims(tensor_name, shape):
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"Tensor '{tensor_name}' shape must be a list/tuple, got {type(shape)}")
    if len(shape) < 2:
        raise ValueError(
            f"Tensor '{tensor_name}' must have rank >= 2 for tiled lowering, got rank {len(shape)}"
        )

    normalized = [_normalize_int_dim(dim, tensor_name, axis) for axis, dim in enumerate(shape)]
    height = normalized[-2]
    width = normalized[-1]
    if height % TILE_SIZE != 0 or width % TILE_SIZE != 0:
        raise ValueError(
            f"Tensor '{tensor_name}' spatial dims ({height}, {width}) must be divisible by {TILE_SIZE}"
        )

    tiled_shape = list(normalized)
    tiled_shape[-2] = height // TILE_SIZE
    tiled_shape[-1] = width // TILE_SIZE

    return {
        "shape": normalized,
        "rank": len(normalized),
        "height": height,
        "width": width,
        "tiled_shape": tiled_shape,
        "tile_h": tiled_shape[-2],
        "tile_w": tiled_shape[-1],
    }


def _is_placeholder_shape(shape):
    return all(dim == 1 for dim in shape)


class D2MASTVisitor(ast.NodeVisitor):
    def __init__(self, ctx, block, tensor_metadata, grid,
                 tile_type_str="!ttcore.tile<32x32, f32>", mlir_dtype="f32",
                 reblock_info=None):
        self.ctx = ctx
        self.block = block
        self.tensor_metadata = tensor_metadata
        self.grid = grid
        self.tile_type_str = tile_type_str
        self.mlir_dtype = mlir_dtype
        self.reblock_info = reblock_info or {}

        self.symbol_table = {}
        self.core_indices = {} # Tracks core_y, core_x
        self.scratch_slot = 0 # Tracks slot index for scratch_allocate
        self.output_names = [name for name, meta in tensor_metadata.items() if meta["is_output"]]
        self.outputs = [None] * len(self.output_names) # Tracks output values to yield
        self.tensor_layout_dims = {
            name: _get_validated_layout_dims(name, meta["shape"])
            for name, meta in tensor_metadata.items()
        }

        # Compute per-core shard shape from the first tensor
        first_layout = next(iter(self.tensor_layout_dims.values()))
        self.per_core_tile_h = first_layout["tile_h"] // grid[0]
        self.per_core_tile_w = first_layout["tile_w"] // grid[1]
        self.shard_shape = list(first_layout["shape"][:-2]) + [self.per_core_tile_h, self.per_core_tile_w]

    def _require_compile_time_int(self, value, expr_name):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"{expr_name} must be a compile-time integer, got {value!r} ({type(value).__name__})"
            )
        return int(value)

    def _get_ranked_shape_and_elem(self, value_type):
        from ttmlir.ir import RankedTensorType, MemRefType

        try:
            ranked = RankedTensorType(value_type)
            return list(ranked.shape), ranked.element_type, True, None
        except ValueError:
            ranked = MemRefType(value_type)
            return list(ranked.shape), ranked.element_type, False, ranked.memory_space

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

    def _emit_index_expr(self, node):
        """Emit arith MLIR ops for an index/scalar expression. Returns a MLIR Value.
        Must be called inside an active InsertionPoint + Location context."""
        from ttmlir.ir import IndexType, IntegerAttr
        idx_type = IndexType.get(self.ctx)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, node.value)).result
        if isinstance(node, ast.Name):
            if node.id in self.core_indices:
                return self.core_indices[node.id]
            if node.id in self.symbol_table:
                return self.symbol_table[node.id]
            return arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, 0)).result
        if isinstance(node, ast.BinOp):
            lhs = self._emit_index_expr(node.left)
            rhs = self._emit_index_expr(node.right)
            if isinstance(node.op, ast.Add):  return arith.AddIOp(lhs, rhs).result
            if isinstance(node.op, ast.Sub):  return arith.SubIOp(lhs, rhs).result
            if isinstance(node.op, ast.Mult): return arith.MulIOp(lhs, rhs).result
        return arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, 0)).result

    def _emit_slice_start(self, slice_node):
        """Emit the start index of a slice as a MLIR Value.
        Must be called inside an active InsertionPoint + Location context."""
        from ttmlir.ir import IndexType, IntegerAttr
        idx_type = IndexType.get(self.ctx)
        if isinstance(slice_node, ast.Slice) and slice_node.lower is not None:
            return self._emit_index_expr(slice_node.lower)
        return arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, 0)).result

    def _emit_virtual_index(self, slice_node, sub_elements):
        """Compute virtual grid coord = slice_start_elements / sub_elements.

        Must be called inside an active InsertionPoint + Location context.
        """
        from ttmlir.ir import IndexType, IntegerAttr
        idx_type = IndexType.get(self.ctx)
        start_val = self._emit_slice_start(slice_node)
        sub_const = arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, sub_elements)).result
        return arith.DivUIOp(start_val, sub_const).result

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

        if lhs.element_type != rhs.element_type:
            raise ValueError(
                f"d2m.add requires matching element types, got {lhs.element_type} and {rhs.element_type}"
            )
        lhs_shape = list(lhs.shape)
        rhs_shape = list(rhs.shape)
        if lhs_shape == rhs_shape:
            return lhs
        if _is_placeholder_shape(lhs_shape):
            return rhs
        if _is_placeholder_shape(rhs_shape):
            return lhs
        if lhs_shape != rhs_shape:
            raise ValueError(
                f"d2m.add requires identical operand shapes, got {lhs_shape} and {rhs_shape}"
            )
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

        if lhs.element_type != rhs.element_type:
            raise ValueError(
                f"d2m.matmul requires matching element types, got {lhs.element_type} and {rhs.element_type}"
            )
        if len(lhs.shape) < 2 or len(rhs.shape) < 2:
            raise ValueError(
                f"d2m.matmul requires rank >= 2 operands, got {len(lhs.shape)} and {len(rhs.shape)}"
            )
        if len(lhs.shape) != len(rhs.shape):
            raise ValueError(
                f"d2m.matmul requires equal ranks, got {len(lhs.shape)} and {len(rhs.shape)}"
            )
        if list(lhs.shape[:-2]) != list(rhs.shape[:-2]):
            raise ValueError(
                f"d2m.matmul requires matching batch dims, got {list(lhs.shape[:-2])} and {list(rhs.shape[:-2])}"
            )
        if lhs.shape[-1] != rhs.shape[-2]:
            raise ValueError(
                f"d2m.matmul requires contracting dims lhs[-1] == rhs[-2], got {lhs.shape[-1]} and {rhs.shape[-2]}"
            )

        prefix = list(lhs.shape[:-2])
        m_dim = lhs.shape[-2]
        n_dim = rhs.shape[-1]
        if is_tensor:
            return RankedTensorType.get(prefix + [m_dim, n_dim], lhs.element_type)
        return MemRefType.get(prefix + [m_dim, n_dim], lhs.element_type, memory_space=lhs.memory_space)

    def _emit_linalg_generic_tile_op(self, op_name, lhs, rhs, out_val, out_type):
        from ttmlir.ir import RankedTensorType, MemRefType

        lhs_shape, lhs_elem, _, _ = self._get_ranked_shape_and_elem(lhs.type)
        rhs_shape, rhs_elem, _, _ = self._get_ranked_shape_and_elem(rhs.type)

        is_tensor = True
        try:
            out_type = RankedTensorType(out_type)
        except ValueError:
            out_type = MemRefType(out_type)
            is_tensor = False

        out_shape = list(out_type.shape)
        rank = len(lhs_shape)
        if rank != len(rhs_shape):
            raise ValueError(f"{op_name} rank mismatch between lhs/rhs: {rank} vs {len(rhs_shape)}")

        # out_val is None when alloc() was used — the actual buffer is allocated here.
        out_val_is_deferred = out_val is None
        if not out_val_is_deferred:
            out_val_shape, out_val_elem, _, _ = self._get_ranked_shape_and_elem(out_val.type)
            if lhs_elem != rhs_elem or lhs_elem != out_val_elem:
                raise ValueError(
                    f"{op_name} requires matching element types across lhs/rhs/out, got {lhs_elem}, {rhs_elem}, {out_val_elem}"
                )

        if op_name == "add":
            lhs_effective = rhs_shape if _is_placeholder_shape(lhs_shape) else lhs_shape
            rhs_effective = lhs_shape if _is_placeholder_shape(rhs_shape) else rhs_shape
            if lhs_effective != rhs_effective or lhs_effective != out_shape:
                raise ValueError(
                    f"d2m.add shape mismatch lhs={lhs_shape}, rhs={rhs_shape}, out_type={out_shape}"
                )
            if not out_val_is_deferred and out_shape != out_val_shape:
                raise ValueError(
                    f"d2m.add output buffer shape mismatch expected={out_shape}, got out_val={out_val_shape}"
                )
        elif op_name == "matmul":
            if rank < 2:
                raise ValueError(f"d2m.matmul requires rank >= 2, got rank {rank}")
            if lhs_shape[:-2] != rhs_shape[:-2]:
                raise ValueError(
                    f"d2m.matmul batch dims mismatch lhs={lhs_shape[:-2]} rhs={rhs_shape[:-2]}"
                )
            if lhs_shape[-1] != rhs_shape[-2]:
                raise ValueError(
                    f"d2m.matmul contracting dim mismatch lhs[-1]={lhs_shape[-1]} rhs[-2]={rhs_shape[-2]}"
                )
            expected_out = list(lhs_shape[:-2]) + [lhs_shape[-2], rhs_shape[-1]]
            if out_shape != expected_out:
                raise ValueError(
                    f"d2m.matmul output shape mismatch expected={expected_out} out_type={out_shape}"
                )
            if not out_val_is_deferred and out_val_shape != expected_out:
                raise ValueError(
                    f"d2m.matmul output buffer shape mismatch expected={expected_out}, got out_val={out_val_shape}"
                )

        # Allocate the output buffer when alloc() was used (deferred allocation).
        if is_tensor and out_val_is_deferred:
            from ttmlir.dialects import tensor as tensor_dialect
            out_val = tensor_dialect.EmptyOp(out_shape, out_type.element_type).result

        from ttmlir.ir import AffineMap, AffineMapAttr, ArrayAttr, Attribute

        if op_name == "matmul":
            from ttmlir.ir import AffineExpr

            iter_rank = rank + 1
            batch_rank = rank - 2

            exprs = [AffineExpr.get_dim(i) for i in range(batch_rank)]
            m = AffineExpr.get_dim(batch_rank)
            n = AffineExpr.get_dim(batch_rank + 1)
            k = AffineExpr.get_dim(batch_rank + 2)

            lhs_map = AffineMap.get(iter_rank, 0, exprs + [m, k])
            rhs_map = AffineMap.get(iter_rank, 0, exprs + [k, n])
            out_map = AffineMap.get(iter_rank, 0, exprs + [m, n])

            indexing_maps = ArrayAttr.get([
                AffineMapAttr.get(lhs_map),
                AffineMapAttr.get(rhs_map),
                AffineMapAttr.get(out_map),
            ])

            iterator_types_list = ["parallel"] * batch_rank + ["parallel", "parallel", "reduction"]
            iterator_types = ArrayAttr.get(
                [
                    Attribute.parse(f"#linalg.iterator_type<{t}>", context=self.ctx)
                    for t in iterator_types_list
                ]
            )
        else:
            identity_map = AffineMap.get_identity(rank)
            indexing_maps = ArrayAttr.get([
                AffineMapAttr.get(identity_map),
                AffineMapAttr.get(identity_map),
                AffineMapAttr.get(identity_map),
            ])
            iterator_types = ArrayAttr.get(
                [
                    Attribute.parse("#linalg.iterator_type<parallel>", context=self.ctx)
                    for _ in range(rank)
                ]
            )

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
                # Defer allocation — no IR emitted here. The actual tensor.empty is
                # emitted at the first use site (remote_load or compute op) where the
                # correct shape is known.
                self.symbol_table[name] = None

        else:
            # General scalar / index expression (e.g. core_offset_y = core_y * 128)
            with Location.unknown(self.ctx), InsertionPoint(self.block):
                result = self._emit_index_expr(val)
                self.symbol_table[name] = result

    def _collect_matmul_writes(self, stmts):
        """Pre-scan loop body statements to find variables written by matmul ops.

        Matmul accumulates into its output across loop iterations (k-loop), so the
        output must be carried as a loop iter_arg to avoid SSA dominance violations
        when the accumulated result is used after the loop.  Add/eltwise ops do NOT
        accumulate, so they do not need iter_args.
        """
        written = set()
        for stmt in stmts:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "d2m":
                    op = call.func.attr
                    if op == "matmul" and len(call.args) >= 3:
                        if isinstance(call.args[2], ast.Name):
                            written.add(call.args[2].id)
            elif isinstance(stmt, ast.For):
                written.update(self._collect_matmul_writes(stmt.body))
        return written

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

        with Location.unknown(self.ctx), InsertionPoint(self.block):
            from ttmlir.ir import IndexType, IntegerAttr
            idx_type = IndexType.get(self.ctx)

            start_int = self._require_compile_time_int(start_val, "range start")
            end_int = self._require_compile_time_int(end_val, "range end")
            step_int = self._require_compile_time_int(step_val, "range step")
            if step_int == 0:
                raise ValueError("range step must be non-zero")

            start_v = arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, start_int)).result
            end_v   = arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, end_int)).result
            step_v  = arith.ConstantOp(idx_type, IntegerAttr.get(idx_type, step_int)).result

            # Find matmul output variables inside the loop body.  These accumulate across
            # loop iterations and must be loop-carried via iter_args to avoid SSA dominance
            # violations when the accumulated result is used after the loop.
            # (Elementwise ops like add completely overwrite their outputs each iteration
            # and are not needed outside the loop, so they don't require iter_args.)
            matmul_written = self._collect_matmul_writes(node.body)
            iter_arg_names = [n for n in matmul_written if n in self.symbol_table]
            # Create correctly-typed init tensors for each iter_arg.  The alloc()
            # placeholder (all-1s) would cause a type mismatch with the linalg.generic
            # result (shard-shaped), so we emit a fresh tensor.empty with the shard shape.
            from ttmlir.ir import Type
            _tile_type = Type.parse(self.tile_type_str, context=self.ctx)
            from ttmlir.dialects import tensor as tensor_dialect
            iter_arg_init_values = [
                tensor_dialect.EmptyOp(
                    [self.per_core_tile_h, self.per_core_tile_w], _tile_type
                ).result
                for _ in iter_arg_names
            ]

            loop = scf.ForOp(start_v, end_v, step_v,
                             iter_args=iter_arg_init_values if iter_arg_names else None)

            if isinstance(node.target, ast.Name):
                self.symbol_table[node.target.id] = loop.induction_variable

            # Map iter_arg names to the loop's inner_iter_args so that uses inside
            # the loop body see the loop-carried values rather than the pre-loop values.
            for i, name in enumerate(iter_arg_names):
                self.symbol_table[name] = loop.inner_iter_args[i]

            prev_block = self.block
            self.block = loop.body
            with InsertionPoint(self.block):
                for stmt in node.body:
                    self.visit(stmt)
                # Yield the (potentially updated) iter_arg values so the loop can carry them.
                yield_vals = [self.symbol_table[n] for n in iter_arg_names]
                scf.YieldOp(yield_vals)
            self.block = prev_block

            # After the loop, update symbol_table with the loop's result values
            # (the final carried values after all iterations).
            for i, name in enumerate(iter_arg_names):
                self.symbol_table[name] = loop.results[i]

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

                # Map src_name to the outer metal tensor
                src_val = self.symbol_table[src_name]

                from ttmlir.ir import Type
                tile_type = Type.parse(self.tile_type_str, context=self.ctx)

                # Compute slice dimensions in tiles
                src_layout = self.tensor_layout_dims.get(src_name)
                y_dim = src_layout["tile_h"] if src_layout else 1
                x_dim = src_layout["tile_w"] if src_layout else 1
                if isinstance(y_size, Integral):
                    y_size_int = int(y_size)
                    if y_size_int <= 0 or y_size_int % TILE_SIZE != 0:
                        raise ValueError(
                            f"remote_load y-slice size must be positive and divisible by {TILE_SIZE}, got {y_size_int}"
                        )
                    y_dim = max(1, y_size_int // TILE_SIZE)
                if isinstance(x_size, Integral):
                    x_size_int = int(x_size)
                    if x_size_int <= 0 or x_size_int % TILE_SIZE != 0:
                        raise ValueError(
                            f"remote_load x-slice size must be positive and divisible by {TILE_SIZE}, got {x_size_int}"
                        )
                    x_dim = max(1, x_size_int // TILE_SIZE)

                # Local buffer type: 2D [y_dim, x_dim] tile tensor (no memory space).
                # The metal tensor is rank-4 (grid + 2D shard) so the CB shard is 2D.
                local_shape = [y_dim, x_dim]

                # For reblocked tensors: use the sub-shard shape and virtual grid indices.
                # For non-reblocked tensors: use the slice-derived shape and element offsets.
                if src_name in self.reblock_info:
                    rb = self.reblock_info[src_name]
                    local_shape = [rb["sub_h"], rb["sub_w"]]
                    y_idx = self._emit_virtual_index(y_slice, rb["sub_h_elements"])
                    x_idx = self._emit_virtual_index(x_slice, rb["sub_w_elements"])
                else:
                    y_idx = self._emit_slice_start(y_slice)
                    x_idx = self._emit_slice_start(x_slice)

                # Create tensor.empty() as local buffer
                from ttmlir.dialects import tensor as tensor_dialect
                local_buf_op = tensor_dialect.EmptyOp(local_shape, tile_type)

                # Generate d2m.remote_load (tensor form)
                import ttmlir.dialects.d2m as d2m_dialect
                mcast_start_index = ArrayAttr.get([])
                mcast_shape_attr = ArrayAttr.get([])
                mcast_dims_attr = ArrayAttr.get([])

                op = d2m_dialect.RemoteLoadOp(
                    local_buf_op.result.type,
                    src_val,
                    [y_idx, x_idx],
                    mcast_start_index,
                    mcast_shape_attr,
                    mcast_dims_attr,
                    localBuffer=local_buf_op.result,
                )

                # Store the loaded tensor value in symbol table
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

                if dest_name in self.reblock_info:
                    rb = self.reblock_info[dest_name]
                    y_idx = self._emit_virtual_index(y_slice, rb["sub_h_elements"])
                    x_idx = self._emit_virtual_index(x_slice, rb["sub_w_elements"])
                else:
                    y_idx = self._emit_slice_start(y_slice)
                    x_idx = self._emit_slice_start(x_slice)

                store_op = d2m_dialect.RemoteStoreOp(
                    dest_val.type,
                    dest_val,
                    [y_idx, x_idx],
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
    # Pre-scan the function body to find the smallest slice sizes used in remote ops.
    # This drives view_layout reblocking when sub-shard access is needed.
    required_slice_sizes = _scan_required_slice_sizes(func_ast.body)

    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                # Create function signature
                func_arg_types = []      # ttnn tensor types for function signature
                input_metal_types = []   # metal tensor types for inputs
                output_metal_types = []  # metal tensor types for outputs
                input_ttnn_types = []    # ttnn tensor types for inputs (same as func_arg_types)
                output_ttnn_types = []   # ttnn tensor types for outputs
                input_names = []
                output_names = []

                # Extract dtype from the first tensor entry
                _first_meta = next(iter(tensor_metadata.values()))
                _mlir_dtype = _dtype_to_mlir_str(_first_meta.get("dtype", "f32"))
                _tile_type_str = f"!ttcore.tile<32x32, {_mlir_dtype}>"

                for name, meta in tensor_metadata.items():
                    layout_dims = _get_validated_layout_dims(name, meta["shape"])
                    shape = layout_dims["shape"]
                    rank = layout_dims["rank"]
                    h = layout_dims["height"]
                    w = layout_dims["width"]
                    tile_h = layout_dims["tile_h"]
                    tile_w = layout_dims["tile_w"]

                    # Per-core tile dimensions
                    per_core_tile_h = tile_h // grid[0]
                    per_core_tile_w = tile_w // grid[1]

                    # ttnn layout for function boundary
                    dims = ", ".join([f"d{i}" for i in range(rank)])
                    affine_map_str = f"({dims}) -> ({dims})"
                    memref_shape = layout_dims["tiled_shape"]
                    memref_shape_str = "x".join(map(str, memref_shape))
                    ttnn_layout_str = f"#ttnn.ttnn_layout<{affine_map_str}, <1x1>, memref<{memref_shape_str}x{_tile_type_str}, #ttnn.buffer_type<dram>>, <interleaved>>"
                    shape_str = "x".join(map(str, shape))
                    func_arg_type_str = f"tensor<{shape_str}xf32, {ttnn_layout_str}>"
                    func_arg_type = Type.parse(func_arg_type_str, context=ctx)

                    # Metal layout: correct logical_shape (original shape, not grid-prefixed)
                    # and rank-matched dim_alignments
                    logical_shape_str = "x".join(map(str, shape))
                    aligns = ["1"] * (rank - 2) + ["32", "32"]
                    dim_alignments_str = "x".join(aligns)

                    # Metal tensor shape: grid_y x grid_x x per_core_tile_h x per_core_tile_w
                    # Rank-4 (2D grid + 2D shard); collapsed_intervals=[[0,-1]] folds batch dims
                    # into the physical height dimension so getGridShape returns 2D [grid_y, grid_x].
                    metal_shape = [grid[0], grid[1], per_core_tile_h, per_core_tile_w]
                    metal_shape_str = "x".join(map(str, metal_shape))

                    metal_type_str = (
                        f"tensor<{metal_shape_str}x{_tile_type_str}, "
                        f"#ttcore.metal_layout<logical_shape={logical_shape_str}, "
                        f"dim_alignments={dim_alignments_str}, "
                        f"collapsed_intervals=dense<[[0, -1]]> : tensor<1x2xi64>, "
                        f"undef, dram, sharded>>"
                    )
                    metal_type = Type.parse(metal_type_str, context=ctx)

                    func_arg_types.append(func_arg_type)
                    if meta["is_output"]:
                        output_metal_types.append(metal_type)
                        output_ttnn_types.append(func_arg_type)
                        output_names.append(name)
                    else:
                        input_metal_types.append(metal_type)
                        input_ttnn_types.append(func_arg_type)
                        input_names.append(name)

                func_type = FunctionType.get(func_arg_types, [])
                func_op = func.FuncOp(func_ast.name, func_type, loc=Location.unknown(ctx))
                func_bb = func_op.add_entry_block()

                with InsertionPoint(func_bb):
                    inputs = []   # metal tensor values for d2m.generic ins
                    outputs = []  # metal tensor values for d2m.generic outs

                    # Cast input func args from ttnn tensor to metal tensor
                    n_inputs = len(input_metal_types)
                    for i in range(n_inputs):
                        arg = func_bb.arguments[i]
                        metal_type = input_metal_types[i]
                        cast_op = Operation.create(
                            "ttir.ttnn_metal_layout_cast",
                            results=[metal_type],
                            operands=[arg],
                            loc=Location.unknown(ctx)
                        )
                        inputs.append(cast_op.result)

                    # For outputs: create d2m.empty (ttnn type) then cast to metal tensor
                    import ttmlir.dialects.d2m as d2m_dialect
                    for i in range(len(output_metal_types)):
                        ttnn_type = output_ttnn_types[i]
                        metal_type = output_metal_types[i]
                        empty_op = d2m_dialect.EmptyOp(ttnn_type)
                        cast_op = Operation.create(
                            "ttir.ttnn_metal_layout_cast",
                            results=[metal_type],
                            operands=[empty_op.result],
                            loc=Location.unknown(ctx)
                        )
                        outputs.append(cast_op.result)

                    # === Reblocking: emit view_layout for sub-shard access patterns ===
                    # When the user's remote_load/store slices are smaller than one
                    # physical shard, we wrap the metal tensor in a d2m.view_layout that
                    # creates a finer virtual grid.  RemoteLoad/Store then use virtual
                    # grid coordinates derived from the element-level slice starts.
                    reblock_info = {}
                    for rb_i, rb_name in enumerate(input_names):
                        if rb_name not in required_slice_sizes:
                            continue
                        sub_h_t, sub_w_t = required_slice_sizes[rb_name]
                        rb_layout = _get_validated_layout_dims(rb_name, tensor_metadata[rb_name]["shape"])
                        pc_h = rb_layout["tile_h"] // grid[0]
                        pc_w = rb_layout["tile_w"] // grid[1]
                        if sub_h_t >= pc_h and sub_w_t >= pc_w:
                            continue  # full-shard access — no reblocking needed
                        f_y = pc_h // sub_h_t
                        f_x = pc_w // sub_w_t
                        rb_logical_str = "x".join(map(str, rb_layout["shape"]))
                        rb_align_str = "x".join(["1"] * (rb_layout["rank"] - 2) + ["32", "32"])
                        with Location.unknown(ctx):
                            inputs[rb_i] = _emit_view_layout(
                                ctx, inputs[rb_i], f_y, f_x, sub_h_t, sub_w_t, _tile_type_str,
                                rb_logical_str, rb_align_str
                            )
                        reblock_info[rb_name] = {
                            "sub_h": sub_h_t, "sub_w": sub_w_t,
                            "sub_h_elements": sub_h_t * TILE_SIZE,
                            "sub_w_elements": sub_w_t * TILE_SIZE,
                        }

                    for rb_i, rb_name in enumerate(output_names):
                        if rb_name not in required_slice_sizes:
                            continue
                        sub_h_t, sub_w_t = required_slice_sizes[rb_name]
                        rb_layout = _get_validated_layout_dims(rb_name, tensor_metadata[rb_name]["shape"])
                        pc_h = rb_layout["tile_h"] // grid[0]
                        pc_w = rb_layout["tile_w"] // grid[1]
                        if sub_h_t >= pc_h and sub_w_t >= pc_w:
                            continue
                        f_y = pc_h // sub_h_t
                        f_x = pc_w // sub_w_t
                        rb_logical_str = "x".join(map(str, rb_layout["shape"]))
                        rb_align_str = "x".join(["1"] * (rb_layout["rank"] - 2) + ["32", "32"])
                        with Location.unknown(ctx):
                            view_val = _emit_view_layout(
                                ctx, outputs[rb_i], f_y, f_x, sub_h_t, sub_w_t, _tile_type_str,
                                rb_logical_str, rb_align_str
                            )
                        outputs[rb_i] = view_val
                        output_metal_types[rb_i] = view_val.type
                        reblock_info[rb_name] = {
                            "sub_h": sub_h_t, "sub_w": sub_w_t,
                            "sub_h_elements": sub_h_t * TILE_SIZE,
                            "sub_w_elements": sub_w_t * TILE_SIZE,
                        }

                    grid_attr = Attribute.parse(f"#ttcore.grid<{grid[0]}x{grid[1]}>", context=ctx)

                    # Build d2m.generic in explicit datamovement form:
                    # empty block_factors, indexing_maps, iterator_types.
                    # This bypasses verifyAffineBlocking entirely (GenericOp::verify
                    # skips the affine-blocking checks when indexing_maps is empty).
                    threads = ArrayAttr.get([Attribute.parse("#d2m.thread<unified>", context=ctx)])
                    indexing_maps = ArrayAttr.get([])
                    iterator_types = ArrayAttr.get([])
                    block_factors = []

                    generic_op = d2m_dialect.GenericOp(
                        output_metal_types,  # results — one per output tensor
                        inputs,
                        outputs,
                        [],       # additionalArgs
                        grid_attr,
                        block_factors,
                        indexing_maps,
                        iterator_types,
                        threads,
                        1,        # num_regions
                        loc=Location.unknown(ctx),
                        ip=InsertionPoint(func_bb),
                    )

                    region = generic_op.regions[0]

                    # CB block arg types: !d2m.cb<tensor<shard_shape x tile_type>>
                    # Shard shape = metal tensor shape without the grid dims (first 2 dims)
                    block_arg_types = []
                    for metal_val in inputs + outputs:
                        metal_ranked = RankedTensorType(metal_val.type)
                        metal_shape_list = list(metal_ranked.shape)
                        inner_shard = metal_shape_list[2:]  # drop grid dims (first 2)
                        shard_tensor_str = "x".join(map(str, inner_shard))
                        cb_type = Type.parse(
                            f"!d2m.cb<tensor<{shard_tensor_str}x{_tile_type_str}>>",
                            context=ctx
                        )
                        block_arg_types.append(cb_type)
                    generic_bb = region.blocks.append(*block_arg_types)

                    visitor = D2MASTVisitor(ctx, generic_bb, tensor_metadata, grid, _tile_type_str, _mlir_dtype, reblock_info)

                    # Map tensor names to outer metal tensor Values (not CB block args)
                    # remote_load/remote_store use these outer operands
                    for i, name in enumerate(input_names):
                        visitor.symbol_table[name] = inputs[i]
                    for i, name in enumerate(output_names):
                        visitor.symbol_table[name] = outputs[i]

                    visitor.visit(func_ast)

                    # Add d2m.yield with the final output tensor value(s)
                    with InsertionPoint(generic_bb):
                        yield_vals = [v for v in visitor.outputs if v is not None]
                        if yield_vals:
                            d2m_dialect.YieldOp(yield_vals)

                    func.ReturnOp([])

    return module
