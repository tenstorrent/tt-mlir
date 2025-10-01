# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from ttmlir.ir import *
from ttmlir.dialects import (
    ttir,
    func,
    ttnn,
    tensor,
    ttcore,
)

from .utils import _discover_dialect_ops, _get_num_pos_args


class TTIRCompiler(ast.NodeVisitor):
    common_nodes = [
        ast.Module,
        ast.FunctionDef,
        ast.Return,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.Expr,
        ast.Call,
        ast.Attribute,
        ast.Assign,
    ]

    def __init__(self, *args, **kwargs):
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.symbol_tables = []
        self.tensor_args = kwargs.get("_tensor_args", {})
        self.backend = kwargs.get("_backend")
        self.max_grid = kwargs.get("_max_grid")
        self._fn_map = _discover_dialect_ops(self.backend)
        self.supported_nodes = self.common_nodes

        # In ttnn mode, should always use ttnn ops instead.
        if self.backend == "metal":
            self.supported_nodes.extend(
                [
                    ast.UnaryOp,
                    ast.BinOp,
                    ast.Compare,
                ]
            )

    def _mlir_dtype_from_ttnn_dtype(self, dtype):
        match int(dtype):
            case 0:
                return BF16Type.get(self.ctx)
            case 1:
                return F32Type.get(self.ctx)
            case 2:
                return U32Type.get(self.ctx)
            case 5:
                return U8Type.get(self.ctx)
            case 6:
                return U16Type.get(self.ctx)
            case 7:
                return I32Type.get(self.ctx)
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _ttcore_dtype_from_ttnn_dtype(self, dtype):
        match int(dtype):
            case 0:
                return ttcore.DataType.BFloat16
            case 1:
                return ttcore.DataType.Float32
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _ttcore_dtype_from_mlir_dtype(self, dtype):
        match str(dtype):
            case "f32":
                return ttcore.DataType.Float32
            case "bf16":
                return ttcore.DataType.BFloat16
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _var_exists(self, var_name):
        for sym_table in reversed(self.symbol_tables):
            if var_name in sym_table:
                return sym_table
        return {}

    def _create_get_device_op(self):
        mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(self.ctx, 1, 1)
        mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(self.ctx, 0, 0)
        return ttnn.get_device(mesh_shape=mesh_shape_attr, mesh_offset=mesh_offset_attr)

    def _create_tensor_layout(self, tensor_arg):
        # Only rank 2 tensors supported
        assert len(tensor_arg.shape) == 2

        shard_spec = tensor_arg.memory_config().shard_spec
        shard_shape = shard_spec.shape

        # Create identity affine map, should be based of tensor shape
        # default to rank 2, don't support shape collapse.
        identity_map = AffineMap.get_identity(2, self.ctx)

        # Create ttcore grid atttr based off max_grid passed by user
        # Can't pull grid info from tensor unless it's sharded
        grid_size_x = self.max_grid[0] + 1
        grid_size_y = self.max_grid[1] + 1
        grid = ttcore.ir.GridAttr.get(self.ctx, [grid_size_x, grid_size_y])

        # Create memref, tile type only.
        # Only L1 buffer supported. NO DRAM yet.
        data_type = self._ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
        shard_shape_tile_x = shard_shape[0] // 32
        shard_shape_tile_y = shard_shape[1] // 32
        tile_type = ttcore.ir.TileType.get(self.ctx, 32, 32, data_type)
        buffer_type = ttnn.ir.BufferTypeAttr.get(self.ctx, ttnn.BufferType.L1)
        memref = MemRefType.get(
            [shard_shape_tile_x, shard_shape_tile_y], tile_type, None, buffer_type
        )

        # Either L1 block sharded or DRAM interleaved. (No DRAM for now)
        ttnn_layout = ttnn.ir.TTNNLayoutAttr.get(
            self.ctx,
            identity_map,
            grid,
            memref,
            ttnn.TensorMemoryLayout.BlockSharded,
            None,
        )
        return ttnn_layout

    def visit_Module(self, node):
        # Set default basic block
        with InsertionPoint(self.insert_point), Location.unknown():
            for stmt in node.body:
                self.visit(stmt)

    def visit_Return(self, node):
        # TODO: handle more than one return, i.e. tuples, expressions etc.
        if node.value:
            # Visit the return value and return it
            return_value = self.visit(node.value)
            func.ReturnOp([return_value])
        else:
            # Empty return
            func.ReturnOp([])

    def visit_Expr(self, node):
        # NOTE: will catch function calls and expressions where return values not used.
        return self.visit(node.value)

    def visit_FunctionDef(self, node):
        input_types = []
        for arg in node.args.args:
            name = arg.arg
            if name in self.tensor_args:
                tensor_arg = self.tensor_args[name]
                shape = list(tensor_arg.shape)
                layout = self._create_tensor_layout(tensor_arg)
                dtype = self._mlir_dtype_from_ttnn_dtype(tensor_arg.dtype)
                tensor_type = RankedTensorType.get(shape, dtype, layout)
                input_types.append(tensor_type)

        # TODO: how to dynamically figure out output shape?
        output_types = [input_types[0]]

        self.func_entry = func.FuncOp(name=node.name, type=(input_types, output_types))
        func_bb = self.func_entry.add_entry_block()

        symbol_table = {}
        for i in range(len(func_bb.arguments)):
            symbol_table[node.args.args[i].arg] = func_bb.arguments[i]

        self.symbol_tables.append(symbol_table)
        with InsertionPoint(func_bb), Location.unknown():
            self.device = self._create_get_device_op()
            for target in node.body:
                self.visit(target)

        self.symbol_tables.pop()

    def visit_Call(self, node):
        # If function is an attribute, it's a ttnn op call (eg: ttnn.exp(tensor))
        if isinstance(node.func, ast.Attribute):
            return self.visit(node.func, args=node.args)

        # Assumption: first arg is always a tensor, and shape is same for input/output
        # input params usually look like (result_type, input1, input2, output, *other_args) for ttir,
        # edge case of of passing *other_args does not work (yet) -> eg: max and min have a keep_dim arg.
        assert node.func.id in self._fn_map, f"Function {node.func.id} not supported"
        assert self.backend != "ttnn", "ttnn backend only supports ttnn ops."
        arg = self.visit(node.args[0])
        result_type = arg.type
        output = ttir.empty(result_type)

        func_args = [result_type]
        for arg in node.args:
            func_args.append(self.visit(arg))
        func_args.append(output)

        func = self._fn_map[node.func.id]
        op = func(*func_args)
        return op

    def visit_Attribute(self, node, args=[]):
        assert self.backend == "ttnn", "Attributes are only supported for ttnn backend"
        assert len(args) >= 1, "Must pass at least one argument (tensor)"
        assert node.attr in self._fn_map, f"Function {node.attr} not supported"
        assert not isinstance(
            args[0], ast.Constant
        ), "First argument cannot be a constant"

        arg = self.visit(args[0])
        result_type = arg.type

        func_args = [result_type, arg]
        for func_arg in args[1:]:
            arg = self.visit(func_arg, tensor_type=result_type)
            func_args.append(arg)

        func = self._fn_map[node.attr]
        op = func(*func_args)
        op.owner.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(self.ctx)

        # Binary ops have 3 pos args: [result_type, lhs, rhs].
        if _get_num_pos_args(func) == 3:
            dtype = ttcore.ir.DataTypeAttr.get(
                self.ctx, self._ttcore_dtype_from_mlir_dtype(result_type.element_type)
            )
            op.owner.attributes["dtype"] = dtype
        return op

    # Expressions
    # I don't think this respects BEDMAS..
    def visit_BinOp(self, node):
        lconst = isinstance(node.left, ast.Constant)
        rconst = isinstance(node.right, ast.Constant)
        assert not (lconst and rconst), "Do not BinOp two constants."
        shape = None
        lhs = None
        rhs = None
        if lconst:
            rhs = self.visit(node.right)
            shape = rhs.type.shape
            lhs = self.visit(node.left, tensor_shape=shape)
        elif rconst:
            lhs = self.visit(node.left)
            shape = lhs.type.shape
            rhs = self.visit(node.right, tensor_shape=shape)
        else:
            lhs = self.visit(node.left)
            rhs = self.visit(node.right)
            assert (
                lhs.type == rhs.type
            ), "We don't know how to figure out output shape yet :("

        if not lhs or not rhs:
            raise ValueError("Binary operands not found")

        # TODO: Once again, how do we get the output type shape?
        result_type = lhs.type
        output = ttir.empty(result_type)
        match node.op:
            case ast.Add():
                return ttir.add(result_type, lhs, rhs, output)
            case ast.Sub():
                return ttir.subtract(result_type, lhs, rhs, output)
            case ast.Mult():
                return ttir.multiply(result_type, lhs, rhs, output)
            case ast.Div():
                return ttir.div(result_type, lhs, rhs, output)
            case ast.Mod():
                return ttir.remainder(result_type, lhs, rhs, output)
            case ast.Pow():
                return ttir.pow(result_type, lhs, rhs, output)
            case ast.BitXor():
                return ttir.bitwise_xor(result_type, lhs, rhs, output)
            case ast.BitOr():
                return ttir.bitwise_or(result_type, lhs, rhs, output)
            case ast.BitAnd():
                return ttir.bitwise_and(result_type, lhs, rhs, output)
            case ast.MatMult():
                return ttir.matmul(result_type, lhs, rhs, output)
            case _:
                raise NotImplementedError(f"Unsupported binary operator: {node.op}")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if not operand:
            raise ValueError("Unary operand not found")
        output = ttir.empty(operand.type)
        match node.op:
            case ast.USub():
                return ttir.neg(operand.type, operand, output)
            case ast.Not():
                return ttir.logical_not(operand.type, operand, output)
            case ast.Invert():
                return ttir.bitwise_not(operand.type, operand, output)
            case _:
                raise NotImplementedError(f"Unsupported unary operator: {node.op}")

    def visit_Compare(self, node):
        assert len(node.comparators) == 1, "Only one comparison supported"
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        if not lhs or not rhs:
            raise ValueError("Binary comparison operands not found")

        for op in node.ops:
            output = ttir.empty(lhs.type)
            match op:
                case ast.Lt():
                    return ttir.lt(lhs.type, lhs, rhs, output)
                case ast.LtE():
                    return ttir.le(lhs.type, lhs, rhs, output)
                case ast.Gt():
                    return ttir.gt(lhs.type, lhs, rhs, output)
                case ast.GtE():
                    return ttir.ge(lhs.type, lhs, rhs, output)
                case ast.Eq():
                    return ttir.eq(lhs.type, lhs, rhs, output)
                case ast.NotEq():
                    return ttir.ne(lhs.type, lhs, rhs, output)
                case _:
                    raise NotImplementedError(f"Unsupported compare operator: {op}")

    # Statements
    def visit_Assign(self, node):
        # If there is any sort of scope change (control flow), this will just break.
        assert len(node.targets) == 1, "Only single assignments supported"
        name = node.targets[0].id
        value = self.visit(node.value)

        if isinstance(value, BlockArgument):
            raise ValueError("Why are you assigning a block argument?")

        sym_table = self.symbol_tables[-1]
        sym_table[name] = value

    # Literals
    def visit_Name(self, node):
        var_name = node.id
        existing_var_table = self._var_exists(var_name)
        if existing_var_table:
            return existing_var_table[var_name]

        return None

    def visit_Constant(self, node, tensor=None):
        assert tensor is not None, "Tensor must be provided for constants"
        element_type = tensor.type.element_type
        if not isinstance(element_type, FloatType) and not isinstance(
            element_type, BFloat16Type
        ):
            raise NotImplementedError(f"Unsupported constant type: {type(node.value)}")

        type_attr = FloatAttr.get(element_type, node.value)
        if self.backend == "ttnn":
            if not isinstance(tensor.type.element_type, FloatType):
                raise NotImplementedError(
                    f"Unsupported constant type: {type(node.value)}"
                )

            dtype = ttcore.ir.DataTypeAttr.get(
                self.ctx, self._ttcore_dtype_from_mlir_dtype(element_type)
            )
            shape = ttnn.ir.ShapeAttr.get(self.ctx, tensor.type.shape)

            # Extract layout from tensor encoding if present
            layout_attr = None
            if tensor.type.encoding:
                layout = ttnn.ir.TTNNLayoutAttr.maybe_downcast(tensor.type.encoding)
                layout_attr = ttnn.ir.LayoutAttr.get(
                    self.ctx, layout.memory_layout_as_int
                )

            # FullOp only supports I32, F32. Might need switch to ConstantOp
            return ttnn.FullOp(
                tensor.type,
                shape,
                type_attr,
                device=self.device,
                dtype=dtype,
                layout=layout_attr,
            )
        else:
            raise NotImplementedError("metal backend not supported for constants")
            # tensor_type = RankedTensorType.get(tensor.type.shape, type_attr.type)
            # dense_attr = DenseElementsAttr.get_splat(tensor.type, type_attr)
            # return ttir.ConstantOp(tensor.type, dense_attr)

    def visit(self, node: ast.AST, **kwargs):
        if any(
            isinstance(node, supported_node) for supported_node in self.supported_nodes
        ):
            # Figure out which node to visit. Not using super().visit() in order to pass kwargs.
            method_name = "visit_" + node.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)

            params = inspect.signature(visitor).parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
            if filtered_kwargs:
                return visitor(node, **filtered_kwargs)
            else:
                return visitor(node)
        else:
            raise NotImplementedError(f"visit {type(node).__name__} not supported")
