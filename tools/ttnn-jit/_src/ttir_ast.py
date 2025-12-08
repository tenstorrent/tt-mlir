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

from .tensor_translator import create_tensor_layout, create_output_tensor

from .utils import (
    discover_dialect_ops,
    get_num_pos_args,
)
from .conversions import (
    mlir_dtype_from_ttnn_dtype,
    ttcore_dtype_from_ttnn_dtype,
    ttcore_dtype_from_mlir_dtype,
)


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
        self._fn_map = discover_dialect_ops("ttnn")
        self.supported_nodes = self.common_nodes
        self.return_type = None

    def _var_exists(self, var_name):
        for sym_table in reversed(self.symbol_tables):
            if var_name in sym_table:
                return sym_table
        return {}

    def _create_get_device_op(self):
        mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(self.ctx, 1, 1)
        mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(self.ctx, 0, 0)
        return ttnn.get_device(mesh_shape=mesh_shape_attr, mesh_offset=mesh_offset_attr)

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
            self.return_type = return_value.type
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
                layout = create_tensor_layout(self.ctx, tensor_arg)
                dtype = mlir_dtype_from_ttnn_dtype(tensor_arg.dtype, self.ctx)
                tensor_type = RankedTensorType.get(shape, dtype, layout)
                input_types.append(tensor_type)

        # Dummy output type until we process the function body
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
        # Create a new function type with the actual output type. Move the body from the old function to the new one.
        if self.return_type is not None:
            # Get the old block before creating new function
            old_block = self.func_entry.regions[0].blocks[0]

            # Create new function with correct return type
            new_func_entry = func.FuncOp(
                name=node.name, type=(input_types, [self.return_type])
            )

            # The new function's region is empty (no blocks), so directly append the old block to it
            new_region = new_func_entry.regions[0]
            old_block.append_to(new_region)

            # Erase the old function
            self.func_entry.erase()
            self.func_entry = new_func_entry

    def visit_Call(self, node):
        # If function is an attribute, it's a ttnn op call (eg: ttnn.exp(tensor))
        if isinstance(node.func, ast.Attribute):
            return self.visit(node.func, args=node.args)

        # Assumption: first arg is always a tensor, and shape is same for input/output
        # input params usually look like (result_type, input1, input2, output, *other_args) for ttir,
        # edge case of of passing *other_args does not work (yet) -> eg: max and min have a keep_dim arg.
        assert node.func.id in self._fn_map, f"Function {node.func.id} not supported"
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
        assert len(args) >= 1, "Must pass at least one argument (tensor)"

        # Map function names to their MLIR dialect equivalents
        attr_name = node.attr
        if attr_name == "pow":
            attr_name = "pow_tensor"

        assert attr_name in self._fn_map, f"Function {node.attr} not supported"
        assert not isinstance(
            args[0], ast.Constant
        ), "First argument cannot be a constant"

        tensor_arg = self.visit(args[0])

        func_args = [tensor_arg]
        for func_arg in args[1:]:
            arg = self.visit(func_arg, tensor=tensor_arg)
            func_args.append(arg)
        result_type = create_output_tensor(
            self.ctx,
            attr_name,
            [RankedTensorType.maybe_downcast(tensor.type) for tensor in func_args],
        )
        func_args = [result_type] + func_args
        func = self._fn_map[attr_name]
        op = func(*func_args)
        op.owner.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(self.ctx)

        # Binary ops have 3 pos args: [result_type, lhs, rhs].
        if get_num_pos_args(func) == 3:
            dtype = ttcore.ir.DataTypeAttr.get(
                self.ctx, ttcore_dtype_from_mlir_dtype(result_type.element_type)
            )
            op.owner.attributes["dtype"] = dtype
        return op

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
        if isinstance(element_type, IntegerType):
            type_attr = IntegerAttr.get(I32Type.get(self.ctx), node.value)
        elif isinstance(element_type, FloatType):
            type_attr = FloatAttr.get(F32Type.get(self.ctx), node.value)
        else:
            raise NotImplementedError(f"Unsupported constant type: {type(node.value)}")

        if not isinstance(tensor.type.element_type, FloatType):
            raise NotImplementedError(
                f"Only float constants are supported, got: {type(node.value)}"
            )

        dtype = ttcore.ir.DataTypeAttr.get(
            self.ctx, ttcore_dtype_from_mlir_dtype(element_type)
        )
        shape = ttnn.ir.ShapeAttr.get(self.ctx, tensor.type.shape)

        # Extract layout from tensor encoding if present
        layout_attr = None
        if tensor.type.encoding:
            layout = ttnn.ir.TTNNLayoutAttr.maybe_downcast(tensor.type.encoding)
            layout_attr = ttnn.ir.LayoutAttr.get(self.ctx, layout.memory_layout_as_int)

        op = ttnn.FullOp(
            tensor.type,
            shape,
            type_attr,
            device=self.device,
            dtype=dtype,
            layout=layout_attr,
        )
        op.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(self.ctx)
        return op

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
