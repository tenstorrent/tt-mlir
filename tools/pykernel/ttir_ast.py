# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ast
import inspect
import functools
import textwrap

from ttmlir.ir import *
from ttmlir.dialects import (
    ttir,
    func,
    tensor,
)
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline, ttmetal_to_flatbuffer_file

from .ast import TTKernelCompiler

try:
    import ttnn
except Exception as e:
    raise ImportError(
        "{e} ttnn is not found. Make sure to run `cmake --build build -- pykernel-env-setup`"
    )

Tensor = ttnn.Tensor


class TTIRCompiler(TTKernelCompiler):
    ttkernel_fn_map = {
        ### Unary ops ###
        "abs": ttir.abs,
        "cbrt": ttir.cbrt,
        "cos": ttir.cos,
        "floor": ttir.floor,
        # gelu:
        "isfinite": ttir.isfinite,
        "tan": ttir.tan,
        "atan": ttir.atan,
        "tanh": ttir.tanh,
        "reciprocal": ttir.reciprocal,
        # ttir.relu
        "rsqrt": ttir.rsqrt,
        # ttir.sigmoid
        # ttir.sign
        "sin": ttir.sin,
        "sqrt": ttir.sqrt,
        # ttir.typecast
        "log": ttir.log,
        "log1p": ttir.log1p,
        "expm1": ttir.expm1,
        "exp": ttir.exp,
        "erf": ttir.erf,
        "erfc": ttir.erfc,
        ### Binary ops ###
        # ttir.logical_and
        # ttir.logical_or
        # ttir.logical_xor
        # "min" : ttir.min,  # -> doesn't work
        # "max" : ttir.max,  # edge case of passing min/max(input, output, ... other args)
        "atan2": ttir.atan2,
    }
    supported_nodes = [
        ### Variables
        ast.Name,
        ### Control-flow
        # ast.If,
        # ast.For,
        ### Literals
        ast.Constant,
        ### Expressions
        ast.Attribute,
        ast.Expr,
        ast.Call,
        ast.UnaryOp,
        ast.BinOp,
        # ast.BoolOp,
        ast.Compare,
        ### Statements
        ast.Assign,
        ### Function-and-class-definitions
        ast.Module,
        ast.FunctionDef,
        ast.Return,
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_args = kwargs.get("_tensor_args", {})

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

    def visit_FunctionDef(self, node):
        # no longer passing CBs, just tensors.
        # use the `tensor` dialect
        input_types = []
        for arg in node.args.args:
            name = arg.arg
            if name in self.tensor_args:
                tensor_arg = self.tensor_args[name]
                shape = list(tensor_arg.shape)
                dtype = self._mlir_dtype_from_ttnn_dtype(tensor_arg.dtype)
                tensor_type = RankedTensorType.get(shape, dtype)
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
            for target in node.body:
                self.visit(target)

        self.symbol_tables.pop()

    def visit_Call(self, node):
        # Assumption: first arg is always a tensor, and shape is same for input/output
        # input params usually look like (result_type, input1, input2, output, *other_args)
        # edge of of passing *other_args does not work (yet) -> eg: max and min have a keep_dim arg
        arg = self.visit(node.args[0])
        result_type = arg.type
        output = ttir.empty(result_type)
        return super().visit_Call(node, [result_type], [output])

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
            # case ast.FloorDiv():
            #     raise NotImplementedError("Floor division not supported")
            case ast.Mod():
                return ttir.remainder(result_type, lhs, rhs, output)
            case ast.Pow():
                return ttir.pow(result_type, lhs, rhs, output)
            # case ast.LShift():
            #     pass
            # case ast.RShift():
            #     pass
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
            # case ast.UAdd():
            # return ttir.add(operand.type, operand, operand, output)
            case ast.USub():
                return ttir.neg(operand.type, operand, output)
            case ast.Not():
                return ttir.logical_not(operand.type, operand, output)
            case ast.Invert():
                return ttir.bitwise_not(operand.type, operand, output)

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
    def visit_Constant(self, node, tensor_shape=[]):
        assert tensor_shape, "Tensor shape must be provided for constants"
        if isinstance(node.value, float):
            attr = FloatAttr.get(F32Type.get(self.ctx), node.value)
        else:
            raise NotImplementedError(f"Unsupported constant type: {type(node.value)}")

        tensor_type = RankedTensorType.get(tensor_shape, attr.type)
        dense_attr = DenseElementsAttr.get_splat(tensor_type, attr)
        return ttir.ConstantOp(tensor_type, dense_attr)


def ttir_compile(verbose: bool = False, to_flatbuffer_file=""):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            # TODO: make this cleanup a helper function since it's repeated
            source_code = inspect.getsource(f)
            source_code = textwrap.dedent(source_code)
            cleaned = [
                line
                for line in source_code.splitlines()
                if not line.strip().startswith("@")
            ]
            source_code = "\n".join(cleaned)

            # Pass the actual tensors as kwargs
            tensor_args = {}
            sig = inspect.signature(f)
            param_names = list(sig.parameters.keys())
            if len(param_names) != len(args):
                raise ValueError(f"How is this even possible???")

            for i, arg in enumerate(args):
                tensor_args[param_names[i]] = arg
            kwargs["_tensor_args"] = tensor_args

            # Parse and compile
            m = ast.parse(source_code)
            if verbose:
                print(ast.dump(m, indent=2) + "\n")
            b = TTIRCompiler(None, *args, **kwargs)
            b.visit(m)

            # Check if generated IR is valid
            ir = b.module
            if verbose:
                print(ir)
            ir.operation.verify()

            system_desc_path = os.getenv("SYSTEM_DESC_PATH")
            assert system_desc_path, "SYSTEM_DESC_PATH must be set."
            ttir_to_ttmetal_backend_pipeline(
                ir, f"system-desc-path={system_desc_path} override-device-shape=1,1"
            )
            if verbose:
                print("---- After ttir_to_ttmetal_backend_pipeline ----")
                print(ir)

            if to_flatbuffer_file:
                ttmetal_to_flatbuffer_file(ir, to_flatbuffer_file, {}, [])

            return ir

        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


# def ttir_compile(verbose: bool = False):
#     return ttkernel_compile(verbose=verbose, Compiler=TTIRCompiler)
