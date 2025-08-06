# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import functools
import textwrap

from ttmlir.ir import *
from ttmlir.dialects import (
    ttcore,
    ttir,
    ttkernel,
    func,
    scf,
    arith,
    memref,
    emitc,
    tensor,
)

from .ast import TTKernelCompiler, ttkernel_compile

try:
    import ttnn
except Exception as e:
    raise ImportError(
        "{e} ttnn is not found. Make sure to run `cmake --build build -- pykernel-env-setup`"
    )

Tensor = ttnn.Tensor


class TTIRCompiler(TTKernelCompiler):
    ttkernel_fn_map = {}
    supported_nodes = [
        ### Variables
        ast.Name,
        ### control-flow
        ast.If,
        ast.For,
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

        self.func_entry = func.FuncOp(name=node.name, type=(input_types, []))
        func_bb = self.func_entry.add_entry_block()

        symbol_table = {}
        for i in range(len(func_bb.arguments)):
            symbol_table[node.args.args[i].arg] = func_bb.arguments[i]

        self.symbol_tables.append(symbol_table)
        with InsertionPoint(func_bb), Location.unknown():
            for target in node.body:
                self.visit(target)

        self.symbol_tables.pop()

    def visit_Return(self, node):
        if node.value:
            # Visit the return value and return it
            return_value = self.visit(node.value)
            func.ReturnOp([return_value])
        else:
            # Empty return
            func.ReturnOp([])

    # Expressions
    # !! BEDMAS !!
    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not lhs or not rhs:
            raise ValueError("Binary operands not found")
        print(lhs, rhs)

        # TODO: Once again, how do we get the output type shape?
        print(lhs.type, rhs.type)
        assert (
            lhs.type == rhs.type
        ), "We don't know how to figure out output shape yet :("
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
        print("Hello", node.op)
        operand = self.visit(node.operand)
        if not operand:
            raise ValueError("Unary operand not found")
        output = ttir.empty(operand.type)
        match node.op:
            # case ast.UAdd():
            # return ttir.add(operand.type, operand, operand, output)
            case ast.USub():
                return ttir.neg(operand.type, operand, operand, output)
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
        # need to handle tensor assignments
        pass


def ttir_compile(verbose: bool = False):
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
            # if len(args) < 2:
            #     raise ValueError(f"Must pass at least 2 tensors: one input and one output")

            for i, arg in enumerate(args):
                tensor_args[param_names[i]] = arg
            kwargs["_tensor_args"] = tensor_args

            # Parse and compile
            m = ast.parse(source_code)
            print(ast.dump(m, indent=2) + "\n")
            b = TTIRCompiler(None, *args, **kwargs)
            b.visit(m)

            # Check if generated IR is valid
            print(b.module)
            b.module.operation.verify()

        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


# def ttir_compile(verbose: bool = False):
#     return ttkernel_compile(verbose=verbose, Compiler=TTIRCompiler)
