# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import functools

from ttmlir.ir import *
from ttmlir.dialects import ttcore, ttkernel, func, scf, arith, memref, emitc, d2m
from ttmlir.dialects._ods_common import get_default_loc_context
from ttmlir.passes import pykernel_compile_pipeline, ttkernel_to_cpp

from .kernel_types import ClassRegistry, Arguments
from .base_ast import PyKernelAstBase
from .utils import _discover_dialect_ops, _cleanup_source_code, _cast, _get_type_str


class TTCompilerBase(PyKernelAstBase):
    def __init__(self, name, kernel_type=None, *args, **kwargs):
        assert kernel_type in [
            None,
            "datamovement",
            "noc",
            "compute",
        ], "Invalid kernel type"
        self.supported_nodes = [
            # Variables
            ast.Name,
            ast.Load,
            ast.Store,
            # control-flow
            ast.If,
            ast.For,
            # Async
            ast.Yield,
            ast.Await,
            # Literals
            ast.Constant,
            # Expressions
            ast.Attribute,
            ast.Expr,
            ast.IfExp,
            ast.Call,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.Invert,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            # Subscripting
            ast.Subscript,
            ast.Attribute,
            ast.List,
            ast.Tuple,
            # Statements
            ast.Pass,
            ast.Assign,
            ast.AugAssign,
            ast.AnnAssign,
            # Function-and-class-definitions
            ast.Module,
            ast.FunctionDef,
            ast.arguments,
            ast.arg,
        ]

        self.name = name
        default_context = get_default_loc_context()
        self.ctx = default_context if default_context is not None else Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.symbol_tables = []
        self.module_symbol_table = None
        self.kernel_type = kernel_type

        self.args = args
        self.ct_args = {}
        self.rt_args = None

        for arg in args:
            if hasattr(arg, "value") and hasattr(arg, "key"):
                # This is a CompileTimeValue
                self.ct_args[arg.key] = arg.value

        # Get rid of appended metadata sent into compiler
        self.verbose = kwargs.get("_verbose", False)
        self.source_code = kwargs.get("_source_code", "")

    # Control Flow
    def visit_If(self, node):
        # NOTE: else-if blocks are not supported in SCF dialect
        # NOTE: Only handling Compare for if statements right now
        # TODO: if cond can be: Name, Expr, Compare, Call, UnaryOp, BoolOp
        # assert isinstance(
        #     node.test, ast.Compare
        # ), "Only Compare supported in if statements"
        if_cond = self.visit(node.test)
        cond_type = None

        if hasattr(if_cond, "result"):
            if_cond = if_cond.result

        if hasattr(if_cond, "type") and isinstance(if_cond.type, memref.MemRefType):
            if_cond = memref.LoadOp(
                if_cond, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
            cond_type = if_cond.type
        elif hasattr(if_cond, "type") and isinstance(if_cond.type, IntegerType):
            cond_type = if_cond.type
        elif isinstance(if_cond, arith.ConstantOp):
            cond_type = if_cond.type

        # Create C-Style comparison if cond_type is not None
        if cond_type is None or not isinstance(cond_type, IntegerType):
            raise ValueError("Cannot Compare Non-Integer Values")

        if cond_type.width != 1:
            # Turn into comparison to make sure value is not 0
            if_cond = arith.cmpi(
                arith.CmpIPredicate.ne, if_cond, arith.ConstantOp(cond_type, 0)
            )
        # if_cond = arith.TruncIOp(IntegerType.get_signless(1), if_cond) # temporary since expr not implemented yet
        if_exp = scf.IfOp(cond=if_cond, hasElse=bool(node.orelse))

        with InsertionPoint(if_exp.then_block), Location.unknown():
            self.symbol_tables.append({})
            for stmt in node.body:
                self.visit(stmt)
            scf.YieldOp([])
            self.symbol_tables.pop()

        if node.orelse:
            with InsertionPoint(if_exp.else_block), Location.unknown():
                self.symbol_tables.append({})
                for stmt in node.orelse:
                    self.visit(stmt)
                scf.YieldOp([])
                self.symbol_tables.pop()

    def visit_For(self, node):
        assert node.iter.func.id == "range", "Only range() supported in for loops"

        if len(node.iter.args) == 1:
            lower_bound = arith.ConstantOp(IndexType.get(self.ctx), 0)
            upper_bound = self.visit(node.iter.args[0])
            step = arith.ConstantOp(IndexType.get(self.ctx), 1)
        elif len(node.iter.args) == 2:
            lower_bound = self.visit(node.iter.args[0])
            upper_bound = self.visit(node.iter.args[1])
            step = arith.ConstantOp(IndexType.get(self.ctx), 1)
        elif len(node.iter.args) == 3:
            lower_bound = self.visit(node.iter.args[0])
            upper_bound = self.visit(node.iter.args[1])
            step = self.visit(node.iter.args[2])

        if isinstance(lower_bound.type, memref.MemRefType):
            lower_bound = memref.LoadOp(
                lower_bound, arith.ConstantOp(IndexType.get(self.ctx), 0)
            )
        if isinstance(upper_bound.type, memref.MemRefType):
            upper_bound = memref.LoadOp(
                upper_bound, arith.ConstantOp(IndexType.get(self.ctx), 0)
            )
        if isinstance(step.type, memref.MemRefType):
            step = memref.LoadOp(step, arith.ConstantOp(IndexType.get(self.ctx), 0))

        if self.verbose:
            comment = self._get_source_comment_block(node)
            emitc.verbatim(comment, [])

        for_op = scf.ForOp(lower_bound, upper_bound, step)
        with InsertionPoint(for_op.body), Location.unknown():
            self.symbol_tables.append({})

            # Add the iterator into the symbol_table
            self.symbol_tables[-1][node.target.id] = for_op.induction_variable

            for stmt in node.body:
                self.visit(stmt)
            scf.YieldOp([])
            self.symbol_tables.pop()

    # Async
    def visit_Yield(self, node):
        if isinstance(node.value, ast.Name):
            yield_args = [self.visit(node.value)]
        elif isinstance(node.value, ast.Tuple):
            yield_args = [self.visit(elem) for elem in node.value.elts]
        else:
            raise NotImplementedError(f"Unsupported type for yield {ast.dump(node)}")
        d2m.YieldOp(yield_args)

    def visit_Await(self, node):
        if isinstance(node.value, ast.Name):
            await_args = [self.visit(node.value)]
        elif isinstance(node.value, ast.Tuple):
            await_args = [self.visit(elem) for elem in node.value.elts]
        else:
            raise NotImplementedError("Unsupported type for yield")
        d2m.AwaitOp(await_args)

    # Statements
    def visit_Name(self, node):
        var_name = node.id

        # NOTE: some kernelops require passing return type as arg
        if var_name == "int":
            return IntegerType.get_signless(32, self.ctx)

        existing_var_table = self._var_exists(var_name)
        if existing_var_table:
            return existing_var_table[var_name]

        return None

    def visit_Assign(self, node):
        # Loosely support slice + tuple assignment for rt_args
        assert len(node.targets) == 1, "Only single assignments supported"
        if isinstance(node.targets[0], ast.Tuple):
            # Make sure that these are being assigned from rt_args
            if self.rt_args is None:
                raise NotImplementedError(
                    "Tuple Assignment except for rt_args not supported."
                )

            if (
                isinstance(node.value, ast.Subscript)
                and node.value.value.id == self.rt_args.arg
            ):
                _tuple = node.targets[0]
                _vars = [self.visit(elt) for elt in _tuple.elts]
                values = self.visit(node.value)
                if not isinstance(values, list):
                    raise ValueError(
                        f"Not enough values to unpack from rt_args slice (expected {len(_vars)}, got 1)"
                    )
                if len(values) != len(_vars):
                    raise ValueError(
                        f"Not enough values to unpack from rt_args slice (expected {len(_vars)}, got {len(values)})"
                    )
                # Since we are unpacking a tuple, types can't be assigned here:
                sym_table = self.symbol_tables[-1]
                for i in range(len(_vars)):
                    sym_table[_tuple.elts[i].id] = values[i]

                # Exit out of function now
                return

        var = self.visit(node.targets[0])
        value = self.visit(node.value)
        sym_table = self.symbol_tables[-1]

        # Handle Subscript Assignment here
        if isinstance(node.targets[0], ast.Subscript):
            # Var will contain a memref.LoadOp here, we can access the memref and write to it
            memref.StoreOp(value, var.memref, var.indices)
            return

        var_name = node.targets[0].id

        if hasattr(var, "type") and isinstance(var.type, MemRefType):
            memref.StoreOp(value, var, [arith.ConstantOp(IndexType.get(self.ctx), 0)])
        else:
            sym_table[var_name] = value

    def visit_AnnAssign(self, node):
        # NOTE: TTKernel types can not be used with memrefs
        var = self.visit(node.target)
        value = self.visit(node.value)
        sym_table = self.symbol_tables[-1]
        var_name = node.target.id

        # Check the annotation for array creation
        if isinstance(node.annotation, ast.List):
            # Syntax is [dtype, *shape]
            if not len(node.annotation.elts) >= 2 or not isinstance(
                node.annotation.elts[0], ast.Name
            ):
                raise ValueError(
                    "Array Initialization must follow [dtype, *shape] syntax."
                )
            # dtype = self.visit(node.annotation.elts[0])
            # We are creating a list with the shape from elts now, use dynamic types to allow for creating variadic index memrefs
            var_type = IntegerType.get_signless(32, self.ctx)

            if all(isinstance(elt, ast.Constant) for elt in node.annotation.elts[1:]):
                # Strictly constant, easiest case. Onus is on the user to not create arrays with other weird values.
                memref_type = MemRefType.get(
                    [elt.value for elt in node.annotation.elts[1:]], var_type
                )
                sym_table[var_name] = memref.alloca(memref_type, [], [])
                return
            else:
                raise NotImplementedError(
                    "Not possible to use dynamic dimensions in EmitC."
                )

        if hasattr(value, "type") and isinstance(value.type, MemRefType):
            raise ValueError(
                "Not allowed to AnnAssign to another AnnAssign'ed variable. Temporary fix is to just add 0 to the variable."
            )

        if not var:
            var_type = value.type
            memref_type = MemRefType.get([1], var_type)
            var = memref.alloca(memref_type, [], [])
            sym_table[var_name] = var
        else:
            assert isinstance(var, MemRefType), "Can not AnnAssign to non-memref types"

        memref.StoreOp(value, var, [arith.ConstantOp(IndexType.get(self.ctx), 0)])

    def visit_AugAssign(self, node):
        target = self.visit(node.target)

        # Target must already be defined in the scope of the symbol table
        if not target:
            raise ValueError(
                "AugAssign can only Assign to values that have been defined"
            )

        value = self.visit(node.value)
        sym_table = self.symbol_tables[-1]

        if not isinstance(target.type, memref.MemRefType):
            raise ValueError("Can not AugAssign to non-memref types")

        _target = memref.LoadOp(
            target, arith.ConstantOp(IndexType.get(self.ctx), 0)
        ).result

        # Determine the operation based on the type of AugAssign
        match node.op:
            case ast.Add():
                result = arith.AddIOp(_target, value)
            case ast.Sub():
                result = arith.SubIOp(_target, value)
            case ast.Mult():
                result = arith.MulIOp(_target, value)
            case _:
                raise NotImplementedError(
                    f"AugAssign operation {type(node.op).__name__} not supported"
                )

        # Store the result back to the target
        memref.StoreOp(result, target, [arith.ConstantOp(IndexType.get(self.ctx), 0)])

    # Function calls
    def visit_Call(self, node):
        def _load_func_arg(func_arg):
            if not func_arg:
                raise ValueError(f"Function argument not found for {node.func.id}")
            if hasattr(func_arg, "type") and isinstance(
                func_arg.type, memref.MemRefType
            ):
                func_arg = memref.LoadOp(
                    func_arg, arith.ConstantOp(IndexType.get(self.ctx), 0)
                )
            return func_arg

        if not isinstance(node.func, ast.Attribute):
            # print is special case to handle string formatting
            if node.func.id == "print":
                return self.visit_Print(node.args)

            # if not an Attribute, it's just a kernel api call.
            assert (
                node.func.id in self._fn_map
            ), f"Function {node.func.id} not supported"
            func = self._fn_map[node.func.id]
            args_as_attr = [False] * len(node.args)
            if type(func) is tuple:
                func, args_as_attr = func
            func_args = []
            assert len(node.args) == len(args_as_attr)
            for arg, as_attr in zip(node.args, args_as_attr):
                arg._ttkernel_as_attr = as_attr
                func_arg = _load_func_arg(self.visit(arg))
                func_args.append(func_arg)
            kwargs = {}
            for kw in node.keywords:
                kwargs[kw.arg] = _load_func_arg(self.visit(kw.value))
            return func(*func_args, **kwargs)  # type checking will occur downstream
        else:
            func_args = []
            for arg in node.args:
                func_arg = _load_func_arg(self.visit(arg))
                func_args.append(func_arg)
            kwargs = {}
            for kw in node.keywords:
                kwargs[kw.arg] = _load_func_arg(self.visit(kw.value))
            self.visit(node.func, func_args=func_args, kwargs=kwargs)  # visit_Attribute

    def visit_Print(self, node):
        fmt = ""
        argv = []
        for arg in node:
            # handles printing vars, eg: print(x)
            if isinstance(arg, ast.Name):
                fmt += "{} "
                argv.append(self.visit(arg))
            # handles printing constants, eg: print("hello world")
            elif isinstance(arg, ast.Constant):
                fmt += str(arg.value) + " "
            # handles printing format strings, eg: print("hello {}".format(x))
            elif isinstance(arg, ast.Call):
                fmt += arg.func.value.value + " "
                for arg in arg.args:
                    argv.append(self.visit(arg))
            else:
                raise NotImplementedError(
                    f"Print argument {type(arg).__name__} not supported"
                )

        fmt = fmt.strip() + "\\n"
        ttkernel.dprint(fmt, argv)

    # Expressions
    def visit_Expr(self, node):
        # NOTE: will catch function calls and expressions where return values not used.
        return self.visit(node.value)

    def visit_BoolOp(self, node):
        values = [self.visit(arg) for arg in node.values]

        # Make sure that each of the values are booleans
        for i in range(len(values)):
            value = values[i]
            value_type = None
            if hasattr(value, "type") and isinstance(value.type, memref.MemRefType):
                value = memref.LoadOp(
                    value, arith.ConstantOp(IndexType.get(self.ctx), 0)
                ).result
                value_type = value.type
            elif hasattr(value, "type") and isinstance(value.type, IntegerType):
                value_type = value.type
            elif isinstance(value, arith.ConstantOp):
                value_type = value.type

            if value_type is None:
                raise ValueError(
                    "BoolOp values must be MemRef, ConstantOp, or IntegerType"
                )

            if not isinstance(value_type, IntegerType):
                raise ValueError(
                    "BoolOp values must be MemRef or ConstantOp of IntegerType"
                )

            if value_type.width != 1:
                # Set the value to 1 if not equal to 0, otherwise 0. This is the C-style way
                values[i] = arith.cmpi(
                    arith.CmpIPredicate.ne, value, arith.ConstantOp(value_type, 0)
                )

        # Chain all of the comparisons together
        def _match_bool_op(lhs, rhs):
            # We will know and assume LHS and RHS are booleans
            match (node.op):
                case ast.And():
                    return arith.andi(lhs, rhs)
                case ast.Or():
                    return arith.ori(lhs, rhs)
                case _:
                    raise NotImplementedError(f"BoolOp {node.op} not supported")

        # Atleast 2 Ops must exist in BoolOp
        chained_op = _match_bool_op(values[0], values[1])

        # Chain all of the remaining values
        for i in range(2, len(values)):
            chained_op = _match_bool_op(chained_op, values[i])

        return chained_op

    def visit_BinOp(self, node):
        # TODO: need to load MemRef types when using variables
        # TODO: need to handle float, unsigned, and signed operations
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not lhs or not rhs:
            raise ValueError("Binary operands not found")

        # load variable if needed
        if isinstance(lhs, OpView):
            lhs = lhs.result

        if isinstance(rhs, OpView):
            rhs = rhs.result

        if isinstance(lhs.type, memref.MemRefType):
            lhs = memref.LoadOp(
                lhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
        if isinstance(rhs.type, memref.MemRefType):
            rhs = memref.LoadOp(
                rhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result

        if lhs.type != rhs.type:
            rhs = _cast(rhs, lhs.type)
        assert lhs.type == rhs.type, f"{lhs.type} != {rhs.type}"
        mlir_type = _get_type_str(lhs.type)

        def qualified_or(attr, otherwise, *args, **kwargs):
            qualified_object_syntax = f"{mlir_type}.{attr}"
            fn = self._fn_map.get(qualified_object_syntax, otherwise)
            return fn(*args, **kwargs)

        def unimplemented(*args, **kwargs):
            raise NotImplementedError(f"{node.op} not implemented")

        match (node.op):
            case ast.Add():
                return qualified_or("__add__", arith.addi, lhs, rhs)
            case ast.Sub():
                return qualified_or("__sub__", arith.subi, lhs, rhs)
            case ast.Mult():
                return qualified_or("__mul__", arith.muli, lhs, rhs)
            case ast.Div():
                return qualified_or("__truediv__", unimplemented, lhs, rhs)
            case ast.MatMult():
                return qualified_or("__matmul__", unimplemented, lhs, rhs)
            case ast.FloorDiv():
                return qualified_or("__floordiv__", arith.floordivsi, lhs, rhs)
            case ast.Mod():
                return qualified_or("__mod__", arith.remsi, lhs, rhs)
            case ast.Pow():
                return qualified_or("__pow__", unimplemented, lhs, rhs)
            case ast.LShift():
                return qualified_or("__lshift__", arith.shli, lhs, rhs)
            case ast.RShift():
                return qualified_or("__rshift__", arith.shrsi, lhs, rhs)
            case ast.BitOr():
                return qualified_or("__or__", arith.ori, lhs, rhs)
            case ast.BitAnd():
                return qualified_or("__and__", arith.andi, lhs, rhs)
            case ast.BitXor():
                return qualified_or("__xor__", arith.xori, lhs, rhs)
            case _:
                raise NotImplementedError(
                    f"Binary operator {type(node.op).__name__} not implemented"
                )

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if not operand:
            raise ValueError("Unary operand not found")

        if isinstance(operand.type, memref.MemRefType):
            operand = memref.LoadOp(
                operand, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result

        match (node.op):
            # need to expose emitc for these unary operators, not sure if this is necessary yet
            case ast.USub():
                return emitc.UnaryMinusOp(operand.type, operand)
            case ast.UAdd():
                return emitc.UnaryPlusOp(operand.type, operand)
            case ast.Not():
                # Must return a 1-bit Signless Integer (bool)
                return emitc.logical_not(IntegerType.get_signless(1, self.ctx), operand)
            case ast.Invert():
                return emitc.bitwise_not(operand.type, operand)
            case _:
                raise NotImplementedError(
                    f"Unary operator {type(node.op).__name__} not implemented"
                )

    def visit_Compare(self, node):
        assert len(node.ops) == 1, "Only single operators supported"
        assert len(node.comparators) == 1, "Only single comparators supported"
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        if not lhs or not rhs:
            raise ValueError("Compare operands not found")

        if isinstance(lhs.type, memref.MemRefType):
            lhs = memref.LoadOp(lhs, arith.ConstantOp(IndexType.get(self.ctx), 0))
        if isinstance(rhs.type, memref.MemRefType):
            rhs = memref.LoadOp(rhs, arith.ConstantOp(IndexType.get(self.ctx), 0))

        if lhs.type != rhs.type:
            rhs = _cast(rhs, lhs.type)
        assert lhs.type == rhs.type, f"{lhs.type} != {rhs.type}"

        match (node.ops[0]):
            case ast.Eq():
                return arith.cmpi(arith.CmpIPredicate.eq, lhs, rhs)
            case ast.NotEq():
                return arith.cmpi(arith.CmpIPredicate.ne, lhs, rhs)
            case ast.Gt():
                return arith.cmpi(arith.CmpIPredicate.sgt, lhs, rhs)
            case ast.GtE():
                return arith.cmpi(arith.CmpIPredicate.sge, lhs, rhs)
            case ast.Lt():
                return arith.cmpi(arith.CmpIPredicate.slt, lhs, rhs)
            case ast.LtE():
                return arith.cmpi(arith.CmpIPredicate.sle, lhs, rhs)
            case _:
                raise NotImplementedError(
                    f"Compare operator {type(node.ops).__name__} not implemented"
                )

    # Subscript Value
    def visit_Subscript(self, node):
        # rt_args has been defined, we know that the id of the array to reference from is "rt_args"
        if self.rt_args is not None and node.value.id == self.rt_args.arg:
            # Get index from slice and ensure it's a single integral value
            if isinstance(node.slice, ast.Constant):
                # Now we have a single integral constant here, construct and return the get_arg_val call.
                arg_index = self.visit(node.slice)
                int_type = IntegerType.get_signless(32, self.ctx)
                return ttkernel.get_arg_val(int_type, arg_index)
            else:
                # Iterate from slice to generate rt_arg calls
                # Upper and Lower must be defined, step can be inferred as 1
                _step = 1 if node.slice.step is None else node.slice.step.value
                _lower = 0 if node.slice.lower is None else node.slice.lower.value
                if node.slice.upper is None:
                    raise IndexError("Runtime Arg Slices must have Upper Bound defined")
                result = []
                for i in range(_lower, node.slice.upper.value, _step):
                    arg_index = self.visit(ast.Constant(i))
                    int_type = IntegerType.get_signless(32, self.ctx)
                    result.append(ttkernel.get_arg_val(int_type, arg_index))
                return result
        elif node.value.id == "ct_args":
            # TODO(vprajapati): error checking, support slicing
            # assume only single integer values is passed into subscript for now
            ct_args_index = node.slice.value
            ct_args_value = self.ct_args[ct_args_index]
            if isinstance(ct_args_value, bool):
                # have to look for bool first, or else it'll be picked up as an integer :/
                return arith.ConstantOp(
                    IntegerType.get_signless(1, self.ctx), ct_args_value
                )
            elif isinstance(ct_args_value, int):
                return arith.ConstantOp(
                    IntegerType.get_signless(32, self.ctx), ct_args_value
                )
            else:
                raise TypeError("ct_args must be int or bool")

        # Now process accessing elements from array types
        # Accesses are done through numpy style tuple indices or constants
        tbl = self._var_exists(node.value.id)

        if not tbl:
            raise ValueError("Array doesn't exist.")

        arr = tbl[node.value.id]

        # Make sure this is a memref type
        if not hasattr(arr, "type") or not (
            isinstance(arr.type, MemRefType) or isinstance(arr.type, RankedTensorType)
        ):
            raise ValueError("Can only subscript Arrays")

        # Ensure slice is valid
        # Visit the slice, if not Tuple or constant
        def _build_index(slice, shape, bounds_check_idx):
            if hasattr(slice, "value"):
                if slice.value >= shape[bounds_check_idx]:
                    raise IndexError("Index out of bounds.")
                return arith.ConstantOp(IndexType.get(self.ctx), slice.value)
            else:
                # Forcefully cast to IndexType if we have a BinOp, etc...
                # Allow MLIR to take care of errors.
                r = self.visit(slice)
                if isinstance(r.type, IndexType):
                    return r
                else:
                    return arith.IndexCastOp(IndexType.get(self.ctx), r)

        # Make sure that the Constant checks up against rank type
        if isinstance(node.slice, ast.Constant):
            # Make sure Constant checks up against rank
            if arr.type.rank < 1:
                raise IndexError("Can only index elements of Array, rank < 1")

            # Check against bounds
            if arr.type.shape[0] <= node.slice.value:
                raise IndexError("Index out of bounds.")

            idx = _build_index(node.slice, arr.type.shape, 0)
        elif isinstance(node.slice, ast.Tuple):
            # Check Rank
            if len(node.slice.elts) > arr.type.rank:
                raise IndexError("Can only index elements of Array, rank >= len(index)")

            idx = [
                _build_index(elt, arr.type.shape, i)
                for i, elt in enumerate(node.slice.elts)
            ]
        else:
            idx = _build_index(slice, arr.type.shape, 0)

        if isinstance(arr.type, MemRefType):
            return memref.LoadOp(arr, idx)
        elif isinstance(arr.type, RankedTensorType):
            return (arr, idx)

    def visit_Attribute(self, node, func_args=[], kwargs={}):
        # type name should be !ttkernel.* if it has attributes
        mlir_value = self._var_exists(node.value.id)[node.value.id]
        mlir_type = _get_type_str(mlir_value.type)
        qualified_object_syntax = f"{mlir_type}.{node.attr}"
        fn = self._fn_map.get(qualified_object_syntax, None)
        if fn is not None:
            fn(mlir_value, *func_args, **kwargs)
            return
        elif not mlir_type.startswith("!ttkernel."):
            raise ValueError(
                f"{node.value.id} is not a ttkernel type, thus can not have attributes."
            )
        # ignore the '!' at the start of the type name
        type_name = mlir_type[1:]

        if ClassRegistry.exists(type_name):
            # Instantiate class and call its emit_mlir method.
            func_args = [mlir_value] + func_args
            attr_class = ClassRegistry.get(type_name)()
            attr_class.emit_mlir(node.attr, func_args)
        else:
            raise ValueError(
                f"{node.value.id} has no attributes. Did you define a PyKernelAttributesBase subclass?"
            )
        return

    def visit_List(self, node):
        # Snoop List for nested loops and get size
        def snoop_list(node):
            result_arr = []
            result_shape = []
            sz = 0

            if any(isinstance(elt, ast.List) for elt in node.elts) and not all(
                isinstance(elt, ast.List) for elt in node.elts
            ):
                # The shape is not consistent, we will raise an error here:
                raise NotImplementedError("All nested arrays must be of same size.")

            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    tbl = self._var_exists(elt.id)
                    elt = tbl[elt.id]
                    if hasattr(elt, "type") and isinstance(elt.type, MemRefType):
                        if elt.type.rank > 1 or elt.type.shape[0] != 1:
                            raise NotImplementedError(
                                "Creating Arrays with Pre-Defined Nested Arrays Not Supported."
                            )
                    sz += 1
                    result_arr.append(
                        memref.LoadOp(
                            elt, arith.ConstantOp(IndexType.get(self.ctx), 0)
                        ).result
                    )
                elif isinstance(elt, ast.List):
                    size, arr = snoop_list(elt)
                    if not result_shape:
                        result_shape = size
                    elif size != result_shape:
                        raise NotImplementedError(
                            "All nested arrays must be of same size."
                        )
                    sz += 1
                    result_arr.append(arr)
                elif isinstance(elt, ast.Constant):
                    elt = self.visit(elt)
                    sz += 1
                    result_arr.append(elt.result)
                else:
                    elt = self.visit(elt)
                    if (
                        not hasattr(elt, "type")
                        or not isinstance(elt.type, IntegerType)
                        or elt.type.width != 32
                    ):
                        raise ValueError("Array element must be an integer type")
                    result_arr.append(elt)
                    sz += 1
            # Collect the size and result_shape
            return ([sz] + result_shape), result_arr

        # Need to deal with nested loops, determine the shape from filled array
        shape, array = snoop_list(node)

        # empty sz catch case
        if shape == [0]:
            raise NotImplementedError(
                "Array object must be filled, otherwise use AnnAssign."
            )

        # Create the memref, must be of integral type.
        # TODO(vprajapati): Consider implementing arrays of other types
        var_type = IntegerType.get_signless(32, self.ctx)
        memref_type = MemRefType.get(shape, var_type)
        var = memref.alloca(memref_type, [], [])

        # Populate the table
        def populate_list(arr, _idx=[]):
            nonlocal var
            for i, elt in enumerate(arr):
                idx = _idx + [arith.ConstantOp(IndexType.get(self.ctx), i)]
                if isinstance(elt, list):
                    populate_list(elt, idx)
                else:
                    memref.StoreOp(elt, var, idx)

        populate_list(array)

        return var

    def visit_Tuple(self, node):
        return tuple(map(self.visit, node.elts))

    # Literals
    def visit_Constant(self, node):
        as_attr = getattr(node, "_ttkernel_as_attr", False)
        op_constructor = IntegerAttr.get if as_attr else arith.ConstantOp
        if callable(as_attr):
            return as_attr(node)
        elif isinstance(node.value, bool):
            return op_constructor(IntegerType.get_signless(1, self.ctx), node.value)
        elif isinstance(node.value, int):
            return op_constructor(IntegerType.get_signless(64, self.ctx), node.value)
        else:
            raise NotImplementedError(
                f"constant type {type(node.value).__name__} not implemented"
            )


class TTKernelCompiler(TTCompilerBase):
    def __init__(self, name, kernel_type=None, *args, **kwargs):
        super().__init__(name, kernel_type, *args, **kwargs)
        self._fn_map = _discover_dialect_ops(ttkernel)
        self.supported_nodes.append(ast.Return)
        self._fn_map["get_compile_time_arg_val"] = (
            ttkernel.get_compile_time_arg_val,
            [True, True],
        )  # True for arg as attribute

    # Root Nodes
    def visit_FunctionDef(self, node):
        # TODO: add alloca args name into symbol table
        assert not self.func_entry, "Cannot declare function within a function"

        rt_args = []
        common_rt_args = []
        ct_args = []
        cb_args = []
        cb_idx = []
        operand_idx = 0
        for i in range(len(node.args.args)):
            # We know that all cb_args will be annotated with CircularBuffer
            # After that we will have the positional rt_args
            # Finally we will have the kwarg ct_args, we have to intelligently parse all of these
            arg = node.args.args[i]

            if not arg.annotation:
                # This is a runtime arg now, wire it up into the function statement
                # Add the name and the index

                # This must be inputted as a RuntimeArgument, initialize as such, if it's an int it's always not common rt_args
                if isinstance(self.args[i], int):
                    # This is not a common_rt_arg for sure
                    rt_args.append((arg.arg, len(rt_args)))
                elif isinstance(self.args[i], Arguments):
                    _arg = self.args[i]
                    if _arg.is_common:
                        common_rt_args.append((arg.arg, len(common_rt_args)))
                    else:
                        rt_args.append((arg.arg, len(rt_args)))
                else:
                    raise TypeError(
                        "Got Positional Argument in IR, unexpected argument type provided."
                    )
                continue
            elif arg.annotation.id == "CompileTimeValue":
                # This is a CT Arg, we can package the metadata needed for passing this value in
                if arg.arg not in self.ct_args:
                    raise ValueError(
                        f"Argument {arg.arg} not provided into kernel call."
                    )
                ct_args.append((arg.arg, self.ct_args[arg.arg]))
                continue
            elif not arg.annotation.id == "CircularBuffer":
                raise TypeError(f"cannot pass {arg.annotation.id} to a pykernel")

            # Follow normal logic to construct CBs
            cb_arg = ttkernel.ir.ArgAttr.get(
                self.ctx, ttkernel.ArgType.CBPort.value, operand_idx
            )

            cb_args.append(cb_arg)
            cb_idx.append(i)

            operand_idx += 1

            tile_type = ttcore.ir.TileType.get(
                self.ctx, 32, 32, getattr(ttcore.DataType, self.args[i].dtype)
            )

        self.func_entry = func.FuncOp(name=node.name, type=([], []))
        # Supply cb_args as ct_args, use rt_args and ct_args "normally"
        arg_spec = ttkernel.ir.ArgSpecAttr.get(self.ctx, [], cb_args)
        self.func_entry.attributes[ttkernel.ir.ArgSpecAttr.name] = arg_spec

        if self.kernel_type:
            self.func_entry.attributes[
                ttkernel.ir.ThreadTypeAttr.name
            ] = ttkernel.ir.ThreadTypeAttr.get(self.ctx, self.kernel_type)
        func_bb = self.func_entry.add_entry_block()

        # update basic block
        self.symbol_tables.append({})
        with InsertionPoint(func_bb), Location.unknown():
            # Insert verbose comment for function, to be picked up by Compiler pass it must exist within function region
            # Need a bit of custom logic to make the function def look pretty:
            # Get the source code from the main function decl:
            if self.verbose and self.source_code:
                comment = f"// --- Python Function Declaration for Above --- \n{self._get_source_comment_block(node)}\n// -- End Function Declaration"
                emitc.verbatim(comment, [])

            # Get all of the CBs using the arg_spec attr
            for indexIndex, i in enumerate(cb_idx):
                tile_type = ttcore.ir.TileType.get(
                    self.ctx, 32, 32, getattr(ttcore.DataType, self.args[i].dtype)
                )
                cb_type = ttkernel.ir.CBType.get(
                    self.ctx, MemRefType.get(self.args[i].tilized_shape, tile_type)
                )
                res = ttkernel.get_compile_time_arg_val(cb_type, indexIndex)
                self.symbol_tables[-1][node.args.args[i].arg] = res

            # Insert a point to create all of the relevant rt_args and ct_args
            int_type = IntegerType.get_signless(32, self.ctx)
            for name, idx in rt_args:
                _idx = arith.ConstantOp(IndexType.get(self.ctx), idx)
                res = ttkernel.get_arg_val(int_type, _idx)
                self.symbol_tables[-1][name] = res

            for name, idx in common_rt_args:
                _idx = arith.ConstantOp(IndexType.get(self.ctx), idx)
                res = ttkernel.get_common_arg_val(int_type, _idx)
                self.symbol_tables[-1][name] = res

            for name, value in ct_args:
                if isinstance(value, bool):
                    res = arith.ConstantOp(IntegerType.get_signless(1, self.ctx), value)
                elif isinstance(value, int):
                    res = arith.ConstantOp(
                        IntegerType.get_signless(32, self.ctx), value
                    )
                else:
                    raise TypeError("ct_args must be int or bool")
                self.symbol_tables[-1][name] = res

            for target in node.body:
                self.visit(target)

            # TODO: Check for a return/terminator insert one if not present

        self.symbol_tables.pop()
