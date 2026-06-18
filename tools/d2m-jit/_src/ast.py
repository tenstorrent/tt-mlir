# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from ttmlir.ir import *
from ttmlir.dialects import (
    d2m,
    func,
    scf,
    arith,
    emitc,
)
from ttmlir.dialects._ods_common import get_default_loc_context

from .errors import D2mJitError, closest_match
from .utils import _discover_dialect_ops, _cast, _get_type_str
from .tensor_layout import Layout

_D2M_KERNEL_TYPES = {None, "datamovement", "noc", "compute", "unified"}


class D2MCompiler(ast.NodeVisitor):
    """Unified AST -> MLIR visitor for d2m_jit.

    Replaces the prior three-level PyKernelAstBase / TTCompilerBase /
    D2MGenericCompiler hierarchy. ttkernel-specific paths (rt_args/ct_args,
    print -> ttkernel.dprint, ClassRegistry TensorAccessor dispatch, source
    code emitc.verbatim comments) have been removed.
    """

    # Populated at import time by the @syntax decorator.
    _syntax = {}

    _SUPPORTED_NODES = (
        # Variables
        ast.Name,
        ast.Load,
        ast.Store,
        # Control flow
        ast.If,
        ast.For,
        # Async (d2m: yield/await emit d2m.YieldOp/AwaitOp)
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
        ast.List,
        ast.Tuple,
        # Statements
        ast.Pass,
        ast.Assign,
        ast.Return,
        # Function/module
        ast.Module,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.arguments,
        ast.arg,
    )

    def __init__(
        self,
        name,
        kernel_type=None,
        captures=None,
        *args,
        source_file=None,
        source_firstlineno=None,
        source_lines=None,
        **kwargs,
    ):
        assert kernel_type in _D2M_KERNEL_TYPES, f"Invalid kernel type {kernel_type}"

        self.name = name
        self.kernel_type = kernel_type
        self.captures = captures if captures is not None else {}
        self.args = args

        # Source metadata used to build D2mJitError messages and to pin each
        # emitted op to a file_line_col Location.
        self._source_file = source_file
        self._source_firstlineno = source_firstlineno
        self._source_lines = source_lines or []

        try:
            default_context = get_default_loc_context()
        except ValueError:
            default_context = None
        self.ctx = default_context if default_context is not None else Context()
        self.cursor = Location.unknown(self.ctx)
        self.loc = Location.name(self.name)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.module_symbol_table = None
        self.symbol_tables = []
        self.supported_nodes = list(self._SUPPORTED_NODES)
        self._fn_map = dict(self._syntax)

    # --- Error / source-location helpers ----------------------------------

    def _abs_line(self, node):
        if self._source_firstlineno is None:
            return None
        return self._source_firstlineno + getattr(node, "lineno", 1) - 1

    def _mlir_loc(self, node):
        """Return an MLIR Location for `node`, falling back to Location.name."""
        if (
            self._source_file is not None
            and self._source_firstlineno is not None
            and hasattr(node, "lineno")
        ):
            line = self._abs_line(node)
            col = getattr(node, "col_offset", 0) + 1
            return Location.file(self._source_file, line, col, self.ctx)
        return self.loc

    def _format_error(self, node, msg_or_exc, hint=None):
        if isinstance(msg_or_exc, BaseException):
            cause = msg_or_exc
            msg = str(msg_or_exc) or type(msg_or_exc).__name__
        else:
            cause = None
            msg = str(msg_or_exc)
        return D2mJitError(
            msg=msg,
            file=self._source_file,
            line=self._abs_line(node),
            col=getattr(node, "col_offset", None),
            source_lines=self._source_lines,
            snippet_line=getattr(node, "lineno", None),
            hint=hint,
            cause=cause,
        )

    def _fail(self, node, msg_or_exc, hint=None):
        raise self._format_error(node, msg_or_exc, hint=hint)

    def _local_names(self):
        names = set()
        for tbl in self.symbol_tables:
            names.update(tbl.keys())
        return names

    def _hint_for_function(self, name):
        match = closest_match(name, self._fn_map.keys())
        return f"did you mean `{match}`?" if match else None

    def _hint_for_method(self, mlir_type, attr):
        prefix = f"{mlir_type}."
        candidates = [
            k[len(prefix) :] for k in self._fn_map.keys() if k.startswith(prefix)
        ]
        match = closest_match(attr, candidates)
        return f"did you mean `{match}`?" if match else None

    def _hint_for_name(self, name):
        match = closest_match(name, self._local_names())
        return f"did you mean `{match}`?" if match else None

    # --- Symbol table helpers ----------------------------------------------

    def _var_exists(self, var_name):
        for sym_table in reversed(self.symbol_tables):
            if var_name in sym_table:
                return sym_table
        return {}

    # --- Dispatch ----------------------------------------------------------

    def visit(self, node, **kwargs):
        if not any(isinstance(node, n) for n in self.supported_nodes):
            self._fail(
                node,
                NotImplementedError(
                    f"unsupported Python syntax inside @kernel: "
                    f"{type(node).__name__}"
                ),
            )
        method_name = "visit_" + node.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        params = inspect.signature(visitor).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        loc = self._mlir_loc(node)
        try:
            with loc:
                if filtered_kwargs:
                    return visitor(node, **filtered_kwargs)
                return visitor(node)
        except D2mJitError:
            # An inner visit already pinned a deeper node; preserve it.
            raise
        except Exception as orig:
            raise self._format_error(node, orig) from orig

    # --- Module / function entry ------------------------------------------

    def visit_Module(self, node):
        with InsertionPoint(self.insert_point), Location.unknown():
            for stmt in node.body:
                self.visit(stmt)

    def _emit_entry(self, node):
        assert not self.func_entry, "Cannot declare function within a function"

        func_operand_types = []
        for i, arg in enumerate(node.args.args):
            rt_arg = self.args[i]
            if isinstance(rt_arg, Layout):
                func_operand_types.append(
                    rt_arg.build_device_tensor_type(self.ctx, blocked=True)
                )
            elif isinstance(rt_arg, int):
                func_operand_types.append(IndexType.get(self.ctx))
            else:
                raise TypeError(
                    f"Unknown kernel argument type {type(rt_arg)} for argument {arg.arg}"
                )

        self.func_entry = func.FuncOp(name=node.name, type=(func_operand_types, []))
        self.func_entry.attributes[d2m.ir.ThreadAttr.name] = d2m.ir.ThreadAttr.get(
            self.ctx, self.kernel_type
        )

        self.symbol_tables.append({})
        func_bb = self.func_entry.add_entry_block()
        for i, bb_arg in enumerate(func_bb.arguments):
            self.symbol_tables[-1][node.args.args[i].arg] = bb_arg
        self.module_symbol_table = SymbolTable(self.module.operation)

        with InsertionPoint(func_bb):
            for capture_name, val in self.captures.items():
                assert isinstance(capture_name, str)
                if isinstance(val, int):
                    self.symbol_tables[-1][capture_name] = arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    )
                else:
                    raise TypeError(
                        f"Invalid capture type for var {capture_name}: {type(val)}"
                    )

            for target in node.body:
                self.visit(target)
            func.ReturnOp([])

        self.symbol_tables.pop()

    def visit_FunctionDef(self, node):
        with self.loc:
            return self._emit_entry(node)

    def visit_AsyncFunctionDef(self, node):
        with self.loc:
            return self._emit_entry(node)

    def visit_Return(self, node):
        if node.value:
            func.ReturnOp([self.visit(node.value)])
        else:
            func.ReturnOp([])

    def visit_Expr(self, node):
        return self.visit(node.value)

    # --- Control flow ------------------------------------------------------

    def visit_If(self, node):
        if_cond = self.visit(node.test)
        if hasattr(if_cond, "result"):
            if_cond = if_cond.result

        cond_type = None
        if hasattr(if_cond, "type") and isinstance(if_cond.type, IntegerType):
            cond_type = if_cond.type
        elif isinstance(if_cond, arith.ConstantOp):
            cond_type = if_cond.type

        if cond_type is None or not isinstance(cond_type, IntegerType):
            raise ValueError("Cannot compare non-integer values")

        if cond_type.width != 1:
            if_cond = arith.cmpi(
                arith.CmpIPredicate.ne, if_cond, arith.ConstantOp(cond_type, 0)
            )

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

        def _to_index(v):
            if isinstance(v.type, IndexType):
                return v
            return arith.IndexCastOp(IndexType.get(self.ctx), v).result

        lower_bound = _to_index(lower_bound)
        upper_bound = _to_index(upper_bound)
        step = _to_index(step)

        for_op = scf.ForOp(lower_bound, upper_bound, step)
        with InsertionPoint(for_op.body), Location.unknown():
            self.symbol_tables.append({})
            self.symbol_tables[-1][node.target.id] = for_op.induction_variable
            for stmt in node.body:
                self.visit(stmt)
            scf.YieldOp([])
            self.symbol_tables.pop()

    # --- Async (d2m) -------------------------------------------------------

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
            raise NotImplementedError("Unsupported type for await")
        d2m.AwaitOp(await_args)

    # --- Statements --------------------------------------------------------

    def visit_Name(self, node):
        var_name = node.id
        if var_name == "int":
            return IntegerType.get_signless(32, self.ctx)
        sym_table = self._var_exists(var_name)
        if sym_table:
            return sym_table[var_name]
        return None

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Only single assignments supported"
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError(
                f"Assign target {type(target).__name__} not supported"
            )
        self.symbol_tables[-1][target.id] = self.visit(node.value)

    # --- Function calls ----------------------------------------------------

    def visit_Call(self, node):
        def _resolve(arg):
            # If args_as_attr supplied a callable, call it on the whole AST
            # node directly -- bypassing visit dispatch -- so attribute-typed
            # args can be expressed as `UnaryOp(USub, Constant(0.5))` (i.e.
            # `-0.5`) and similar non-bare-Constant forms. visit_Constant
            # also honors this, but only for raw Constants; doing it here
            # generalises to any AST shape the callable wants to handle.
            as_attr = getattr(arg, "_ttkernel_as_attr", False)
            if callable(as_attr):
                signature = inspect.signature(as_attr)
                if len(signature.parameters) == 2:
                    return as_attr(arg, self)
                return as_attr(arg)
            v = self.visit(arg)
            if v is None:
                self._fail(
                    node,
                    NameError(
                        f"could not resolve argument to function "
                        f"'{getattr(node.func, 'id', '?')}'"
                    ),
                )
            return v

        if not isinstance(node.func, ast.Attribute):
            if node.func.id not in self._fn_map:
                self._fail(
                    node,
                    NameError(f"unknown function '{node.func.id}' in kernel scope"),
                    hint=self._hint_for_function(node.func.id),
                )
            fn = self._fn_map[node.func.id]
            args_as_attr = [False] * len(node.args)
            kwargs_as_attr = {}
            if isinstance(fn, tuple):
                if len(fn) == 2:
                    fn, args_as_attr = fn
                else:
                    fn, args_as_attr, kwargs_as_attr = fn
                if args_as_attr is None:
                    args_as_attr = [False] * len(node.args)
            if len(node.args) != len(args_as_attr):
                self._fail(
                    node,
                    TypeError(
                        f"function '{node.func.id}' takes "
                        f"{len(args_as_attr)} positional args; "
                        f"got {len(node.args)}"
                    ),
                )
            func_args = []
            for arg, as_attr in zip(node.args, args_as_attr):
                arg._ttkernel_as_attr = as_attr
                func_args.append(_resolve(arg))
            kwargs = {}
            for kw in node.keywords:
                kw.value._ttkernel_as_attr = kwargs_as_attr.get(kw.arg, False)
                kwargs[kw.arg] = _resolve(kw.value)
            return fn(*func_args, **kwargs)

        func_args = [_resolve(arg) for arg in node.args]
        kwargs = {kw.arg: _resolve(kw.value) for kw in node.keywords}
        return self.visit(node.func, func_args=func_args, kwargs=kwargs)

    # --- Operators ---------------------------------------------------------

    def visit_BoolOp(self, node):
        values = [self.visit(arg) for arg in node.values]

        for i, value in enumerate(values):
            value_type = None
            if hasattr(value, "type") and isinstance(value.type, IntegerType):
                value_type = value.type
            elif isinstance(value, arith.ConstantOp):
                value_type = value.type

            if value_type is None or not isinstance(value_type, IntegerType):
                raise ValueError("BoolOp values must be ConstantOp or IntegerType")

            if value_type.width != 1:
                values[i] = arith.cmpi(
                    arith.CmpIPredicate.ne, value, arith.ConstantOp(value_type, 0)
                )

        def _match(lhs, rhs):
            match (node.op):
                case ast.And():
                    return arith.andi(lhs, rhs)
                case ast.Or():
                    return arith.ori(lhs, rhs)
                case _:
                    raise NotImplementedError(f"BoolOp {node.op} not supported")

        chained = _match(values[0], values[1])
        for i in range(2, len(values)):
            chained = _match(chained, values[i])
        return chained

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not lhs or not rhs:
            raise ValueError("Binary operands not found")

        if isinstance(lhs, OpView):
            lhs = lhs.result
        if isinstance(rhs, OpView):
            rhs = rhs.result

        mlir_type = _get_type_str(lhs.type)

        def qualified_or(attr, otherwise):
            fn = self._fn_map.get(f"{mlir_type}.{attr}")
            if fn is not None:
                return fn(lhs, rhs)
            cast_rhs = rhs
            if lhs.type != cast_rhs.type:
                cast_rhs = _cast(cast_rhs, lhs.type)
            assert lhs.type == cast_rhs.type, f"{lhs.type} != {cast_rhs.type}"
            return otherwise(lhs, cast_rhs)

        def unimplemented(*args, **kwargs):
            raise NotImplementedError(f"{node.op} not implemented")

        match (node.op):
            case ast.Add():
                return qualified_or("__add__", arith.addi)
            case ast.Sub():
                return qualified_or("__sub__", arith.subi)
            case ast.Mult():
                return qualified_or("__mul__", arith.muli)
            case ast.Div():
                return qualified_or("__truediv__", unimplemented)
            case ast.MatMult():
                return qualified_or("__matmul__", unimplemented)
            case ast.FloorDiv():
                return qualified_or("__floordiv__", arith.divsi)
            case ast.Mod():
                return qualified_or("__mod__", arith.remsi)
            case ast.Pow():
                return qualified_or("__pow__", unimplemented)
            case ast.LShift():
                return qualified_or("__lshift__", arith.shli)
            case ast.RShift():
                return qualified_or("__rshift__", arith.shrsi)
            case ast.BitOr():
                return qualified_or("__or__", arith.ori)
            case ast.BitAnd():
                return qualified_or("__and__", arith.andi)
            case ast.BitXor():
                return qualified_or("__xor__", arith.xori)
            case _:
                raise NotImplementedError(
                    f"Binary operator {type(node.op).__name__} not implemented"
                )

    def visit_UnaryOp(self, node):
        if (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
        ):
            return arith.ConstantOp(IndexType.get(self.ctx), -node.operand.value)

        operand = self.visit(node.operand)
        if not operand:
            raise ValueError("Unary operand not found")

        mlir_type = _get_type_str(operand.type)

        def qualified_or(attr, otherwise, *args, **kwargs):
            fn = self._fn_map.get(f"{mlir_type}.{attr}", otherwise)
            return fn(*args, **kwargs)

        match (node.op):
            case ast.USub():
                return qualified_or(
                    "__neg__", lambda v: emitc.UnaryMinusOp(v.type, v), operand
                )
            case ast.UAdd():
                return qualified_or(
                    "__pos__", lambda v: emitc.UnaryPlusOp(v.type, v), operand
                )
            case ast.Not():
                return qualified_or(
                    "__not__",
                    lambda v: emitc.logical_not(
                        IntegerType.get_signless(1, self.ctx), v
                    ),
                    operand,
                )
            case ast.Invert():
                return qualified_or(
                    "__invert__", lambda v: emitc.bitwise_not(v.type, v), operand
                )
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

    # --- Subscript / attribute --------------------------------------------

    def visit_Subscript(self, node):
        tbl = self._var_exists(node.value.id)
        if not tbl:
            self._fail(
                node,
                NameError(f"unknown variable '{node.value.id}'"),
                hint=self._hint_for_name(node.value.id),
            )
        arr = tbl[node.value.id]

        if not hasattr(arr, "type") or not isinstance(arr.type, RankedTensorType):
            raise ValueError("Can only subscript tensors")

        def _build_index(slice_, shape, bounds_check_idx):
            if hasattr(slice_, "value"):
                if slice_.value >= shape[bounds_check_idx]:
                    raise IndexError("Index out of bounds.")
                return arith.ConstantOp(IndexType.get(self.ctx), slice_.value)
            r = self.visit(slice_)
            if isinstance(r.type, IndexType):
                return r
            return arith.IndexCastOp(IndexType.get(self.ctx), r)

        if isinstance(node.slice, ast.Constant):
            if arr.type.rank < 1:
                raise IndexError("Can only index elements of array, rank < 1")
            if arr.type.shape[0] <= node.slice.value:
                raise IndexError("Index out of bounds.")
            idx = _build_index(node.slice, arr.type.shape, 0)
        elif isinstance(node.slice, ast.Tuple):
            if len(node.slice.elts) > arr.type.rank:
                raise IndexError("Can only index elements of array, rank >= len(index)")
            idx = [
                _build_index(elt, arr.type.shape, i)
                for i, elt in enumerate(node.slice.elts)
            ]
        else:
            idx = [_build_index(node.slice, arr.type.shape, 0)]

        return (arr, idx)

    def visit_Attribute(self, node, func_args=[], kwargs={}):
        # node.value may be any expression (Name, Call, Attribute, ...) --
        # support chained method calls like `a.add(b).sigmoid()`.
        if isinstance(node.value, ast.Name):
            sym_table = self._var_exists(node.value.id)
            if not sym_table:
                self._fail(
                    node,
                    NameError(f"unknown variable '{node.value.id}'"),
                    hint=self._hint_for_name(node.value.id),
                )
            mlir_value = sym_table[node.value.id]
            receiver_repr = node.value.id
        else:
            mlir_value = self.visit(node.value)
            if hasattr(mlir_value, "result"):
                mlir_value = mlir_value.result
            receiver_repr = type(node.value).__name__
        mlir_type = _get_type_str(mlir_value.type)
        fn = self._fn_map.get(f"{mlir_type}.{node.attr}", None)
        if fn is None:
            self._fail(
                node,
                AttributeError(
                    f"`{receiver_repr}` of type {mlir_type} has no method "
                    f"'{node.attr}'"
                ),
                hint=self._hint_for_method(mlir_type, node.attr),
            )
        return fn(mlir_value, *func_args, **kwargs)

    # --- Containers --------------------------------------------------------

    def visit_List(self, node):
        return self.visit_Tuple(node)

    def visit_Tuple(self, node):
        return tuple(map(self.visit, node.elts))

    # --- Literals ----------------------------------------------------------

    def visit_Constant(self, node):
        as_attr = getattr(node, "_ttkernel_as_attr", False)
        op_constructor = IntegerAttr.get if as_attr else arith.ConstantOp
        if callable(as_attr):
            return as_attr(node)
        if isinstance(node.value, bool):
            return op_constructor(IndexType.get(self.ctx), node.value)
        if isinstance(node.value, int):
            return op_constructor(IndexType.get(self.ctx), node.value)
        raise NotImplementedError(
            f"constant type {type(node.value).__name__} not implemented"
        )


def syntax(syntax_name, args_as_attr=None, kwargs_as_attr=None):
    if syntax_name.startswith("!"):

        def _class_wrapper(cls):
            nonlocal args_as_attr, kwargs_as_attr
            assert isinstance(cls, type)
            for name, method in cls.__dict__.items():
                if callable(method):
                    sig = inspect.signature(method)
                    first_arg_name = next(iter(sig.parameters.keys()))
                    if first_arg_name == "ast_self":
                        setattr(cls, name, staticmethod(method))
                        qualified = f"{syntax_name}.{name}"
                        if args_as_attr is None and kwargs_as_attr is None:
                            D2MCompiler._syntax[qualified] = method
                        else:
                            assert args_as_attr is None or isinstance(
                                args_as_attr, list
                            )
                            assert kwargs_as_attr is None or isinstance(
                                kwargs_as_attr, dict
                            )
                            D2MCompiler._syntax[qualified] = (
                                method,
                                args_as_attr,
                                kwargs_as_attr or {},
                            )
            return cls

        return _class_wrapper

    def _fn_wrapper(fn):
        nonlocal args_as_attr, kwargs_as_attr
        assert callable(fn)
        if args_as_attr is None and kwargs_as_attr is None:
            D2MCompiler._syntax[fn.__name__] = fn
        else:
            assert args_as_attr is None or isinstance(args_as_attr, list)
            assert kwargs_as_attr is None or isinstance(kwargs_as_attr, dict)
            D2MCompiler._syntax[fn.__name__] = (
                fn,
                args_as_attr,
                kwargs_as_attr or {},
            )
        return fn

    return _fn_wrapper
