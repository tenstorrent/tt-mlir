# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from ttmlir.ir import *
from ttmlir.dialects import (
    ttcore,
    d2m,
    func,
    scf,
    arith,
    emitc,
)
from ttmlir.dialects._ods_common import get_default_loc_context

from .utils import _discover_dialect_ops, _cast, _get_type_str


class TensorLayout:
    def __init__(
        self,
        tensor,
        block_shape,
        grid_shape=None,
        dtype=None,
        tiled=True,
        collapse=True,
        mem_space=ttcore.MemorySpace.DeviceL1,
    ):
        dtype = TensorLayout._to_data_type(
            str(tensor.dtype if dtype is None else dtype)
        )
        self.tensor = tensor
        self.logical_shape = tensor.shape
        self.block_shape = block_shape
        self.blocked_grid_shape = TensorLayout._derive_blocked_grid_shape(
            list(tensor.shape), block_shape, tiled
        )
        self.grid_shape = self.logical_grid_shape if grid_shape is None else grid_shape
        self.dtype = dtype
        self.tiled = tiled
        self.collapse = collapse
        self.mem_space = TensorLayout._to_mem_space(mem_space)
        self._cached_layout = None

    @staticmethod
    def _to_data_type(dtype: str):
        if dtype in {"torch.float32", "fp32"}:
            return ttcore.DataType.Float32
        elif dtype in {"torch.float16", "fp16"}:
            return ttcore.DataType.Float16
        elif dtype in {"torch.bfloat16", "bf16"}:
            return ttcore.DataType.BFloat16
        else:
            raise TypeError(f"Unsupported dtype {dtype}")

    @staticmethod
    def _to_mem_space(mem_space):
        if isinstance(mem_space, ttcore.MemorySpace):
            return mem_space
        if mem_space in {"l1", "sram"}:
            return ttcore.MemorySpace.DeviceL1
        elif mem_space == "dram":
            return ttcore.MemorySpace.DeviceDRAM
        else:
            raise TypeError(f"Unsupported mem_space {mem_space}")

    @staticmethod
    def _derive_blocked_grid_shape(logical_shape, block_shape, tiled):
        assert len(logical_shape) == len(block_shape)
        if tiled:
            for i in range(len(logical_shape)):
                logical_shape[i] = (logical_shape[i] + 31) // 32

        blocked_grid_shape = []
        for ls, bs in zip(logical_shape, block_shape):
            assert ls % bs == 0
            blocked_grid_shape.append(ls // bs)

        return blocked_grid_shape

    def get_tile_shape(self):
        return [32, 32] if self.tiled else []

    def get_scalar_type(self, ctx):
        if self.dtype == ttcore.DataType.Float32:
            return F32Type.get(ctx)
        elif self.dtype == ttcore.DataType.Float16:
            return F16Type.get(ctx)
        elif self.dtype == ttcore.DataType.BFloat16:
            return BF16Type.get(ctx)
        else:
            raise TypeError(f"Unsupported data type {self.dtype}")

    def get_host_elem_type(self, ctx):
        return self.get_scalar_type(ctx)

    def get_device_elem_type(self, ctx):
        elem_type = self.get_scalar_type(ctx)
        if self.tiled:
            tile_shape = self.get_tile_shape()
            elem_type = ttcore.ir.TileType.get(
                ctx, tile_shape[0], tile_shape[1], self.dtype
            )
        return elem_type

    def get_device_shape(self, ctx, grid_shape):
        layout = self.build_metal_layout(ctx)
        metal_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
        return metal_layout.getDeviceShape(grid_shape, self.get_tile_shape())

    def build_host_tensor_type(self, ctx):
        return RankedTensorType.get(self.logical_shape, self.get_host_elem_type(ctx))

    def build_metal_layout(self, ctx):
        if self._cached_layout is not None:
            return self._cached_layout

        if self.collapse:
            self._cached_layout = ttcore.ir.MetalLayoutAttr.get(
                ctx,
                list(self.logical_shape),
                int(ttcore.OOBVal.Undef),
                int(self.mem_space),
                int(ttcore.TensorMemoryLayout.Sharded),
            )
        else:
            empty_interval_type = RankedTensorType.get(
                [0, 2], IntegerType.get_signless(64)
            )
            empty_collapse_intervals = DenseIntElementsAttr.get(empty_interval_type, [])
            self._cached_layout = ttcore.ir.MetalLayoutAttr.get(
                ctx,
                list(self.logical_shape),
                int(ttcore.OOBVal.Undef),
                int(self.mem_space),
                int(ttcore.TensorMemoryLayout.Sharded),
                empty_collapse_intervals,
                [],
            )

        return self._cached_layout

    def build_device_tensor_type(self, ctx, blocked=False):
        grid_shape = self.blocked_grid_shape if blocked else self.grid_shape
        layout = self.build_metal_layout(ctx)
        elem_type = self.get_device_elem_type(ctx)
        device_shape = self.get_device_shape(ctx, grid_shape)
        return RankedTensorType.get(device_shape, elem_type, encoding=layout)

    def build_to_device(self, ctx, val):
        output_type = self.build_device_tensor_type(ctx)
        output = d2m.empty(output_type)
        res = d2m.ToLayoutOp(
            [output_type],
            val,
            output,
        ).result
        return self.build_blocked_view(ctx, res)

    def build_blocked_view(self, ctx, val):
        if self.blocked_grid_shape == self.grid_shape:
            # Nothing to do
            return val
        device_shape = self.get_device_shape(ctx, self.grid_shape)
        blocked_device_shape = self.get_device_shape(ctx, self.blocked_grid_shape)
        blocked_type = self.build_device_tensor_type(ctx, blocked=True)
        reblock_map = d2m.ir.calculate_reblock_map(
            device_shape, blocked_device_shape, ctx
        )
        return d2m.ViewLayoutOp(blocked_type, val, reblock_map).result

    def build_device_view(self, ctx, val):
        if self.blocked_grid_shape == self.grid_shape:
            # Nothing to do
            return val
        device_shape = self.get_device_shape(ctx, self.grid_shape)
        blocked_device_shape = self.get_device_shape(ctx, self.blocked_grid_shape)
        device_type = self.build_device_tensor_type(ctx, blocked=False)
        reblock_map = d2m.ir.calculate_reblock_map(
            blocked_device_shape, device_shape, ctx
        )
        return d2m.ViewLayoutOp(device_type, val, reblock_map).result

    def build_from_device(self, ctx, val):
        output_type = self.build_host_tensor_type(ctx)
        output = d2m.empty(output_type)
        return d2m.ToLayoutOp(
            [output_type],
            val,
            output,
        ).result


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

    def __init__(self, name, kernel_type=None, captures=None, *args, **kwargs):
        assert kernel_type in _D2M_KERNEL_TYPES, f"Invalid kernel type {kernel_type}"

        self.name = name
        self.kernel_type = kernel_type
        self.captures = captures if captures is not None else {}
        self.args = args

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
        self.streams = set()
        self.supported_nodes = list(self._SUPPORTED_NODES)
        self._fn_map = dict(self._syntax)

    # --- Symbol table helpers ----------------------------------------------

    def _var_exists(self, var_name):
        for sym_table in reversed(self.symbol_tables):
            if var_name in sym_table:
                return sym_table
        return {}

    # --- Dispatch ----------------------------------------------------------

    def visit(self, node, **kwargs):
        if not any(isinstance(node, n) for n in self.supported_nodes):
            raise NotImplementedError(f"visit {type(node).__name__} not supported")
        method_name = "visit_" + node.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        params = inspect.signature(visitor).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        if filtered_kwargs:
            return visitor(node, **filtered_kwargs)
        return visitor(node)

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
            if isinstance(rt_arg, TensorLayout):
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
            v = self.visit(arg)
            if v is None:
                raise ValueError(f"Function argument not found for {node.func.id}")
            return v

        if not isinstance(node.func, ast.Attribute):
            assert (
                node.func.id in self._fn_map
            ), f"Function {node.func.id} not supported"
            fn = self._fn_map[node.func.id]
            args_as_attr = [False] * len(node.args)
            if isinstance(fn, tuple):
                fn, args_as_attr = fn
            assert len(node.args) == len(args_as_attr)
            func_args = []
            for arg, as_attr in zip(node.args, args_as_attr):
                arg._ttkernel_as_attr = as_attr
                func_args.append(_resolve(arg))
            kwargs = {kw.arg: _resolve(kw.value) for kw in node.keywords}
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
                raise ValueError(
                    "BoolOp values must be ConstantOp or IntegerType"
                )

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

        if lhs.type != rhs.type:
            rhs = _cast(rhs, lhs.type)
        assert lhs.type == rhs.type, f"{lhs.type} != {rhs.type}"
        mlir_type = _get_type_str(lhs.type)

        def qualified_or(attr, otherwise, *args, **kwargs):
            fn = self._fn_map.get(f"{mlir_type}.{attr}", otherwise)
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
                return qualified_or("__floordiv__", arith.divsi, lhs, rhs)
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
            raise ValueError("Array doesn't exist.")
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
                raise IndexError(
                    "Can only index elements of array, rank >= len(index)"
                )
            idx = [
                _build_index(elt, arr.type.shape, i)
                for i, elt in enumerate(node.slice.elts)
            ]
        else:
            idx = [_build_index(node.slice, arr.type.shape, 0)]

        return (arr, idx)

    def visit_Attribute(self, node, func_args=[], kwargs={}):
        mlir_value = self._var_exists(node.value.id)[node.value.id]
        mlir_type = _get_type_str(mlir_value.type)
        fn = self._fn_map.get(f"{mlir_type}.{node.attr}", None)
        if fn is None:
            raise ValueError(
                f"{node.value.id} of type {mlir_type} has no attribute {node.attr}"
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


def syntax(syntax_name, args_as_attr=None):
    if syntax_name.startswith("!"):

        def _class_wrapper(cls):
            nonlocal args_as_attr
            assert isinstance(cls, type)
            for name, method in cls.__dict__.items():
                if callable(method):
                    sig = inspect.signature(method)
                    first_arg_name = next(iter(sig.parameters.keys()))
                    if first_arg_name == "ast_self":
                        setattr(cls, name, staticmethod(method))
                        qualified = f"{syntax_name}.{name}"
                        if args_as_attr is None:
                            D2MCompiler._syntax[qualified] = method
                        else:
                            assert isinstance(args_as_attr, list)
                            D2MCompiler._syntax[qualified] = (method, args_as_attr)
            return cls

        return _class_wrapper

    def _fn_wrapper(fn):
        nonlocal args_as_attr
        assert callable(fn)
        if args_as_attr is None:
            D2MCompiler._syntax[fn.__name__] = fn
        else:
            assert isinstance(args_as_attr, list)
            D2MCompiler._syntax[fn.__name__] = (fn, args_as_attr)
        return fn

    return _fn_wrapper


class Stream:
    def __init__(self, tensor, num_buffers=None):
        assert hasattr(
            tensor, "_global_name"
        ), "Stream must be created from a top level tensor argument"
        self.name = tensor._global_name
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        assert num_buffers is None, "Unsupported"
        self.num_buffers = num_buffers
