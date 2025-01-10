# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

from ttmlir.ir import *
from ttmlir.dialects import tt, ttkernel, func, scf, arith
import ast
import inspect
import functools


class TTKernelBuilder(ast.NodeVisitor):
    builtin_fns = {"Tensix"}
    cb_fn_map = {
        "wait": ttkernel.cb_wait_front,
        "reserve": ttkernel.cb_reserve_back,
        "push": ttkernel.cb_push_back,
        "pop": ttkernel.cb_pop_front,
    }
    t6_fn_map = {
        "tile_regs_acquire": ttkernel.tile_regs_acquire,
        "tile_regs_release": ttkernel.tile_regs_release,
        "pack": ttkernel.pack,
        "unpack_a": ttkernel.unpack_a,
        "unpack_ab": ttkernel.unpack_ab,
        "add": ttkernel.add,
    }

    def __init__(self, name, arg_shapes, arg_dtypes):
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.name = name
        assert len(arg_shapes) == len(arg_dtypes)
        self.arg_shapes = arg_shapes
        self.arg_dtypes = arg_dtypes
        self.arg_tile_shapes = [(32, 32) for i in range(len(arg_shapes))]
        self.symbol_table = {}
        self.int_constant_map = {}
        ttkernel.register_dialect(self.ctx)

    def get_constant(self, value):
        if isinstance(value, int):
            if value in self.int_constant_map:
                return self.int_constant_map[value]
            u32_ty = IntegerType.get_signless(32, self.ctx)
            result = arith.constant(u32_ty, value)
            if value == 0 or value == 1:
                self.int_constant_map[value] = result
            return result
        else:
            raise NotImplementedError(f"get_constant {value} not implemented")

    def get_tilized_memref_for_arg(self, idx):
        arg_shape = list(self.arg_shapes[idx])
        arg_dtype = self.arg_dtypes[idx]
        arg_tile_shape = list(self.arg_tile_shapes[idx])
        assert len(arg_shape) >= 2
        arg_shape[-2] = (arg_shape[-2] + arg_tile_shape[-2] - 1) // arg_tile_shape[-2]
        arg_shape[-1] = (arg_shape[-1] + arg_tile_shape[-1] - 1) // arg_tile_shape[-1]
        element_type = tt.ir.TileType.get(
            self.ctx, arg_tile_shape[-2], arg_tile_shape[-1], arg_dtype
        )
        return MemRefType.get(arg_shape, element_type)

    def emit_entry_func(self, node):
        assert isinstance(node, ast.FunctionDef)
        num_defaults = len(node.args.defaults)
        num_args = len(node.args.args) - num_defaults
        operandTypes = [
            ttkernel.ir.CBType.get(
                self.ctx,
                0,
                idx,
                self.get_tilized_memref_for_arg(idx),
            )
            for idx in range(num_args)
        ]
        resultTypes = []
        entry = func.FuncOp(self.name, (operandTypes, resultTypes))
        entry_block = Block.create_at_start(entry.body, operandTypes)
        for idx in range(num_args):
            self.symbol_table[node.args.args[idx].arg] = entry_block.arguments[idx]
        return entry_block

    def extract_range(self, node):
        assert isinstance(node, ast.Call)
        assert node.func.id == "range"
        assert len(node.args) <= 2, "Step not supported for now"
        if len(node.args) == 1:
            lower_bound = self.get_constant(0)
        else:
            lower_bound = self.emit_expr(node.args[0])
        upper_bound = self.emit_expr(node.args[-1])
        return lower_bound, upper_bound, self.get_constant(1)

    def emit_for(self, node):
        assert isinstance(node, ast.For)
        lower_bound, upper_bound, step = self.extract_range(node.iter)
        scf_for = scf.ForOp(lower_bound, upper_bound, step)
        return Block.create_at_start(scf_for.regions[0]) # this is hardcoded and probably wrong, will always create bb0 and does not scf.yield

    def emit_expr(self, node):
        if isinstance(node, ast.Attribute):
            return self.emit_attribute(node)
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
                return self.emit_cb_shape(node.value, node.slice)
            return self.emit_attribute(node)
        elif isinstance(node, ast.Call):
            return self.emit_call(node)
        elif isinstance(node, ast.Name):
            return self.emit_name(node)
        elif isinstance(node, ast.Constant):
            return self.emit_constant(node)
        else:
            raise NotImplementedError(f"emit_expr {ast.dump(node)} not implemented")

    def emit_cb_shape(self, value, slice):
        assert isinstance(value, ast.Attribute)
        assert value.attr == "shape"
        sym_ty = ttkernel.ir.CBType.cast(self.symbol_table[value.value.id].type)
        shape = sym_ty.shape
        if isinstance(slice, ast.UnaryOp):
            assert isinstance(slice.op, ast.USub)
            assert isinstance(slice.operand, ast.Constant)
            idx = -slice.operand.value
        elif isinstance(slice, ast.Constant):
            idx = slice.value
        elif isinstance(slice, ast.Index):
            return self.emit_cb_shape(value, slice.value)
        else:
            raise NotImplementedError(f"emit_cb_shape {slice} not implemented")
        idx += len(shape)
        if idx < 0 or idx >= len(shape):
            return self.get_constant(1)
        return self.get_constant(shape[idx])

    def emit_call(self, value):
        assert isinstance(value, ast.Call)
        if isinstance(value.func, ast.Name) and value.func.id in self.builtin_fns:
            if value.func.id == "Tensix":
                # TODO emit set config data types
                return "Tensix"
        elif isinstance(value.func, ast.Attribute):
            if value.func.attr in TTKernelBuilder.cb_fn_map:
                func = TTKernelBuilder.cb_fn_map[value.func.attr]
                cb = self.symbol_table[value.func.value.id]
                assert ttkernel.ir.CBType.cast(cb.type)
                return func(cb, self.get_constant(1))
            elif self.symbol_table[value.func.value.id] == "Tensix":
                assert (
                    value.func.attr in TTKernelBuilder.t6_fn_map
                ), f"Unknown method on builtin type Tensix {value.func.attr}"
                func = TTKernelBuilder.t6_fn_map[value.func.attr]
                args = [self.emit_expr(arg) for arg in value.args]
                return func(*args)
            else:
                raise NotImplementedError(
                    f"emit_call ast.Attribute {ast.dump(value)} not implemented"
                )
        else:
            raise NotImplementedError(f"emit_call {ast.dump(value)} not implemented")

    def emit_name(self, node):
        assert isinstance(node, ast.Name)
        return self.symbol_table[node.id]

    def emit_constant(self, node):
        assert isinstance(node, ast.Constant)
        return self.get_constant(node.value)

    def emit_assign(self, node):
        assert isinstance(node, ast.Assign)
        assert len(node.targets) == 1, "Only support single item assignment for now"
        target = node.targets[0]
        value = self.emit_expr(node.value)
        self.symbol_table[target.id] = value

    def visit(self, node):
        if isinstance(node, ast.Module):
            with InsertionPoint(self.insert_point), Location.unknown():
                self.generic_visit(node)
        elif isinstance(node, ast.FunctionDef):
            entry_block = self.emit_entry_func(node)
            with InsertionPoint(entry_block), Location.unknown():
                # Always emit constant 0 and 1 for now
                self.get_constant(0)
                self.get_constant(1)
                for stmt in node.body:
                    self.visit(stmt)
        elif isinstance(node, ast.For):
            for_block = self.emit_for(node)
            with InsertionPoint(for_block), Location.unknown():
                for stmt in node.body:
                    self.visit(stmt)
        elif isinstance(node, ast.Expr):
            self.emit_expr(node.value)
        elif isinstance(node, ast.Assign):
            self.emit_assign(node)
        else:
            self.module.dump()
            raise NotImplementedError(ast.dump(node))


class Tensix:
    def __init__(self, srcA_dtype, srcB_dtype, dst_dtype):
        pass

    def tile_regs_acquire(self):
        pass

    def tile_regs_release(self):
        pass




def to_data_type(dtype):
    if dtype == "float32":
        return tt.DataType.Float32
    else:
        raise NotImplementedError(f"to_data_type {dtype} not implemented")


def ttkernel_compile(f):
    @functools.wraps(f)
    def _wrapper(*args, **kwargs):
        arg_shapes = [tuple(arg.shape) for arg in args]
        arg_dtypes = [to_data_type(arg.dtype) for arg in args]
        m = ast.parse(inspect.getsource(f))
        b = TTKernelBuilder(f.__name__, arg_shapes, arg_dtypes)
        # print(ast.dump(m, indent=4))
        b.visit(m)
        # CHECK: "func.func"[[C:.*]]
        # CHECK: %[[C:.*]] = "arith.constant"[[C:.*]]
        # CHECK: "scf.for"[[C:.*]]
        # CHECK: "ttkernel.cb_wait_front"[[C:.*]]
        # CHECK: "ttkernel.cb_reserve_back"[[C:.*]]
        # CHECK: "ttkernel.tile_regs_acquire"[[C:.*]]
        # CHECK: "ttkernel.unpack_ab"[[C:.*]]
        # CHECK: "ttkernel.add"[[C:.*]]
        # CHECK: "ttkernel.pack"[[C:.*]]
        # CHECK: "ttkernel.tile_regs_release"[[C:.*]]
        # CHECK: "ttkernel.cb_pop_front"[[C:.*]]
        # CHECK: "ttkernel.cb_push_back"[[C:.*]]
        print(b.module)
        # return f(*args, **kwargs)

    return _wrapper

