# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from ttmlir.ir import *
from ttmlir.dialects import (
    ttcore,
    d2m,
    func,
    arith,
)
from ttmlir.dialects._ods_common import get_default_loc_context

from .kernel_types import *
from .utils import _discover_dialect_ops, _cast
from .kernel_ast import TTCompilerBase


class D2MGenericCompiler(TTCompilerBase):
    _syntax = {}

    def __init__(self, name, kernel_type=None, captures={}, *args, **kwargs):
        super().__init__(name, kernel_type, *args, **kwargs)
        self.loc = Location.name(self.name)
        self.captures = captures
        self.streams = set()
        self.supported_nodes.append(ast.AsyncFunctionDef)
        self._fn_map = {}
        self._fn_map["iter_index"] = (
            d2m.iter_index,
            [True],
        )  # True for arg as attribute
        self._fn_map["core_index"] = (
            d2m.core_index,
            [True],
        )  # True for arg as attribute
        for name, val in D2MGenericCompiler._syntax.items():
            self._fn_map[name] = val

    def _emit_entry(self, node):
        # TODO: add alloca args name into symbol table
        assert not self.func_entry, "Cannot declare function within a function"

        func_operand_types = []
        for i in range(len(node.args.args)):
            arg = node.args.args[i]

            if not arg.annotation:
                raise TypeError("All kernel arguments must have a type annotation")
            elif arg.annotation.id == "TensorBlock":
                shape = self.args[i].shape
                dtype = F32Type.get(self.ctx)
                func_operand_types.append(RankedTensorType.get(shape, dtype))
            elif arg.annotation.id == "CircularBuffer":
                shape = self.args[i].shape
                dtype = F32Type.get(self.ctx)
                tensor = RankedTensorType.get(shape, dtype)
                func_operand_types.append(d2m.ir.CBType.get(self.ctx, tensor))
            elif arg.annotation.id == "Semaphore":
                func_operand_types.append(d2m.ir.SemaphoreType.get(self.ctx))
            else:
                raise TypeError(
                    f"Unknown kernel arguments type annotation {arg.annotation.id}"
                )

        self.func_entry = func.FuncOp(name=node.name, type=(func_operand_types, []))

        self.func_entry.attributes[d2m.ir.ThreadAttr.name] = d2m.ir.ThreadAttr.get(
            self.ctx, self.kernel_type
        )

        self.symbol_tables.append({})

        # prepopulate bb arguments into symbol table
        func_bb = self.func_entry.add_entry_block()
        for i, bb_arg in enumerate(func_bb.arguments):
            self.symbol_tables[-1][node.args.args[i].arg] = bb_arg

        self.module_symbol_table = SymbolTable(self.module.operation)

        # update basic block
        with InsertionPoint(func_bb):
            # prepopulate captures at the top of the scope
            for name, val in self.captures.items():
                assert isinstance(name, str)
                if isinstance(val, int):
                    self.symbol_tables[-1][name] = arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    )
                elif isinstance(val, Stream):
                    with InsertionPoint.at_block_begin(self.module.body):
                        dtype = F32Type.get(self.ctx)
                        tensor = RankedTensorType.get(shape, dtype)
                        globalTensor = ttcore.GlobalOp(val.name, tensor)
                        self.module_symbol_table.insert(globalTensor.operation)
                    self.symbol_tables[-1][name] = ttcore.get_global(tensor, val.name)
                    self.streams.add(val.name)
                else:
                    raise TypeError(f"Invalid capture type for var {name}: {type(val)}")

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


def syntax(syntax_name):
    if syntax_name.startswith("!"):

        def _class_wrapper(cls):
            assert isinstance(cls, type)

            for name, method in cls.__dict__.items():
                if callable(method): 
                    sig = inspect.signature(method)
                    first_arg_name = next(iter(sig.parameters.keys()))
                    if first_arg_name == "ast_self":
                        setattr(cls, name, staticmethod(method))
                        qualified = f"{syntax_name}.{name}"
                        D2MGenericCompiler._syntax[qualified] = method

            return cls

        return _class_wrapper
    else:

        def _fn_wrapper(fn):
            assert callable(fn)
            D2MGenericCompiler._syntax[fn.__name__] = fn
            return fn

        return _fn_wrapper


# TODO MOVE TO d2m_api.py
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
