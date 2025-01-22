# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import tt, ttkernel, func, scf, arith, memref
import ast
import inspect
import functools

def get_supported_nodes():
    return [
        # Variables
        ast.Name,
        ast.Load, 
        ast.Store,

        # control-flow
        ast.If,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,

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

        # Statements
        ast.Pass,
        ast.Assign,
        ast.AugAssign,

        # Function-and-class-definitions
        ast.Module,
        ast.FunctionDef,
        ast.Return,
        ast.arguments,
        ast.arg,
    ]

class TTKernelCompiler(ast.NodeVisitor):
    def __init__(self, name):
        self.name = name
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.symbol_tables = []
        self.supported_nodes = get_supported_nodes()

        ttkernel.register_dialect(self.ctx)
    
    def var_exists(self, var_name):
        for sym_table in reversed(self.symbol_tables):
            if var_name in sym_table:
                return sym_table
        return {}

    # Root Nodes
    def visit_Module(self, node):
        # set default basic block
        with InsertionPoint(self.insert_point), Location.unknown():
            for stmt in node.body:
                self.visit(stmt)

    def visit_FunctionDef(self, node):
        # print(f"visit_FunctionDef")
        # TODO: add alloca args name into symbol table
        if(self.func_entry): raise IndexError("Cannot declare function within a function")

        self.symbol_tables.append({}) 
        arg_types = []
        for arg in node.args.args:
            if (arg.annotation.id == "int"):
                arg_types.append(IntegerType.get_signless(32, self.ctx))
            else: 
                raise NotImplementedError(f"function arg type {arg.annotation.id} not implemented")
        
        self.func_entry = func.FuncOp(name=node.name, type=(arg_types, []))
        func_bb = self.func_entry.add_entry_block()

        # update basic block
        with InsertionPoint(func_bb), Location.unknown():
            for target in node.body:
                self.visit(target)

            # TODO: Check for a return/terminator insert one if not present

    # Function/Class definitions
    def visit_Return(self, node):
        # TODO: handle more than one return, i.e. tuples, expressions etc.
        # TODO: need a symbol table in order to return the right thing
        if (node.value):
            self.visit(node.value)
        func.ReturnOp([])

    # Control Flow
    def visit_If(self, node):
        # NOTE: else-if blocks are not supported in SCF dialect
        # TODO: if cond can be: Name, Expr, Compare, Call, UnaryOp, BoolOp
        if_cond = self.visit(node.test)
        if not isinstance (if_cond, Value): # not sure if Value is the right type here.. need IntegerType i1
            if_cond = arith.TruncIOp(IntegerType.get_signless(1), if_cond) # temporary since expr not implemented yet
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
    
    def visit_For():
        pass

    def visit_While():
        pass

    def visit_Break():
        pass

    def visit_Continue():
        pass

    # Statements
    def visit_Name(self, node):
        # print(f"visit_Name")
        var_name = node.id
        existing_var_table = self.var_exists(var_name)
        if existing_var_table:
            return existing_var_table[var_name]
        
        return None


    def visit_Assign(self, node):
        # print(f"visit_Assign")
        assert len(node.targets) == 1, "Only single assignments supported"
        var = self.visit(node.targets[0])
        value = self.visit(node.value)

        if not var:
            sym_table = self.symbol_tables[-1]
            var_name = node.targets[0].id
            var_type = value.type
            memref_type = MemRefType.get([1], var_type)
            var = memref.alloca(memref_type, [], [])
            sym_table[var_name] = var

        # TODO: need to handle arrays and other types
        memref.StoreOp(value, var, [arith.ConstantOp(IndexType.get(self.ctx), 0)])

    def visit_AugAssign(self, node):
        pass

    # Expressions
    def visit_Expr(self, node):
        # print(f"visit_Expr")
        return self.visit(node.value)

    # Literals
    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return arith.ConstantOp(IntegerType.get_signless(32, self.ctx), node.value)
        else:
            raise NotImplementedError(f"constant type {type(node.value).__name__} not implemented")


    def visit(self, node : ast.AST):
        if any(isinstance(node, supported_node) for supported_node in self.supported_nodes):
            return super().visit(node)
        else:
            raise NotImplementedError(f"visit {type(node).__name__} not supported")

def ttkernel_compile(f):
    @functools.wraps(f)
    def _wrapper(*args, **kwargs):
        m = ast.parse(inspect.getsource(f))
        b = TTKernelCompiler(f.__name__)
        # print(ast.dump(m, indent=4) + "\n")
        b.visit(m)
        print(b.module)

    return _wrapper

