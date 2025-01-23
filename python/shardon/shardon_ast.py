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
        if not if_cond.result.type.width == 1 or not if_cond.result.type.is_signless:
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
    
    def visit_For(self, node):
        # print(help(node))
        assert node.iter.func.id == "range", "Only range() supported in for loops"
        assert len(node.iter.args) == 3, "Must specify range(start, end, step) in for loops"
        lower_bound = self.visit(node.iter.args[0])
        upper_bound = self.visit(node.iter.args[1])
        step = self.visit(node.iter.args[2])
        for_op = scf.ForOp(lower_bound, upper_bound, step)
        with(InsertionPoint(for_op.body)), Location.unknown():
            self.symbol_tables.append({})
            for stmt in node.body:
                self.visit(stmt)
            scf.YieldOp([])
            self.symbol_tables.pop()

    def visit_While(self, node):
        # TODO: while cond like if stmt, need support for at least: Name, Expr, Compare, Call, UnaryOp, BoolOp
        # TODO: support initial values based off variables used in the while loop
        # NOTE: while loops are hard since scf.WhileOp doesn't support basic blocks?
        # init_values = [arith.ConstantOp(IntegerType.get_signless(1), 0)]
        # result_types = [IntegerType.get_signless(1)]

        # while_op = scf.WhileOp(result_types, init_values)
        # while_cond_bb = Block.create_at_start(while_op.before)

        # with InsertionPoint(while_cond_bb), Location.unknown():
        #     while_cond = self.visit(node.test)
        #     while_cond = arith.TruncIOp(IntegerType.get_signless(1), while_cond)
        #     scf.ConditionOp(while_cond, while_cond_bb.arguments)
        
        # while_body_bb = Block.create_at_start(while_op.after)
        # with InsertionPoint(while_body_bb), Location.unknown():
        #     self.symbol_tables.append({})
        #     for stmt in node.body:
        #         self.visit(stmt)
        #     scf.YieldOp(while_body_bb.arguments)
        #     self.symbol_tables.pop()
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
    def visit_Call(self, node):
        # print(f"visit_Call")
        return None
    
    def visit_Expr(self, node):
        # NOTE: will catch function calls and expressions where return values not used.
        return self.visit(node.value)

    def visit_BinOp(self, node):
        # TODO: need to load MemRef types when using variables
        # TODO: need to handle float, unsigned, and signed operations
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        # print(help(lhs.type))
        # print(f"types: {lhs.type}, {rhs.type}")
        if not lhs or not rhs:
            raise ValueError("Binary operands not found")
        
        # load variable if needed
        if isinstance(lhs.type, memref.MemRefType):
            lhs = memref.LoadOp(lhs, arith.ConstantOp(IndexType.get(self.ctx), 0))
        if isinstance(rhs.type, memref.MemRefType):
            rhs = memref.LoadOp(rhs, arith.ConstantOp(IndexType.get(self.ctx), 0))
        
        match(node.op):
            case ast.Add():
                return arith.addi(lhs, rhs)
            case ast.Sub():
                return arith.subi(lhs, rhs)
            case ast.Mult():
                return arith.muli(lhs, rhs)
            # case ast.Div(): # only worried about integers right now
                # return arith.divf(lhs, rhs)
            case _:
                raise NotImplementedError(f"Binary operator {type(node.op).__name__} not implemented")
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if not operand:
            raise ValueError("Unary operand not found")

        if isinstance(operand.type, memref.MemRefType):
            operand = memref.LoadOp(operand, arith.ConstantOp(IndexType.get(self.ctx), 0))
        
        match(node.op):
            # need to expose emitc for these unary operators, not sure if this is necessary yet
            # case ast.USub():
            #     # emitc has a unary minus operator
            #     return arith.subi(arith.ConstantOp(IntegerType.get_signless(32, self.ctx), 0), operand)
            # case ast.Not():
            #     return arith.xori(operand, arith.ConstantOp(IntegerType.get_signless(32, self.ctx), 1))
            # case ast.Invert():
            #     return arith.xori(operand, arith.ConstantOp(IntegerType.get_signless(32, self.ctx), -1))
            case _:
                raise NotImplementedError(f"Unary operator {type(node.op).__name__} not implemented")

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
        
        match(node.ops[0]):
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
                raise NotImplementedError(f"Compare operator {type(node.ops).__name__} not implemented")

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

