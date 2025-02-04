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
        ast.AnnAssign,
        # Function-and-class-definitions
        ast.Module,
        ast.FunctionDef,
        ast.Return,
        ast.arguments,
        ast.arg,
    ]


class TTKernelCompiler(ast.NodeVisitor):
    ttkernel_fn_map = {
        "unary_op_init_common": ttkernel.unary_op_init_common,
        "get_arg_val": ttkernel.get_arg_val,
        "cb_wait_front": ttkernel.cb_wait_front,
        "cb_reserve_back": ttkernel.cb_reserve_back,
        "cb_push_back": ttkernel.cb_push_back,
        "cb_pop_front": ttkernel.cb_pop_front,
        "tile_regs_acquire": ttkernel.tile_regs_acquire,
        "tile_regs_release": ttkernel.tile_regs_release,
        "pack": ttkernel.pack,
        "pack_tile": ttkernel.pack_tile,
        "copy_tile": ttkernel.copy_tile,
        "unpack_a": ttkernel.unpack_a,
        "unpack_ab": ttkernel.unpack_ab,
        "add": ttkernel.add,
        "get_compile_time_arg_val": ttkernel.get_compile_time_arg_val,
        "get_write_ptr": ttkernel.get_write_ptr,
        "get_read_ptr": ttkernel.get_read_ptr,
        "get_tile_size": ttkernel.get_tile_size,
        "get_noc_addr_from_bank_id": ttkernel.get_noc_addr_from_bank_id,
        "noc_async_read": ttkernel.noc_async_read,
        "noc_async_write": ttkernel.noc_async_write,
        "noc_async_read_barrier": ttkernel.noc_async_read_barrier,
        "noc_async_write_barrier": ttkernel.noc_async_write_barrier,
    }

    def __init__(self, name, args):
        self.name = name
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.initial_cbs = args
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
        # Set default basic block
        with InsertionPoint(self.insert_point), Location.unknown():
            for stmt in node.body:
                self.visit(stmt)

    def visit_FunctionDef(self, node):
        # TODO: add alloca args name into symbol table
        if self.func_entry:
            raise IndexError("Cannot declare function within a function")

        arg_types = []
        for i in range(len(node.args.args)):
            arg = node.args.args[i]
            if not arg.annotation:
                raise ValueError("Function arguments must have type annotations")
            if arg.annotation.id == "int":
                tile_type = tt.ir.TileType.get(self.ctx, 32, 32, tt.DataType.UInt32)
            elif arg.annotation.id == "float":
                tile_type = tt.ir.TileType.get(self.ctx, 32, 32, tt.DataType.Float32)
            else:
                raise ValueError(f"cannot pass {arg.annotation.id} to a pykernel")

            # TODO: investigate if this is the only type we can pass into functions
            cb_type = ttkernel.ir.CBType.get(
                self.ctx,  # mlir context
                0,  # address
                self.initial_cbs[i],
                MemRefType.get(
                    [8, 4, 4], tile_type
                ),  # hardcoded dimensions for now - this is usually lowered from tensors?
            )
            arg_types.append(cb_type)

        func_sym_table = {}
        self.func_entry = func.FuncOp(name=node.name, type=(arg_types, []))
        func_bb = self.func_entry.add_entry_block()
        for i in range(len(func_bb.arguments)):
            func_sym_table[node.args.args[i].arg] = func_bb.arguments[i]

        # update basic block
        self.symbol_tables.append(func_sym_table)
        with InsertionPoint(func_bb), Location.unknown():
            for target in node.body:
                self.visit(target)

            # TODO: Check for a return/terminator insert one if not present

        self.symbol_tables.pop()

    # Function/Class definitions
    def visit_Return(self, node):
        # TODO: handle more than one return, i.e. tuples, expressions etc.
        # TODO: need a symbol table in order to return the right thing
        if node.value:
            self.visit(node.value)
        func.ReturnOp([])

    # Control Flow
    def visit_If(self, node):
        # NOTE: else-if blocks are not supported in SCF dialect
        # NOTE: Only handling Compare for if statements right now
        # TODO: if cond can be: Name, Expr, Compare, Call, UnaryOp, BoolOp
        assert isinstance(
            node.test, ast.Compare
        ), "Only Compare supported in if statements"
        if_cond = self.visit(node.test)
        # if not if_cond.result.type.width == 1 or not if_cond.result.type.is_signless:
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
        assert (
            len(node.iter.args) == 3
        ), "Must specify range(start, end, step) in for loops"
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

        for_op = scf.ForOp(lower_bound, upper_bound, step)
        with (InsertionPoint(for_op.body)), Location.unknown():
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
        raise NotImplementedError("While loops not supported yet")

    def visit_Break():
        raise NotImplementedError("Break not supported yet")

    def visit_Continue():
        raise NotImplementedError("Continue not supported yet")

    # Statements
    def visit_Name(self, node):
        var_name = node.id

        # NOTE: some kernelops require passing return type as arg
        if var_name == "int":
            return IntegerType.get_signless(32, self.ctx)

        existing_var_table = self.var_exists(var_name)
        if existing_var_table:
            return existing_var_table[var_name]

        return None

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "Only single assignments supported"
        var = self.visit(node.targets[0])
        value = self.visit(node.value)
        sym_table = self.symbol_tables[-1]
        var_name = node.targets[0].id
        # print(var.type)
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

        if not var:
            var_type = value.type
            memref_type = MemRefType.get([1], var_type)
            var = memref.alloca(memref_type, [], [])
            sym_table[var_name] = var
        else:
            assert isinstance(var, MemRefType), "Can not AugAssign to non-memref types"

        memref.StoreOp(value, var, [arith.ConstantOp(IndexType.get(self.ctx), 0)])

    def visit_AugAssign(self, node):
        raise NotImplementedError("AugAssign not supported yet")

    # Function calls
    def visit_Call(self, node):
        # print(f"visit_Call")
        assert (
            node.func.id in self.ttkernel_fn_map
        ), f"Function {node.func.id} not supported"
        func = self.ttkernel_fn_map[node.func.id]
        func_args = []
        for arg in node.args:
            func_arg = self.visit(arg)
            if not func_arg:
                raise ValueError("Function argument not found")

            if hasattr(func_arg, "type") and isinstance(
                func_arg.type, memref.MemRefType
            ):
                func_arg = memref.LoadOp(
                    func_arg, arith.ConstantOp(IndexType.get(self.ctx), 0)
                )

            func_args.append(func_arg)

        return func(*func_args)  # how do i make sure the types are correct?

    # Expressions
    def visit_Expr(self, node):
        # NOTE: will catch function calls and expressions where return values not used.
        return self.visit(node.value)

    def visit_BinOp(self, node):
        # TODO: need to load MemRef types when using variables
        # TODO: need to handle float, unsigned, and signed operations
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if not lhs or not rhs:
            raise ValueError("Binary operands not found")

        # load variable if needed
        if isinstance(lhs.type, memref.MemRefType):
            lhs = memref.LoadOp(lhs, arith.ConstantOp(IndexType.get(self.ctx), 0))
        if isinstance(rhs.type, memref.MemRefType):
            rhs = memref.LoadOp(rhs, arith.ConstantOp(IndexType.get(self.ctx), 0))

        match (node.op):
            case ast.Add():
                return arith.addi(lhs, rhs)
            case ast.Sub():
                return arith.subi(lhs, rhs)
            case ast.Mult():
                return arith.muli(lhs, rhs)
            # case ast.Div(): # only worried about integers right now
            # return arith.divf(lhs, rhs)
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
            )

        match (node.op):
            # need to expose emitc for these unary operators, not sure if this is necessary yet
            # case ast.USub():
            #     # emitc has a unary minus operator
            #     return arith.subi(arith.ConstantOp(IntegerType.get_signless(32, self.ctx), 0), operand)
            # case ast.Not():
            #     return arith.xori(operand, arith.ConstantOp(IntegerType.get_signless(32, self.ctx), 1))
            # case ast.Invert():
            #     return arith.xori(operand, arith.ConstantOp(IntegerType.get_signless(32, self.ctx), -1))
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

    # Literals
    def visit_Constant(self, node):
        if isinstance(node.value, int):
            return arith.ConstantOp(IntegerType.get_signless(32, self.ctx), node.value)
        else:
            raise NotImplementedError(
                f"constant type {type(node.value).__name__} not implemented"
            )

    def visit(self, node: ast.AST):
        if any(
            isinstance(node, supported_node) for supported_node in self.supported_nodes
        ):
            return super().visit(node)
        else:
            raise NotImplementedError(f"visit {type(node).__name__} not supported")


def ttkernel_compile(f):
    @functools.wraps(f)
    def _wrapper(*args, **kwargs):
        m = ast.parse(inspect.getsource(f))
        b = TTKernelCompiler(f.__name__, args)
        # print(ast.dump(m, indent=4) + "\n")
        b.visit(m)
        print(b.module)

        # Check if generated IR is valid
        b.module.operation.verify()

    return _wrapper
