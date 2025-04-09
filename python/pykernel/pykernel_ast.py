# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import functools
import os
from ttmlir.ir import *
from ttmlir.dialects import tt, ttkernel, func, scf, arith, memref, emitc
from ttmlir.passes import ttkernel_to_cpp, pykernel_compile_pipeline

# ttmlir-translate --ttkernel-to-cpp-noc


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
        "binary_op_init_common": ttkernel.binary_op_init_common,
        "add_tiles_init": ttkernel.add_tiles_init,
        "get_arg_val": ttkernel.get_arg_val,
        "cb_wait_front": ttkernel.cb_wait_front,
        "cb_reserve_back": ttkernel.cb_reserve_back,
        "cb_push_back": ttkernel.cb_push_back,
        "cb_pop_front": ttkernel.cb_pop_front,
        "tile_regs_acquire": ttkernel.tile_regs_acquire,
        "tile_regs_release": ttkernel.tile_regs_release,
        "tile_regs_commit": ttkernel.tile_regs_commit,
        "tile_regs_wait": ttkernel.tile_regs_wait,
        "pack_tile": ttkernel.pack_tile,
        "copy_tile": ttkernel.copy_tile,
        "add_tiles": ttkernel.add_tiles,
        "get_compile_time_arg_val": ttkernel.get_compile_time_arg_val,
        "get_write_ptr": ttkernel.get_write_ptr,
        "get_read_ptr": ttkernel.get_read_ptr,
        "get_tile_size": ttkernel.get_tile_size,
        "get_dataformat": ttkernel.get_dataformat,
        "get_noc_addr_from_bank_id": ttkernel.get_noc_addr_from_bank_id,
        "noc_async_read": ttkernel.noc_async_read,
        "noc_async_read_tile": ttkernel.noc_async_read_tile,
        "noc_async_write": ttkernel.noc_async_write,
        "noc_async_write_tile": ttkernel.noc_async_write_tile,
        "noc_async_read_barrier": ttkernel.noc_async_read_barrier,
        "noc_async_write_barrier": ttkernel.noc_async_write_barrier,
        "get_interleaved_addr_gen_fast": ttkernel.get_interleaved_addr_gen_fast,
        "exp_tile_init": ttkernel.exp_tile_init,
        "exp_tile": ttkernel.exp_tile,
    }

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.symbol_tables = []
        self.supported_nodes = get_supported_nodes()

        self.cb_args = args
        self.rt_args = None

        self.verbose = kwargs.get("verbose", False)
        self.source_code = kwargs.get("source_code", "")

    def get_source_comment(self, node):
        """
        Retrieve the source snippet corresponding to the given node and format it as comments.

        This function extracts the relevant lines of source code using the node's location
        attributes (lineno, end_lineno, col_offset, end_col_offset), prefixes each line with
        '//', and returns the concatenated snippet as a single string.

        Args:
            node: An AST node that contains information about the source code segment location.

        Returns:
            str: The snippet of source code formatted with '//' at the beginning of each line.
        """
        result = ""
        if self.verbose and self.source_code:
            for i in range(node.lineno - 1, node.end_lineno):
                result += (
                    "// "
                    + self.source_code[i][node.col_offset : node.end_col_offset]
                    + "\n"
                )
        return result.strip()

    def get_source_comment_block(self, node, delim: str = "):"):
        """
        Generates a comment block extracted from the source code related to the given AST node.

        This function examines lines of source code starting at node.lineno and continuing up to
        node.end_lineno, looking for the specified delimiter. Each line is prefixed with "// " to form
        a comment block. If the delimiter is found, it stops appending further lines.

        Args:
            node: An AST node that provides line number boundaries (lineno, end_lineno) for source extraction.
            delim (str): The string delimiter to indicate where to stop collecting lines. Defaults to "):".

        Returns:
            str: A multi-line comment string containing the relevant source code lines, each prefixed with "// ".
        """
        result = ""
        if self.verbose and self.source_code:
            idx = node.lineno - 1
            result = "// "
            while idx < node.end_lineno:
                line = self.source_code[idx]
                end_pattern = line.find(delim)
                if end_pattern != -1:
                    # First occurence of end_pattern detected, save the current splice of the string + exist
                    result += line[: end_pattern + 2].lstrip()
                    break
                idx += 1
                result += f"{line}\n// "
        return result

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
        assert not self.func_entry, "Cannot declare function within a function"

        arg_types = []
        for i in range(len(node.args.args)):
            arg = node.args.args[i]

            # Check for rt_args
            # TODO: Decide between strict rt_args name _or_ type of list[int] defining argument as rt_args.
            if arg.arg == "rt_args":
                # This is a valid defined rt_args object
                # We don't want this to be defined in the EmitC module since it's bootstrapped to call get_arg_val
                # Instead set a flag for ast.Subscript to check if this value is being called.
                self.rt_args = arg
                continue

            if not arg.annotation:
                raise ValueError("Function arguments must have type annotations")
            elif not arg.annotation.id == "CircularBuffer":
                raise TypeError(f"cannot pass {arg.annotation.id} to a pykernel")

            tile_type = tt.ir.TileType.get(
                self.ctx, 32, 32, getattr(tt.DataType, self.cb_args[i].dtype)
            )
            cb_type = ttkernel.ir.CBType.get(
                self.ctx,  # mlir context
                0,  # address
                self.cb_args[i].cb_id,
                MemRefType.get(
                    self.cb_args[i].tilized_shape, tile_type
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
            # Insert verbose comment for function, to be picked up by Compiler pass it must exist within function region
            # Need a bit of custom logic to make the function def look pretty:
            # Get the source code from the main function decl:
            if self.verbose and self.source_code:
                comment = f"// --- Python Function Declaration for Above --- \n{self.get_source_comment_block(node)}\n// -- End Function Declaration"
                emitc.verbatim(comment)

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

        if self.verbose:
            comment = self.get_source_comment_block(node)
            emitc.verbatim(comment)

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
        assert (
            node.func.id in self.ttkernel_fn_map
        ), f"Function {node.func.id} not supported"
        func = self.ttkernel_fn_map[node.func.id]
        func_args = []
        for arg in node.args:
            func_arg = self.visit(arg)
            if not func_arg:
                raise ValueError(f"Function argument not found for {node.func.id}")

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
        if isinstance(lhs.type, memref.MemRefType):
            lhs = memref.LoadOp(
                lhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
        if isinstance(rhs.type, memref.MemRefType):
            rhs = memref.LoadOp(
                rhs, arith.ConstantOp(IndexType.get(self.ctx), 0)
            ).result
        match (node.op):
            case ast.Add():
                return arith.addi(lhs, rhs)
            case ast.Sub():
                return arith.subi(lhs, rhs)
            case ast.Mult():
                return arith.muli(lhs, rhs)
            case ast.FloorDiv():
                return arith.floordivsi(lhs, rhs)
            case ast.Mod():
                return arith.remsi(lhs, rhs)
            case ast.LShift():
                return arith.shli(lhs, rhs)
            case ast.RShift():
                return arith.shrsi(lhs, rhs)
            case ast.BitOr():
                return arith.ori(lhs, rhs)
            case ast.BitAnd():
                return arith.andi(lhs, rhs)
            case ast.BitXor():
                return arith.xori(lhs, rhs)
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
        # This is where we can invoke the rt_args for the kernel to be loaded in
        if self.rt_args is not None:
            # rt_args has been defined, we know that the id of the array to reference from is "rt_args"
            if node.value.id == self.rt_args.arg:
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
                        raise IndexError(
                            "Runtime Arg Slices must have Upper Bound defined"
                        )
                    result = []
                    for i in range(_lower, node.slice.upper.value, _step):
                        arg_index = self.visit(ast.Constant(i))
                        int_type = IntegerType.get_signless(32, self.ctx)
                        result.append(ttkernel.get_arg_val(int_type, arg_index))
                    return result
        raise NotImplementedError(
            "Loading from Subscripts except Runtime Args not supported"
        )

    # Literals
    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return arith.ConstantOp(IntegerType.get_signless(1, self.ctx), node.value)
        elif isinstance(node.value, int):
            return arith.ConstantOp(IntegerType.get_signless(32, self.ctx), node.value)
        else:
            raise NotImplementedError(
                f"constant type {type(node.value).__name__} not implemented"
            )

    def visit(self, node: ast.AST):
        if any(
            isinstance(node, supported_node) for supported_node in self.supported_nodes
        ):
            if self.verbose and isinstance(
                node, (ast.Assign, ast.AnnAssign, ast.AugAssign)
            ):
                # Create a verbatim Op here to store the comment
                source_code = self.get_source_comment(node)
                emitc.verbatim(source_code)
            return super().visit(node)
        else:
            raise NotImplementedError(f"visit {type(node).__name__} not supported")


def ttkernel_compile(kernel_type=None, verbose: bool = False, optimize: bool = True):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            if verbose is True:
                # Create easily index-able object to store source code:
                source_code = inspect.getsource(f).split("\n")
                kwargs["source_code"] = source_code
                kwargs["verbose"] = True
            m = ast.parse(inspect.getsource(f))
            b = TTKernelCompiler(f.__name__, *args, **kwargs)
            # print(ast.dump(m, indent=4) + "\n")
            b.visit(m)

            # Check if generated IR is valid
            # print(b.module)
            b.module.operation.verify()

            # Run the PyKernel Compile Pipeline to fit model for Translation
            if optimize:
                pykernel_compile_pipeline(b.module)
                # print("---- Optimized PyKernel Module ----", b.module, sep="\n\n")

            if kernel_type:
                assert kernel_type in ["noc", "tensix"], "Invalid kernel type"
                is_tensix_kernel = kernel_type == "tensix"
                kernel_string = ttkernel_to_cpp(b.module, is_tensix_kernel)
                return kernel_string

        return _wrapper

    return _decorator


def ttkernel_tensix_compile(verbose: bool = False, optimize: bool = True):
    return ttkernel_compile(kernel_type="tensix", verbose=verbose, optimize=optimize)


def ttkernel_noc_compile(verbose: bool = False, optimize: bool = True):
    return ttkernel_compile(kernel_type="noc", verbose=verbose, optimize=optimize)
