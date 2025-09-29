# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from ttmlir.ir import *
from ttmlir.dialects import func, emitc


class PyKernelAstBase(ast.NodeVisitor):
    def __init__(self, *args, **kwargs):
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.func_entry = None
        self.symbol_tables = []
        self.supported_nodes = [ast.Module, ast.Return, ast.Expr]
        self._fn_map = {}

    def _get_source_comment(self, node):
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

    def _get_source_comment_block(self, node, delim: str = "):"):
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

    def _var_exists(self, var_name):
        for sym_table in reversed(self.symbol_tables):
            if var_name in sym_table:
                return sym_table
        return {}

    def visit_Module(self, node):
        # Set default basic block
        with InsertionPoint(self.insert_point), Location.unknown():
            for stmt in node.body:
                self.visit(stmt)

    def visit_Return(self, node):
        # TODO: handle more than one return, i.e. tuples, expressions etc.
        if node.value:
            # Visit the return value and return it
            return_value = self.visit(node.value)
            func.ReturnOp([return_value])
        else:
            # Empty return
            func.ReturnOp([])

    def visit_Expr(self, node):
        # NOTE: will catch function calls and expressions where return values not used.
        return self.visit(node.value)

    def visit(self, node: ast.AST, **kwargs):
        if any(
            isinstance(node, supported_node) for supported_node in self.supported_nodes
        ):
            if self.verbose and isinstance(
                node, (ast.Assign, ast.AnnAssign, ast.AugAssign)
            ):
                # Create a verbatim Op here to store the comment
                source_code = self._get_source_comment(node)
                emitc.verbatim(source_code, [])

            # Figure out which node to visit. Not using super().visit() in order to pass kwargs.
            method_name = "visit_" + node.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)

            params = inspect.signature(visitor).parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
            if filtered_kwargs:
                return visitor(node, **filtered_kwargs)
            else:
                return visitor(node)
        else:
            raise NotImplementedError(f"visit {type(node).__name__} not supported")
