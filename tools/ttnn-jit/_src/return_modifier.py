# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap

"""
There's a known limitation in graph capture that it does not populate the output tensor shape and info for ops without a consumer.
For instance, imagine a function f like this:
def someFunction(input_tensor):
    # ... some functionality
    last_tensor = ttnn.someOp(...)
    return last_tensor
In this case, the output tensor shape and info for ttnn.someOp(...) is not populated.
To this aim, the following class inserts ttnn.identity calls before the return statements.
So we convert the aforementioned function to this:
def someFunction(input_tensor):
    # ... some functionality
    last_tensor = ttnn.someOp(...)
    ttnn.identity(last_tensor)
    return last_tensor

Therefore, we can correctly capture the output tensor shape and info for ttnn.someOp(...).
This also helps in identifying the final output tensor of the function.
Note 1: that this code also supports multiple return values and expressions.
Note 2: This is a temporary workaround for this problem. In order to fix it, we should either:
- Add a ttnn.no-op as a pass through op to ttnn.
- Add support for capturing the output tensor shape and info for ops without a consumer.
  TODO(sgholami): https://github.com/tenstorrent/tt-metal/issues/33914
Note 3: We assert that the original function does not contain any ttnn.identity calls.
Note 4: This approach should work for any passthrough op. Rn, we use ttnn.identity as an example.
"""

# Placeholder op is used to mark the output of a function without performing any computation.
# One example of a placeholder op is "identity" (ttnn.identity), but other passthrough
# operations could also serve this purpose.
PLACEHOLDER_OP_NAME = "ttnn.identity"


def get_placeholder_op_name():
    """Get the name of the placeholder operation (e.g., 'identity')."""
    return PLACEHOLDER_OP_NAME.split(".")[-1]


def get_placeholder_op_dialect():
    """Get the dialect of the placeholder operation (e.g., 'ttnn')."""
    return PLACEHOLDER_OP_NAME.split(".")[0]


class ReturnModifier(ast.NodeTransformer):
    """AST transformer that inserts passthrough calls before return statements."""

    def visit_Return(self, node):
        if node.value is None:
            return node

        # Check if the return value is a tuple (multiple return values)
        if isinstance(node.value, ast.Tuple):
            # Handle multiple return values
            statements = []
            temp_var_names = []

            # For each element in the tuple, create a temp variable and assignment
            for idx, elt in enumerate(node.value.elts):
                temp_var_name = f"__temp_return_value_{idx}"
                temp_var_names.append(temp_var_name)

                # Create assignment: __temp_return_value_i = <expression>
                assign = ast.Assign(
                    targets=[ast.Name(id=temp_var_name, ctx=ast.Store())],
                    value=elt,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
                statements.append(assign)

            # Create passthrough calls for each temp variable
            for temp_var_name in temp_var_names:
                placeholder_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(
                                id=get_placeholder_op_dialect(), ctx=ast.Load()
                            ),
                            attr=get_placeholder_op_name(),
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=temp_var_name, ctx=ast.Load())],
                        keywords=[],
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    ),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
                statements.append(placeholder_call)

            # Create return statement with tuple of temp variables
            new_return = ast.Return(
                value=ast.Tuple(
                    elts=[ast.Name(id=name, ctx=ast.Load()) for name in temp_var_names],
                    ctx=ast.Load(),
                ),
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            statements.append(new_return)

            return statements
        else:
            # Handle single return value
            temp_var = ast.Name(id="__temp_return_value", ctx=ast.Store())

            # Create assignment: __temp_return_value = <original_return_value>
            assign = ast.Assign(
                targets=[temp_var],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )

            # Create placeholder call to mark the return value: placeholder_op(__temp_return_value)
            placeholder_call = ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=get_placeholder_op_dialect(), ctx=ast.Load()),
                        attr=get_placeholder_op_name(),
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id="__temp_return_value", ctx=ast.Load())],
                    keywords=[],
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ),
                lineno=node.lineno,
                col_offset=node.col_offset,
            )

            # Create return statement: return __temp_return_value
            new_return = ast.Return(
                value=ast.Name(id="__temp_return_value", ctx=ast.Load()),
                lineno=node.lineno,
                col_offset=node.col_offset,
            )

            # Return all three statements wrapped in a list
            return [assign, placeholder_call, new_return]

    def visit_FunctionDef(self, node):
        # Visit the body statements and flatten any lists
        new_body = []
        for stmt in node.body:
            result = self.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)

        node.body = new_body
        return node


def create_modified_function(f):
    """
    Create a modified version of function f that inserts passthrough calls before returns.
    """
    # Get the source code of the function
    source = inspect.getsource(f)

    # Assert that the original function does not contain any placeholder op calls
    assert PLACEHOLDER_OP_NAME not in source, (
        "The jit-ed function cannot include " + f"{PLACEHOLDER_OP_NAME} op for now"
    )

    # Parse the source code into an AST
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # Apply the transformation
    modifier = ReturnModifier()
    modified_tree = modifier.visit(tree)

    # Fix missing locations in the AST
    ast.fix_missing_locations(modified_tree)

    # Compile the modified AST back to code
    code = compile(modified_tree, filename="<ast>", mode="exec")

    # Execute the code in a namespace that includes the original function's globals
    namespace = f.__globals__.copy()

    # Add closure variables to the namespace if the function has any
    if f.__closure__:
        closure_vars = {}
        # Get the names of closure variables from the function's code object
        if f.__code__.co_freevars:
            for var_name, cell in zip(f.__code__.co_freevars, f.__closure__):
                closure_vars[var_name] = cell.cell_contents
        namespace.update(closure_vars)

    exec(code, namespace)

    # Get the function from the namespace (it will have the same name as the original)
    g = namespace[f.__name__]

    return g
