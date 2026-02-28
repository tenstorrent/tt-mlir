// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD


include "EmitPyAttrs.td"
include "EmitPyBase.td"
include "EmitPyInterfaces.td"
include "EmitPyTypes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"


//===----------------------------------------------------------------------===//
// EmitPy ops definitions.
//===----------------------------------------------------------------------===//


def EmitPy_CallOpaqueOp : EmitPy_Op<"call_opaque", [PyExpressionInterface]> {
  let summary = "Opaque call operation";
  let description = [{
    The `emitpy.call_opaque` operation represents a Python function call. The callee
    can be an arbitrary non-empty string.

    Example:

    ```mlir
    %2 = emitpy.call_opaque "ttnn.add"(%0, %1) {args = [0 : index, 1 : index, #emitpy.opaque<"ttnn.DataType.BFLOAT16">, #emitpy.opaque<"ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)">], keyword_args = ["", "", "dtype", "memory_config"]} : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
    ```
  }];

  let arguments = (ins
    Arg<StrAttr, "the Python function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and further attributes">:$args,
    Variadic<AnyType>:$operands,
    Arg<OptionalAttr<StrArrayAttr>>:$keyword_args
  );

  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$result_types,
      "::llvm::StringRef":$callee,
      "::mlir::ValueRange":$operands,
      CArg<"::mlir::ArrayAttr", "{}">:$args,
      CArg<"::mlir::ArrayAttr", "{}">:$keyword_args), [{
        build($_builder, $_state, result_types, callee, args, operands, keyword_args);
      }]
    >
  ];

  let hasVerifier = 1;
  let assemblyFormat = [{$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)}];
}

def EmitPy_ImportOp : EmitPy_Op<"import"> {
  let summary = "Import operation";
  let description = [{
     The `emitpy.import` operation allows to define a Python module import
     via various forms of the `import` statement.

    Example:

    ```mlir
     emitpy.import import "ttnn"
     ```
  }];

  let arguments = (ins
    Arg<StrAttr, "module to import">:$module_name,
    Arg<OptionalAttr<StrAttr>, "module alias">:$module_alias,
    Arg<OptionalAttr<StrArrayAttr>, "specific members to import from the module">:$members_to_import,
    Arg<OptionalAttr<StrArrayAttr>, "aliases for the module or the single imported members">:$member_aliases,
    Arg<OptionalAttr<UnitAttr>, "import all members with *">:$import_all
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def EmitPy_LiteralOp : EmitPy_Op<"literal", [Pure, PyExpressionInterface]> {
  let summary = "Literal operation";
  let description = [{
    The `emitpy.literal` operation produces an SSA value equal to some constant
    specified by an attribute.

    Example:

    ```mlir
    %0 = emitpy.literal "0" : index
    ```
  }];

  let arguments = (ins StrAttr:$value);

  let results = (outs Index:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def EmitPy_SubscriptOp : EmitPy_Op<"subscript", [Pure, PyExpressionInterface]> {
  let summary = "Subscript operation";
  let description = [{
    With the `emitpy.subscript` operation the subscript operator `[]` can be applied
    to variables or arguments of opaque type.

    Example:

    ```mlir
    %0 = emitpy.literal "0" : index
    %1 = emitpy.subscript %arg0[%0] : (!emitpy.opaque<"[ttnn.Tensor]">, index) -> !emitpy.opaque<"ttnn.Tensor">
    ```
  }];

  let arguments = (ins
    Arg<EmitPy_OpaqueType, "the value to subscript">:$value,
    Index:$index);

  let results = (outs EmitPy_OpaqueType:$result);

  // TODO (amilovanovic) : add custom verify method
  let assemblyFormat = "$value `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

def EmitPy_AssignOp : EmitPy_Op<"assign"> {
  let summary = "Assign operation";
  let description = [{
    The `emitpy.assign` operation represents a Python variable assignment.
    This models `new_var = old_var` or `var = constant`.

    Example:

    ```mlir
    %2 = emitpy.assign %1 : <!emitpy.opaque<"ttnn.Tensor">>
    ```
    ```Python
    // Code emitted for the operation above.
    v2 = v1;
    ```
  }];

  let arguments = (ins
    EmitPy_OpaqueType:$value);

  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` functional-type(operands, results)";
}

def EmitPy_GetAttrOp : EmitPy_Op<"get_attr", [Pure, PyExpressionInterface]> {
  let summary = "Get attribute from an object";
  let description = [{
    The `emitpy.get_attr` operation represents Python attribute access
    via the dot operator.

    Example:

    ```mlir
    %w = emitpy.get_attr %self, "weight" : (!emitpy.class<"Model">) -> !emitpy.opaque<"ttnn.Tensor">
    ```
    ```python
    # Code emitted:
    w = self.weight
    ```
  }];

  let arguments = (ins
    AnyType:$object,
    Arg<StrAttr, "attribute name">:$attr_name
  );

  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$object `,` $attr_name attr-dict `:` functional-type(operands, results)";
}

def EmitPy_SetAttrOp : EmitPy_Op<"set_attr"> {
  let summary = "Set attribute on an object";
  let description = [{
    The `emitpy.set_attr` operation represents Python attribute assignment
    via the dot operator.

    Example:

    ```mlir
    emitpy.set_attr %self, "weight", %w : (!emitpy.opaque<"Model">, !emitpy.opaque<"ttnn.Tensor">)
    ```
    ```python
    # Code emitted:
    self.weight = w
    ```
  }];

  let arguments = (ins
    AnyType:$object,
    Arg<StrAttr, "attribute name">:$attr_name,
    AnyType:$value
  );

  let hasVerifier = 1;
  let assemblyFormat = "$object `,` $attr_name `,` $value attr-dict `:` `(` type($object) `,` type($value) `)`";
}

def EmitPy_ClassOp : EmitPy_Op<"class", [Symbol, SymbolTable, IsolatedFromAbove,
                                        SingleBlock, NoTerminator]> {
  let summary = "Class definition";
  let description = [{
    The `emitpy.class` operation represents a Python class definition.
    It is a symbol and contains a single-region body with class members.

    Class verification enforces:
    - At most one `__init__` method.
    - `__init__` returns nothing and is an instance method.
    - Methods can be annotated with `emitpy.method_kind` attribute on
      `func.func` using one of: `instance`, `staticmethod`, `classmethod`.
    - Instance and class methods must have a receiver argument as the first
      argument; if named, it must be `self` (instance) or `cls` (classmethod).

    Example:

    ```mlir
    emitpy.class @Model(#emitpy.opaque<"torch.nn.Module">) {
      func.func @__init__(%self: !emitpy.class<"Model">) {
        %w = emitpy.call_opaque "ttnn.load_weight"() : () -> !emitpy.opaque<"ttnn.Tensor">
        emitpy.set_attr %self, "weight", %w : (!emitpy.class<"Model">, !emitpy.opaque<"ttnn.Tensor">)
        return
      }
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<ArrayAttr>:$base_classes
  );

  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def EmitPy_CreateDictOp : EmitPy_Op<"create_dict", [Pure]> {
  let summary = "Create dictionary operation";
  let description = [{
    Creates a Python dictionary with a specified name.

    Can be created in two ways:
    1. From key-value pairs (items operands) - pairs are alternating key, value
    2. From a Python literal expression (literal_expr attribute)

    When literal_expr is provided, items must be empty.
    When items are provided, they must be even count (key-value pairs).

    Example:

    ```mlir
    // Empty dict using literal_expr
    %dict1 = emitpy.create_dict "my_dict" {literal_expr = "{}"} : () -> !emitpy.dict

    // Dict from key-value operands
    %dict2 = emitpy.create_dict "cache" (%k0, %v0) : (index, !emitpy.opaque<"None">) -> !emitpy.dict

    // Dict from literal expression (most compact for complex dicts)
    %dict3 = emitpy.create_dict "_CONST_EVAL_CACHE" {literal_expr = "{i: None for i in range(133)}"} : () -> !emitpy.dict
    ```

    ```python
    // Code emitted for the operations above:
    my_dict = {}
    cache = {0: None}
    _CONST_EVAL_CACHE = {i: None for i in range(133)}
    ```
  }];

  let arguments = (ins
    Arg<StrAttr>:$dict_name,
    Variadic<AnyType>:$items,
    OptionalAttr<StrAttr>:$literal_expr
  );

  let results = (outs EmitPy_DictType:$result);

  let hasVerifier = 1;
  let assemblyFormat = [{
    $dict_name (`(` $items^ `)`)? attr-dict `:` functional-type(operands, results)
  }];
}

def EmitPy_SetValueForDictKeyOp : EmitPy_Op<"set_value_for_dict_key"> {
  let summary = "Set dictionary value for a given key";
  let description = [{
    Changes or inserts a value in the dictionary for a given key.
    The key must be either an index type or an opaque type representing a string.

    Example:
    ```mlir
    %dict = emitpy.global_statement @_CONST_EVAL_CACHE : !emitpy.dict
    %key = emitpy.literal "5" : index
    emitpy.set_value_for_dict_key %dict[%key] = %tensors : (!emitpy.dict, index, !emitpy.opaque<"[ttnn.Tensor]">)
    ```
    ```python
    // Code emitted:
    _CONST_EVAL_CACHE[5] = tensors
    ```
  }];

  let arguments = (ins
    EmitPy_DictType:$dict,
    AnyTypeOf<[Index, EmitPy_OpaqueType]>:$key,
    EmitPy_Type:$value
  );

  let hasVerifier = 1;

  let assemblyFormat = "$dict `[` $key `]` `=` $value attr-dict `:` `(` type($dict) `,` type($key) `,` type($value) `)`";
}

def EmitPy_GetValueForDictKeyOp : EmitPy_Op<"get_value_for_dict_key", [Pure]> {
  let summary = "Get dictionary value for a given key";
  let description = [{
    Retrieves a value for a given key in a dictionary.
    The key must be either an index type or an opaque type representing a string.

    Example:
    ```mlir
    %dict = emitpy.global_statement @_CONST_EVAL_CACHE : !emitpy.dict
    %key = emitpy.literal "5" : index
    %tensors = emitpy.get_value_for_dict_key %dict[%key] : (!emitpy.dict, index) -> !emitpy.opaque<"[ttnn.Tensor]">
    ```
    ```python
    // Code emitted:
    tensors = _CONST_EVAL_CACHE[5]
    ```
  }];

  let arguments = (ins
    EmitPy_DictType:$dict,
    AnyTypeOf<[Index, EmitPy_OpaqueType]>:$key
  );

  let results = (outs EmitPy_Type:$result);

  let hasVerifier = 1;

  let assemblyFormat = "$dict `[` $key `]` attr-dict `:` functional-type(operands, results)";
}

def EmitPy_VerbatimOp : EmitPy_Op<"verbatim"> {
  let summary = "Verbatim operation";
  let description = [{
    The `emitpy.verbatim` operation produces no results and the value is emitted as is
    followed by a line break ('\n' character) during translation.

    This operation can be used in situations where a more suitable operation is
    not yet implemented in the dialect.

    Note: Use with caution. This operation can have arbitrary effects on the
    semantics of the emitted code. Use semantically more meaningful operations
    whenever possible. Additionally this op is *NOT* intended to be used to
    inject large snippets of code.
  }];

  let extraClassDeclaration = [{
    FailureOr<SmallVector<::mlir::tt::emitpy::ReplacementItem>> parseFormatString();
  }];

  let arguments = (ins StrAttr:$value, Variadic<AnyType>:$fmtArgs);

  let builders = [OpBuilder<(ins "::mlir::StringAttr":$value), [{ build($_builder, $_state, value, {}); }]>,
                  OpBuilder<(ins "::llvm::StringRef":$value), [{ build($_builder, $_state, value, {}); }]>];

  let hasVerifier = 1;
  let assemblyFormat = "$value (`args` $fmtArgs^ `:` type($fmtArgs))? attr-dict";
}

def EmitPy_ConstantOp : EmitPy_Op<"constant", [ConstantLike]> {
  let summary = "Constant operation";
  let description = [{
    The `emitpy.constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants.
  }];

  let arguments = (ins EmitPy_OpaqueOrTypedAttr:$value);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;
}


def EmitPy_GlobalOp : EmitPy_Op<"global", [Symbol]> {
  let summary = "A global variable";
  let description = [{
    The `emitpy.global` operation defines a named global variable.

    Example:

    ```mlir
    emitpy.global @x : #emitpy.opaque<"[]">
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       EmitPy_OpaqueOrTypedAttr:$initial_value);

  let hasVerifier = 1;
  let assemblyFormat = [{
       $sym_name custom<EmitPyGlobalOpInitialValue>($initial_value) attr-dict
  }];
}

def EmitPy_GlobalStatementOp : EmitPy_Op<"global_statement",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Marks a variable as global for the current function scope";
  let description = [{
    The `emitpy.global_statement` operation represents a Python `global`
    statement.

    It marks an identifier, specified by its symbol `@name`, as
    referring to a variable in the global (module) scope, rather
    than a new local variable.

    Example:

    ```mlir
    %0 = emitpy.global_statement @x : !emitpy.opaque<"[ttnn.Tensor]">
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

def EmitPy_AssignGlobalOp : EmitPy_Op<"assign_global",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Assign a value to a global variable";
  let description = [{
    The `emitpy.assign_global` assigns the value to an existing named global variable.

    Example:

    ```mlir
    emitpy.assign_global @value = %arg0 : !emitpy.opaque<"[ttnn.Tensor]">
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name,
                       AnyType:$value);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def EmitPy_YieldOp : EmitPy_Op<"yield", [Pure, ReturnLike, Terminator,
                                          ParentOneOf<["ExpressionOp"]>]> {
  let summary = "Yield operation for expression termination";
  let description = [{
    The `emitpy.yield` terminates its parent EmitPy operation's region, optionally yielding
    an SSA value. The semantics of how the values are yielded is defined by the
    parent operation.

    Example:

    ```mlir
    %0 = emitpy.expression(%arg0, %arg1) : (!emitpy.opaque, !emitpy.opaque) -> !emitpy.opaque {
    ^bb0(%a: !emitpy.opaque, %b: !emitpy.opaque):
      %1 = emitpy.call_opaque "foo"(%a, %b) : (!emitpy.opaque, !emitpy.opaque) -> !emitpy.opaque
      emitpy.yield %1 : !emitpy.opaque
    }
    ```
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

def EmitPy_ExpressionOp : EmitPy_Op<"expression", [
    HasOnlyGraphRegion, IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Expression operation to inline Python code";
  let description = [{
    The `emitpy.expression` operation returns a single SSA value yielded by its
    single-basic-block region. The operations inside the region must form a DAG
    (directed acyclic graph) of computations that produces a single result value.
    The expression's operands are mapped to block arguments and can be used
    within the expression body.

    When generating Python code, operations within the expression body are emitted
    inline without intermediate variable assignments, producing cleaner code.

    The optional `do_not_inline` attribute can be used to force the translator to
    emit the expression as separate statements rather than inline.

    Example:

    ```mlir
    %expression = emitpy.expression(%a, %b) : (!emitpy.opaque, !emitpy.opaque) -> !emitpy.opaque {
    ^bb0(%arg0: !emitpy.opaque, %arg1: !emitpy.opaque):
      %0 = emitpy.call_opaque "ttnn.mul"(%arg0, %arg1) : (!emitpy.opaque, !emitpy.opaque) -> !emitpy.opaque
      %1 = emitpy.call_opaque "ttnn.add"(%0, %arg1) : (!emitpy.opaque, !emitpy.opaque) -> !emitpy.opaque
      emitpy.yield %1 : !emitpy.opaque
    }
    return %expression : !emitpy.opaque
    ```

    This would generate:
    ```python
    return ttnn.add(ttnn.mul(a, b), b)
    ```

    If the `do_not_inline` attribute is set:
    ```python
    %expression = ttnn.add(ttnn.mul(a, b), b)
    return %expression
    ```

  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       UnitAttr:$do_not_inline);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// Get the body region of the expression.
    Region &getBodyRegion() { return getBody(); }

    /// Get the block of the body region.
    Block *getBodyBlock() { return &getBody().front(); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
