// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD


include "EmitPyAttrs.td"
include "EmitPyBase.td"
include "EmitPyInterfaces.td"
include "EmitPyTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"

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
    %2 = emitpy.call_opaque "ttnn.add"(%0, %1) {args = [0 : index, 1 : index]} : (!emitpy.opaque<"ttnn.Tensor">, !emitpy.opaque<"ttnn.Tensor">) -> !emitpy.opaque<"ttnn.Tensor">
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

def EmitPy_SubscriptOp : EmitPy_Op<"subscript", [PyExpressionInterface]> {
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

def EmitPy_ExpressionOp : EmitPy_Op<"expression",
      [HasOnlyGraphRegion, OpAsmOpInterface,
       SingleBlockImplicitTerminator<"emitpy::YieldOp">, NoRegionArguments]> {
  let summary = "Expression operation";
  let description = [{
    The `emitpy.expression` operation returns a single SSA value which is yielded by
    its single-basic-block region. The operation doesn't take any arguments.

    As the operation is to be emitted as a Python expression, the operations within
    its body must form a single Def-Use tree of emitpy ops whose result is
    yielded by a terminating `emitpy.yield`.

    The operations allowed within expression body are EmitPy operations with the
    PyExpressionInterface interface.

    When specified, the optional `do_not_inline` indicates that the expression is
    to be emitted as seen above, i.e. as the rhs of an EmitPy SSA value
    definition. Otherwise, the expression may be emitted inline, i.e. directly
    at its use.
  }];

  let arguments = (ins UnitAttr:$do_not_inline);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$region);

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict (`noinline` $do_not_inline^)? `:` type($result) $region";

  let extraClassDeclaration = [{
    bool hasSideEffects() {
      auto predicate = [](Operation &op) {
        assert(isa<emitpy::PyExpressionInterface>(op) && "Expected a Py expression");
        return cast<emitpy::PyExpressionInterface>(op).hasSideEffects();
      };
      return llvm::any_of(getRegion().front().without_terminator(), predicate);
    };
    Operation *getRootOp();

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// EmitPy ops in the body can omit their 'emitpy.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "emitpy";
    }
  }];
}

def EmitPy_YieldOp : EmitPy_Op<"yield",
      [Pure, Terminator, ParentOneOf<["ExpressionOp"]>]> {
  let summary = "Block termination operation";
  let description = [{
    The `emitpy.yield` terminates its parent EmitPy op's region, optionally yielding
    an SSA value. The semantics of how the values are yielded is defined by the
    parent operation.

    If `emitpy.yield` has an operand, the operand must match the parent operation's
    result. If the parent operation defines no values, then the `emitpy.yield`
    may be left out in the custom syntax and the builders will insert one
    implicitly. Otherwise, it has to be present in the syntax to indicate which
    value is yielded.
  }];

  let arguments = (ins Optional<AnyType>:$result);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let hasVerifier = 1;
  let assemblyFormat = [{ attr-dict ($result^ `:` type($result))? }];
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

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
