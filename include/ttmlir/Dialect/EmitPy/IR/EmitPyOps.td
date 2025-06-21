// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "EmitPyBase.td"
include "EmitPyTypes.td"
include "EmitPyAttrs.td"

//===----------------------------------------------------------------------===//
// EmitPy ops definitions
//===----------------------------------------------------------------------===//


def EmitPy_CallOpaqueOp : EmitPy_Op<"call_opaque"> {
  let summary = "Opaque call operation";
  let description = [{
    The `emitpy.call_opaque` operation represents a Python function call. The callee
    can be an arbitrary non-empty string.

    Example:

    ```mlir
    // Custom form defining a call to `foo()`.
    %0 = emitpy.call_opaque "foo" () : () -> i32
    ```
  }];

  let arguments = (ins
    Arg<StrAttr, "the Python function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and further attributes">:$args,
    Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$result_types,
      "::llvm::StringRef":$callee,
      "::mlir::ValueRange":$operands,
      CArg<"::mlir::ArrayAttr", "{}">:$args), [{
        build($_builder, $_state, result_types,  callee, args, operands);
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

  let hasCustomAssemblyFormat = 1;
}

def EmitPy_LiteralOp : EmitPy_Op<"literal", [Pure]> {
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

def EmitPy_SubscriptOp : EmitPy_Op<"subscript"> {
  let summary = "Subscript operation";
  let description = [{
    With the `emitpy.subscript` operation the subscript operator `[]` can be applied
    to variables or arguments of opaque type.

    Example:

    ```mlir
    %0 = emitpy.literal "0" : index
    %1 = emitpy.subscript %arg0[%0] : (!emitpy.opaque<"vector<ttnn.Tensor>">, index) -> !emitpy.lvalue<!emitpy.opaque<"ttnn.Tensor">>
    ```
  }];

  let arguments = (ins
    Arg<EmitPy_OpaqueType, "the value to subscript">:$value,
    Variadic<Index>:$indices);

  let results = (outs EmitPy_LValueType:$result);

  //TODO (amilovanovic) : add custom verify method
  //let hasVerifier = 1;
  let assemblyFormat = "$value `[` $indices `]` attr-dict `:` functional-type(operands, results)";
}

def EmitPy_LoadOp : EmitPy_Op<"load", [TypesMatchWith<"result type matches value type of 'operand'",
                  "operand", "result",
                  "::llvm::cast<LValueType>($_self).getValueType()">]> {
  let summary = "Load an lvalue into an SSA value.";
  let description = [{
    This operation loads the content of a modifiable lvalue into an SSA value.
    Modifications of the lvalue executed after the load are not observable on
    the produced value.

    Example:

    ```mlir
    %2 = emitpy.load %1 : <!emitpy.opaque<"ttnn.Tensor">>
    ```
    ```Python
    // Code emitted for the operation above.
    v2 = v1;
    ```
  }];

  let arguments = (ins
    Res<EmitPy_LValueType, "", [MemRead<DefaultResource, 0, FullEffect>]>:$operand);

  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

def EmitC_VerbatimOp : EmitPy_Op<"verbatim"> {
  let summary = "Verbatim operation";
  let description = [{
    The `emitc.verbatim` operation produces no results and the value is emitted as is
    followed by a line break  ('\n' character) during translation.

    Note: Use with caution. This operation can have arbitrary effects on the
    semantics of the emitted code. Use semantically more meaningful operations
    whenever possible. Additionally this op is *NOT* intended to be used to
    inject large snippets of code.

    This operation can be used in situations where a more suitable operation is
    not yet implemented in the dialect or where preprocessor directives
    interfere with the structure of the code.
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

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
