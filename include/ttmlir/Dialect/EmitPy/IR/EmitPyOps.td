// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

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

    // Generic form of the same operation.
    %0 = "emitpy.call_opaque"() {callee = "foo"} : () -> i32
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

  let assemblyFormat = [{$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)}];

  let hasVerifier = 1;
}

def EmitPy_ImportOp : EmitPy_Op<"import"> {
  let summary = "Import operation";
  let description = [{
     The `emitpy.import` operation allows to define a Python module import
     via various forms of the `import` statement.
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

def EmitPy_MainOp : EmitPy_Op<"call_main"> {
  let summary = "Program entry point";
  let description = [{
    The `emitpy.call_main` allows the addition of
    ```if __name__ == '__main__':
      main()
    ``` during emission of Python code.
  }];
}

#endif
