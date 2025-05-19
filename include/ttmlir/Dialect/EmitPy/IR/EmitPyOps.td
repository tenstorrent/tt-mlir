#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYOPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

include "EmitPyBase.td"

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
    Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let hasVerifier = 1;
}

#endif
