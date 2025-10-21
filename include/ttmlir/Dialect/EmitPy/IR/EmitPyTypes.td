// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYTYPES_TD

include "ttmlir/Dialect/EmitPy/IR/EmitPyBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// EmitPy type definitions
//===----------------------------------------------------------------------===//

class EmitPy_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EmitPy_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def EmitPy_OpaqueType : EmitPy_Type<"Opaque", "opaque"> {
  let summary = "EmitPy opaque type";

  let description = [{
    An opaque data type of which the value gets emitted as is.

    Example:

    ```mlir
    !emitpy.opaque<"ttnn.Tensor">
    !emitpy.opaque<"[ttnn.Tensor]">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYTYPES_TD
