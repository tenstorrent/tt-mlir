// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYATTRIBUTES_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYATTRIBUTES_TD

include "ttmlir/Dialect/EmitPy/IR/EmitPyBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EmitPy attribute definitions
//===----------------------------------------------------------------------===//

class EmitPy_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<EmitPy_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def EmitPy_OpaqueAttr : EmitPy_Attr<"Opaque", "opaque"> {
  let summary = "An opaque attribute";

  let description = [{
    An opaque attribute of which the value gets emitted as is.

    Example:

    ```mlir
    #emitpy.opaque<"">
    #emitpy.opaque<"None">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def EmitPy_OpaqueOrTypedAttr : AnyAttrOf<[EmitPy_OpaqueAttr, TypedAttrInterface]>;

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYATTRIBUTES_TD
