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

def EmitPy_DecoratorAttr : EmitPy_Attr<"Decorator", "decorator"> {
  let summary = "A Python decorator attribute";

  let description = [{
    A decorator attribute that represents a Python decorator to be applied to a function.
    Multiple decorators can be applied using an array of DecoratorAttr.

    Example:

    ```mlir
    #emitpy.decorator<"@pytest.mark.parametrize('x', [1, 2, 3])">
    #emitpy.decorator<"@staticmethod">
    ```
  }];

  let parameters = (ins StringRefParameter<"the decorator string">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def EmitPy_OpaqueOrTypedAttr : AnyAttrOf<[EmitPy_OpaqueAttr, TypedAttrInterface]>;

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYATTRIBUTES_TD
