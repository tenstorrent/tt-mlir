#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYATTRIBUTES_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "ttmlir/Dialect/EmitPy/IR/EmitPyBase.td"

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
    #emitpy.opaque<"nullptr">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYATTRIBUTES_TD
