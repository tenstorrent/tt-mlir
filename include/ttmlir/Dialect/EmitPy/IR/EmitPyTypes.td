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

def EmitPy_ClassType : EmitPy_Type<"Class", "class"> {
  let summary = "EmitPy class type";

  let description = [{
    A type representing a Python class defined in EmitPy.

    Example:

    ```mlir
    !emitpy.class<"Model">
    ```
  }];

  let parameters = (ins StringRefParameter<"the class name">:$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}

def EmitPy_DictType : EmitPy_Type<"Dict", "dict"> {
  let summary = "EmitPy dictionary type";

  let description = [{
    A type representing a Python dictionary.
    Key and value types are optional, allowing heterogeneous dictionaries like in Python.

    Example:

    ```mlir
    !emitpy.dict                    // Untyped dictionary (Any, Any)
    !emitpy.dict<i32, f32>          // Both key and value types specified
    !emitpy.dict<!emitpy.opaque<"str">, !emitpy.opaque<"ttnn.Tensor">>
    ```
  }];

  let parameters = (ins
    OptionalParameter<"Type">:$keyType,
    OptionalParameter<"Type">:$valueType
  );

  let builders = [];

  let hasCustomAssemblyFormat = 1;
}

// Type constraint that matches any EmitPy dialect type
def EmitPy_Type : AnyTypeOf<[EmitPy_OpaqueType, EmitPy_ClassType, EmitPy_DictType]>;

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_IR_EMITPYTYPES_TD
