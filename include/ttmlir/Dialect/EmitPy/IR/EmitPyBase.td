#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_EMITPYDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EmitPy dialect definition.
//===----------------------------------------------------------------------===//

def EmitPy_Dialect : Dialect {
    let name = "emitpy";
    let cppNamespace = "::mlir::tt::emitpy";

    let summary = "Dialect to generate Python from MLIR.";
    let description = [{
      [include "Dialects/emitpy.md"]
    }];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base EmitPy operation definition.
//===----------------------------------------------------------------------===//

class EmitPy_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitPy_Dialect, mnemonic, traits>;

#endif
