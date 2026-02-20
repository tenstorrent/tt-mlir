// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"

//===----------------------------------------------------------------------===//
// TTIR type definitions
//===----------------------------------------------------------------------===//

class TTIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TTIR_MemTx : TTIR_Type<"MemTx", "mem_tx"> {
    let summary = "TTIR memory transaction type.";
    let description = "Memory transaction returned by dma op, used to wait for completion.";
}

def TTIR_Semaphore : TTIR_Type<"Semaphore", "semaphore"> {
    let summary = "TTIR semaphore type.";
    let description = "Semaphore primitive type used in combination with semaphore ops to synchronize cores.";
}

def TTIR_ComplexTensor : TTIR_Type<"ComplexTensor", "complex_tensor"> {
    let summary = "TTIR complex tensor type.";
    let description = [{
        A complex tensor type represented as an optional real tensor and a
        required imaginary tensor. When the real component is absent, the type
        represents a purely imaginary complex tensor (e.g. the result of
        TTIR_ImagOp). This is distinct from MLIR's built-in complex type,
        which operates on scalar element types.
    }];
    let parameters = (ins
        OptionalParameter<"::mlir::RankedTensorType">:$realType,
        "::mlir::RankedTensorType":$imagType
    );
    let assemblyFormat = "`<` (`real` `=` $realType^ `,`)? `imag` `=` $imagType `>`";
    let genVerifyDecl = 1;
    let extraClassDeclaration = [{
        bool hasRealType() const { return static_cast<bool>(getRealType()); }
    }];
}

#endif  // TTMLIR_TTMLIR_DIALECT_TTIR_IR_TTIROPSTYPES_TD
