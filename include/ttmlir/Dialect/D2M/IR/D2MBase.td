// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_D2M_D2MDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_D2M_D2MDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// D2M dialect definition (DirectToMetal subset of TTIR).
//===----------------------------------------------------------------------===//

def D2M_Dialect : Dialect {
    let name = "d2m";
    let summary = "Direct-to-metal subset of TTIR";
    let description = [{
      The D2M dialect contains the subset of TTIR used by the
      direct-to-metal lowering path (TTMetal). It hosts generic
      dispatch ops and related region ops required post TTIRToTTIRGeneric.
    }];
    let cppNamespace = "::mlir::tt::d2m";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class D2M_Op<string mnemonic, list<Trait> traits = []> :
        Op<D2M_Dialect, mnemonic, traits>;

// Common type shortcuts used across D2M ops
def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

//===----------------------------------------------------------------------===//
// D2M traits definition.
//===----------------------------------------------------------------------===//

class D2M_Trait<string name, list<Trait> traits = []> : NativeOpTrait<name, traits> {
  let cppNamespace = "::mlir::tt::d2m";
}

// Marker traits mirroring TTIR's compute/datamovement region traits
def D2M_GenericRegionComputeOpTrait : D2M_Trait<"D2MGenericRegionComputeOpTrait">;
def D2M_GenericRegionDatamovementOpTrait : D2M_Trait<"D2MGenericRegionDatamovementOpTrait">;

#endif // TTMLIR_TTMLIR_DIALECT_D2M_D2MDIALECT_TD
