// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// TTIR dialect definition.
//===----------------------------------------------------------------------===//

def TTIR_Dialect : Dialect {
    let name = "ttir";
    let summary = "TTIR dialect provides high level semantics for dispatching work to TT HW.";
    let description = [{
        This dialect provides high level semantics for dispatching work to TT HW.
        It defines a set of declarative/high level operations that are used to
        describe the dispatch, but is largely agnostic to the set of operations
        or dialects that are actually supported by a consuming backend.
    }];
    let cppNamespace = "::mlir::tt::ttir";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::func::FuncDialect",
      "::mlir::tensor::TensorDialect",
      "::mlir::linalg::LinalgDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::ttcore::TTCoreDialect"
    ];

    let hasConstantMaterializer = 1;

    let extraClassDeclaration = [{
        void registerTypes();

    private:
        // Logger for IR dumping, owned by the dialect
        // Forward declare to avoid circular dependency
        struct MLIRModuleLoggerHolder;
        std::unique_ptr<MLIRModuleLoggerHolder> irLogger;
    }];
}

//===----------------------------------------------------------------------===//
// Base TTIR operation definition.
//===----------------------------------------------------------------------===//

class TTIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TTIR traits definition.
//===----------------------------------------------------------------------===//

class TTIR_Trait<string name, list<Trait> traits = []> : NativeOpTrait<name, traits> {
  let cppNamespace = "::mlir::tt::ttir";
}

def TwoOperands : ParamNativeOpTrait<"NOperands", "2">;
def ThreeOperands : ParamNativeOpTrait<"NOperands", "3">;

// Involution is property of an operation where applying the operation twice results in the original value.
// Example: not(not(x)) == x
def TTIR_Involution : TTIR_Trait<"TTIRInvolution", [DestinationStyleOpInterface, TwoOperands, NoMemoryEffect]>;
// Idempotence is property of an operation where applying the operation twice results in the same value as applying it once.
// Example: abs(abs(x)) == abs(x)
def TTIR_Idempotence : TTIR_Trait<"TTIRIdempotence", [DestinationStyleOpInterface, TwoOperands, NoMemoryEffect]>;
// BinaryIdempotence is property of a binary operation where applying the operation on the same value results in the same value.
// Example: and(x, x) == x
def TTIR_BinaryIdempotence : TTIR_Trait<"TTIRBinaryIdempotence", [DestinationStyleOpInterface, ThreeOperands, NoMemoryEffect]>;
// GenericRegionComputeOpTrait is a trait that verifies that the operation is inside a compute thread.
def TTIR_GenericRegionComputeOpTrait : TTIR_Trait<"TTIRGenericRegionComputeOpTrait">;
// GenericRegionDatamovementOpTrait is a trait that verifies that the operation is inside a datamovement thread.
def TTIR_GenericRegionDatamovementOpTrait : TTIR_Trait<"TTIRGenericRegionDatamovementOpTrait">;

def TTIR_Broadcastable : TTIR_Trait<"Broadcastable">;

//===----------------------------------------------------------------------===//
// TTIR common types.
//===----------------------------------------------------------------------===//

def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;

#endif
