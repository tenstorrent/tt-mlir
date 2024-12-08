// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"


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
    let useDefaultAttributePrinterParser = 1;
    let dependentDialects = [
      "::mlir::arith::ArithDialect",
      "::mlir::func::FuncDialect",
      "::mlir::tensor::TensorDialect",
      "::mlir::linalg::LinalgDialect",
      "::mlir::scf::SCFDialect",
      "::mlir::cf::ControlFlowDialect",
      "::mlir::tt::TTDialect"
    ];

    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base TTIR operation definition.
//===----------------------------------------------------------------------===//

class TTIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTIR_Dialect, mnemonic, !listconcat([Pure], traits)>;

//===----------------------------------------------------------------------===//
// TTIR traits definition.
//===----------------------------------------------------------------------===//

def TwoOperands : ParamNativeOpTrait<"NOperands", "2">;
def ThreeOperands : ParamNativeOpTrait<"NOperands", "3">;
def FourOperands : ParamNativeOpTrait<"NOperands", "4">;

class TTIR_Trait<string name, list<Trait> traits = []> : NativeOpTrait<name, traits> {
  let cppNamespace = "::mlir::tt::ttir::OpTrait";
}

def TTIR_Involution : TTIR_Trait<"TTIRInvolution", [DestinationStyleOpInterface, TwoOperands, NoMemoryEffect]>;
def TTIR_Idempotence : TTIR_Trait<"TTIRIdempotence", [DestinationStyleOpInterface, TwoOperands, NoMemoryEffect]>;

#endif
