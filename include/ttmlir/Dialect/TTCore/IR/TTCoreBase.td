// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_TTCOREBASE_TD
#define TTMLIR_DIALECT_TTCORE_IR_TTCOREBASE_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TTCore dialect definition.
//===----------------------------------------------------------------------===//

def TTCore_Dialect : Dialect {
    let name = "ttcore";
    let summary = "TT core types and attributes common to all TT dialects.";
    let description = [{
        This dialect defines types and attributes common to all TT dialects.
    }];
    let cppNamespace = "::mlir::tt::ttcore";

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
    let dependentDialects = [];
}

//===----------------------------------------------------------------------===//
// Base TT operation definition.
//===----------------------------------------------------------------------===//

class TTCore_Op<string mnemonic, list<Trait> traits = []> :
        Op<TTCore_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Base TT trait definition.
//===----------------------------------------------------------------------===//

class TTCore_Trait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::tt::ttcore::Trait";
}

// Trait for ops which should be duplicated in all const-eval subgraphs + original.
def TTCore_DuplicateConstEvalTrait : TTCore_Trait<"TTCoreDuplicateConstEvalTrait">;
// Trait for ops which create tensors -- these ops will be const-eval hoisted if any of uses are const-eval ops.
def TTCore_CreationOpTrait : TTCore_Trait<"TTCoreCreationOpTrait">;
// Trait for ops which should not be cached/hoisted for const-eval (e.g., rand, empty).
def TTCore_NonCacheableTrait : TTCore_Trait<"TTCoreNonCacheableTrait">;

#endif // TTMLIR_DIALECT_TTCORE_IR_TTCOREBASE_TD
