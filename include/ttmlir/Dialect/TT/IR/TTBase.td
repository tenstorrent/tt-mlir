// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTDIALECT_TD
#define TTMLIR_TTMLIR_TTDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TT dialect definition.
//===----------------------------------------------------------------------===//

def TT_Dialect : Dialect {
    let name = "tt";
    let summary = "TT types and attributes common to all TT dialects.";
    let description = [{
        This dialect defines types and attributes common to all TT dialects.
    }];
    let cppNamespace = "::mlir::tt";

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

class TT_Op<string mnemonic, list<Trait> traits = []> :
        Op<TT_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Base TT trait definition.
//===----------------------------------------------------------------------===//

class TT_Trait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::mlir::tt::Trait";
}

// Trait for ops which should never be hoisted by const-eval pass.
def TT_IgnoreConstEvalTrait : TT_Trait<"TTIgnoreConstEvalTrait">;
// Trait for ops which should be duplicated in all const-eval subgraphs + original.
def TT_ForkConstEvalTrait : TT_Trait<"TTForkConstEvalTrait">;
// Trait for ops which create tensors -- these ops will be const-eval hoisted if any of uses are const-eval ops.
def TT_CreationOpTrait : TT_Trait<"TTCreationOpTrait">;
#endif
