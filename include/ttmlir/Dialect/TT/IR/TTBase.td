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
// Aux TT definitions.
//===----------------------------------------------------------------------===//

def TT_Bool : TypeAlias<I1, "Alias for boolean which is a 1-bit integer">;

#endif
