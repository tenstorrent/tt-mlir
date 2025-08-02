// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TT_TTTYPEINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TT_TTTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def BlockTypeInterface : TypeInterface<"BlockType"> {
  let cppNamespace = "::mlir::tt";
  let description = [{
    Interface used to tag all block types. Intent is to use this interface
    to check if a type is a block type, similar to how we check if type is a
    FloatType without needing to know the specific type.
  }];
}

#endif // TTMLIR_TTMLIR_DIALECT_TT_TTTYPEINTERFACES_TD
