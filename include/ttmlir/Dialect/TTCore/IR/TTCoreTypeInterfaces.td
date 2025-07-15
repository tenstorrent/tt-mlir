// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TT_TTTYPEINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TT_TTTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def BlockTypeInterface : TypeInterface<"BlockType"> {
  let cppNamespace = "::mlir::tt";
  let description = [{
    Indicates that the type is a block type.
  }];
}

#endif // TTMLIR_TTMLIR_DIALECT_TT_TTTYPEINTERFACES_TD
