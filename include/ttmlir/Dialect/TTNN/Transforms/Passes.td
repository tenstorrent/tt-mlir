// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD

include "mlir/Pass/PassBase.td"

def TTNNDeallocate: Pass<"ttnn-deallocate", "::mlir::ModuleOp"> {
  let summary = "Insert deallocate ops for tensors.";
  let description = [{
    This pass inserts deallocate ops after a tensor value's last use.
  }];
}

def TTNNDecomposeLayouts: Pass<"ttnn-decompose-layouts", "::mlir::ModuleOp"> {
  let summary = "Decompose ToLayoutOps to more granular memory ops.";
  let description = [{
    This pass decomposes ToLayoutOps to memory ops (e.g. toDevice, toMemoryConfig etc.).
  }];
}

def TTNNLayout : Pass<"ttnn-layout", "::mlir::ModuleOp"> {
  let summary = "Add layout information to tensors.";
  let description = [{
    This pass adds layout information to tensors.
  }];
}

#endif
