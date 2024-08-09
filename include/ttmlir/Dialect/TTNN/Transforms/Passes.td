// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TRANSFORMS_TTNNPASSES_TD

include "mlir/Pass/PassBase.td"

def TTNNOpenDevice: Pass<"ttnn-open-device", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];
}

def TTNNGeneric: Pass<"ttnn-generic", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Wrap top level ops in a generic op.
  }];
}

#endif
