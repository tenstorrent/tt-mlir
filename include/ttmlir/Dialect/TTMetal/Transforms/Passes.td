// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD

include "mlir/Pass/PassBase.td"

def TTMetalControlDstSection: Pass<"ttmetal-control-dst-section", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    Analyze the graph and insert tile_regs_* ops to control the dst critical section.
  }];
}

#endif
