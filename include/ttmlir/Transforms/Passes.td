// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TRANSFORMS_PASSES_TD
#define TTMLIR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConstEvalHoistTransform: Pass<"const-eval-hoist-transform", "::mlir::ModuleOp"> {
  let summary = "Transform to hoist const-eval subgraph into separate func(s)";
  let description = [{
    Transform pass which runs an analysis pass to find const-eval subgraphs in an input
    and hoist them into separate functions. This pass is dialect-agnostic and works with
    any dialect that implements the ConstEvalHoistingInterface.

    Example: (TODO)
  }];

  let dependentDialects = ["::mlir::tt::TTDialect"];
}

#endif
