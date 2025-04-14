// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_StableHLO_DIALECT_PASSES_TD
#define TTMLIR_StableHLO_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"
include "shardy/dialect/sdy/ir/dialect.td"
include "shardy/dialect/sdy/ir/op_interface.td"

def ShardyAutomaticParallelization: Pass<"shardy-automatic-parallelization", "::mlir::ModuleOp">
{
  let summary = "Perform shardy automatic parallelization.";
  let description = [{
    This pass performs shardy automatic parallelization in the compiler to split the graph across multiple devices.
  }];

  let options = [
    ListOption<"meshShape", "mesh-shape", "int64_t", "Set the mesh shape">,
  ];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

#endif
