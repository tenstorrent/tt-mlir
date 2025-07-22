// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_StableHLO_DIALECT_PASSES_TD
#define TTMLIR_StableHLO_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"
include "shardy/dialect/sdy/ir/dialect.td"
include "shardy/dialect/sdy/ir/op_interface.td"

def ShardyAnnotateArgumentsPass : Pass<"shardy-annotate-arguments", "::mlir::ModuleOp">
{
  let summary = "Annotate arguments with shardy tensor annotations.";
  let description = [{
    This pass will analyze the module and annotate all the arguments with their respective shardy tensor annotations. It will use existing annotations or determine new ones to support.
  }];

  let options = [
    ListOption<"meshShape", "mesh-shape", "int64_t", "Set the mesh shape">,
    Option<"automaticArgAnalysis", "automatic-arg-analysis", "bool", /*default=*/"false", "Automatically determine argument shardings">,
  ];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def ShardyWrapManualComputationPass : Pass<"shardy-wrap-manual-computation", "::mlir::ModuleOp">
{
  let summary = "Wrap all operations within a shardy manual computation op.";
  let description = [{
    This pass will wrap all the operations within a module under a manual computation op that defines per device tensor shapes.
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def UpdateAutomaticShardShapesPass : Pass<"update-automatic-shard-shapes", "::mlir::ModuleOp">
{
  let summary = "Update all the tensor dimensions based on their tensor sharding annotation.";

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

#endif
