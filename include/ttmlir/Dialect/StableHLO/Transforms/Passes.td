// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_StableHLO_DIALECT_PASSES_TD
#define TTMLIR_StableHLO_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"
include "shardy/dialect/sdy/ir/dialect.td"
include "shardy/dialect/sdy/ir/op_interface.td"

def ApplyArgumentShardStatusPass : Pass<"apply-argument-shard-status", "::mlir::ModuleOp">
{
  let summary = "Annotate arguments with their shard status.";
  let description = [{
    This pass will analyze the module arguments and annotate arguments with whether they are pre-sharded or not. It will determine this based on analyzing shardy annotations.
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def AnalyzeMeshPass : Pass<"analyze-mesh", "::mlir::ModuleOp">
{
  let summary = "Analyze the mesh compilation target.";
  let description = [{
    This pass will analyze the mesh of the graph and update shardings or annotations to match the target device. It will attempt to automatically parallelize if user provided option.
  }];

  let options = [
    ListOption<"meshShape", "mesh-shape", "int64_t", "Set the mesh shape">,
    Option<"automaticArgAnalysis", "automatic-arg-analysis", "bool", /*default=*/"false", "Automatically determine argument shardings">,
  ];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def WrapUnderManualComputationPass : Pass<"wrap-under-manual-computation", "::mlir::ModuleOp">
{
  let summary = "Wrap all operations within a sdy manual computation op.";
  let description = [{
    This pass will wrap all the operations within a module under a manual computation op that defines per device tensor shapes.
  }];

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

def UpdateGlobalToLocalShapesPass : Pass<"update-global-to-local-shapes", "::mlir::ModuleOp">
{
  let summary = "Update all the tensor dimensions based on their tensor sharding annotation from global shapes to local per device shapes.";

  let dependentDialects = [
    "::mlir::sdy::SdyDialect"
  ];
}

#endif
