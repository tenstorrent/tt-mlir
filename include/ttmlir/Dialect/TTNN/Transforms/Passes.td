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

def TTNNOptimizer: Pass<"ttnn-optimizer", "::mlir::ModuleOp"> {
  let summary = "Determine op configurations for maximum performance.";
  let description = [{
    Go through the ops, set sharding specs for each op based on sharding analysis,
    by updating layout attribute of each op.
  }];
  let options = [
    Option<"overrideOutputLayout", "override-output-layout",
          "llvm::StringMap<LayoutOverrideParams>",
          /*default=*/"llvm::StringMap<LayoutOverrideParams>()",
           "Override output tensor layout for specific ops.">,
    Option<"shardingPassEnabled", "sharding-pass-enabled",
          "bool",
          /*default=*/"false",
           "Enable sharding pass.">,
    Option<"reshardingEnabled", "resharding-enabled",
          "bool",
          /*default=*/"false",
          "Resharding pass. Temp disabled till we support all types of shard specs.">,
    Option<"maxLegalLayouts", "max-legal-layouts",
          "int64_t",
          /*default=*/"64",
          "Override maximum number of legal layouts for grid analysis.">
  ];
}

#endif
