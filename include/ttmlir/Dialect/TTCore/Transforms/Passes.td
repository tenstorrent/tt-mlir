// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_TRANSFORMS_PASSES_TD
#define TTMLIR_DIALECT_TTCORE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TTCoreWrapDeviceModulePass : Pass<"ttcore-wrap-device-module", "::mlir::ModuleOp"> {
  let summary = "Wrap top-level builtin.module in ttcore.device_module";
  let description = [{
    Utility pass to convert a top-level ModuleOp into a module wrapped inside a ttcore::DeviceModuleOp.
    Example:
    Input:
    module {
      func.func foo() {}
    }

    Output:
    module {
      ttcore.device_module {
        module {
          func.func foo() {}
        }
      }
    }
  }];
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}

def TTCoreUnwrapDeviceModulePass : Pass<"ttcore-unwrap-device-module", "::mlir::ModuleOp"> {
  let summary = "Move ttcore.device_module contents to top-level builtin.module";
  let description = [{
    Utility pass to extract the ModuleOp inside a ttcore::DeviceModuleOp to the top-level ModuleOp.  This might help run passes via CLI more simply.

    Example:
    Input:
    module {
      ttcore.device_module {
        module {
          func.func foo() {}
        }
      }
    }
    Output:
    module {
      func.func foo() {}
    }
  }];
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}

def TTCoreRegisterDevicePass : Pass<"ttcore-register-device", "::mlir::ModuleOp"> {
  let summary = "Move ttcore.device_module contents to top-level builtin.module";
  let description = [{
    Utility pass to register the default device.
  }];

  list<Option> options = [
        Option<"mockSystemDescArch", "mock-system-desc-arch", "ttcore::Arch",
            /*default=*/"ttcore::Arch::WormholeB0",
            "Arch name for constructing a mock system descriptor in lieu of system-desc-path.",
            [{::llvm::cl::values(
              clEnumValN(ttcore::Arch::WormholeB0, "wormhole_b0",
               "Use mock wormhole_b0 system desc."),
              clEnumValN(ttcore::Arch::Blackhole, "blackhole",
               "Use mock blackhole system desc.")
             )}]>,
        Option<"systemDescPath", "system-desc-path", "std::string", "", "System desc path">,
        ListOption<"meshShape", "mesh-shape", "int64_t", "Set the mesh shape">,
    ];
  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}

def TTCoreOptimizationBarrierFold : Pass<"ttcore-optimization-barrier-fold", "::mlir::ModuleOp">
{
  let summary = "Fold optimization barrier operations.";
  let description = [{
    This pass folds optimization barrier operations by replacing them with their operands.
    Since optimization barriers do not hold and function runtime meaning, they can be safely
    removed.
  }];
}

def TTCoreOneShotBufferizePass : Pass<"ttcore-one-shot-bufferize", "::mlir::ModuleOp"> {
  let summary = "One-Shot Bufferize with TTCore MetalLayoutAttr support";
  let description = [{
    This pass runs one-shot-bufferize with a custom type converter that properly
    handles tensor types with MetalLayoutAttr encoding. It converts such tensors
    to memref types with appropriate layout attributes (ShardLayoutAttr,
    ViewLayoutAttr, InterleavedLayoutAttr) and explicit memory space attributes.
    In addition to function bounadaries, d2m.global tensors are also handled.
  }];
  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect",
    "::mlir::tt::ttcore::TTCoreDialect"
  ];
}

#endif
