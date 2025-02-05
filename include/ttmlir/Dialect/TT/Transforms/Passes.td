// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TT_TTPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TT_TTPASSES_TD

include "mlir/Pass/PassBase.td"

def TTWrapDeviceModulePass : Pass<"tt-wrap-device-module", "::mlir::ModuleOp"> {
  let summary = "Wrap top-level builtin.module in tt.device_module";
  let description = [{
    Utility pass to convert a top-level ModuleOp into a module wrapped inside a tt::DeviceModuleOp.
    Example:
    Input:
    module {
      func.func foo() {}
    }

    Output:
    module {
      tt.device_module {
        module {
          func.func foo() {}
        }
      }
    }
  }];
  let dependentDialects = ["::mlir::tt::TTDialect"];
}

def TTUnwrapDeviceModulePass : Pass<"tt-unwrap-device-module", "::mlir::ModuleOp"> {
  let summary = "Move tt.device_module contents to top-level builtin.module";
  let description = [{
    Utility pass to extract the ModuleOp inside a tt::DeviceModuleOp to the top-level ModuleOp.  This might help run passes via CLI more simply.

    Example:
    Input:
    module {
      tt.device_module {
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
  let dependentDialects = ["::mlir::tt::TTDialect"];
}

#endif
