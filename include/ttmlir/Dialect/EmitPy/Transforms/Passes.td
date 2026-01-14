// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_EMITPY_TRANSFORMS_EMITPYPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_EMITPY_TRANSFORMS_EMITPYPASSES_TD

include "mlir/Pass/PassBase.td"

def EmitPySplitFiles: Pass<"emitpy-split-files-creation", "::mlir::ModuleOp"> {
  let summary = "Organize EmitPy operations into separate files.";
  let description = [{
    This pass organizes EmitPy operations into two files:
    - main.py: Runtime execution logic.
    - consteval.py: Complete consteval logic.
  }];
  let constructor = "createEmitPySplitFiles()";
}

#endif // TTMLIR_TTMLIR_DIALECT_EMITPY_TRANSFORMS_EMITPYPASSES_TD
