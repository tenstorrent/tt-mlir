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

def TTNNDecomposeLayouts: Pass<"ttnn-decompose-layouts", "::mlir::ModuleOp"> {
  let summary = "Decompose ToLayoutOps to more granular memory ops.";
  let description = [{
    This pass decomposes ToLayoutOps to memory ops (e.g. toDevice, toMemoryConfig etc.).
  }];
}

def TTNNLayout : Pass<"ttnn-layout", "::mlir::ModuleOp"> {
  let summary = "Add layout information to tensors.";
  let description = [{
    This pass adds layout information to tensors.
  }];
}

def TTNNWorkarounds : Pass<"ttnn-workaround", "::mlir::ModuleOp"> {
  let summary = "Apply TTNN workarounds to the IR.";
  let description = [{
    This pass applies necessary TTNN workarounds to the IR in order to create
    a valid and functional IR that can be executed on the hardware.
  }];
}

def TTNNCreateInputGenerators: Pass<"ttnn-create-input-gens", "::mlir::ModuleOp"> {
  let summary = "Create input generators for the forward functions.";
  let description = [{
    This pass creates input generators for the "forward" functions. It
    additionally creates a main function to run the forward function with the
    generated inputs.

    The pass is useful for EmitC path. By creating input generators before
    converting to Emitc Dialect, followed by transformation to C++ code, the
    resulting code won't require any edits to run.

    Given a forward function like this:

    ```
    func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
      %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      return %0 : tensor<32x32xbf16>
    }
    ```

    The pass will create two function like this:

    ```
    func.func @createInputsFor_add() -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
      %0 = "ttnn.empty"() <{shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16>
      %1 = "ttnn.empty"() <{shape = #ttnn.shape<32x32>}> : () -> tensor<32x32xbf16>
      return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
    }

    func.func @main() -> i32 {
      %0:2 = call @createInputsFor_add() : () -> (tensor<32x32xbf16>, tensor<32x32xbf16>)
      %1 = call @add(%0#0, %0#1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
      %c0_i32 = arith.constant 0 : i32
      return %c0_i32 : i32
    }
    ```
  }];
}

#endif
