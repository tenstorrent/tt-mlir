// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNN_REQUIRES_EXPLICIT_BROADCAST_INTERFACE_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNN_REQUIRES_EXPLICIT_BROADCAST_INTERFACE_TD

include "mlir/IR/OpBase.td"

// This interface is used to specify that an operation needs explicit broadcast on all operands.
def TTNN_RequiresExplicitBroadcastInterface : OpInterface<"TTNNRequiresExplicitBroadcastInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";

  let verify = [{
    return verifyTTNNRequiresExplicitBroadcastInterface($_op);
  }];
}

#endif
