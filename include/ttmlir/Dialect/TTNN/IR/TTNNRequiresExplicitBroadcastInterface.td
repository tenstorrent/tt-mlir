// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNREQUIRESBROADCASTINTERFACE_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNREQUIRESBROADCASTINTERFACE_TD

include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"

def TTNN_RequiresExplicitBroadcastInterface : OpInterface<"RequiresExplicitBroadcastInterface"> {
  let description = "Used to specify that an operation needs explicit broadcast on all operands.";
  let cppNamespace = "::mlir::tt::ttnn";

  let verify = [{
    if ($_op->getOperands().size() <= 1) {
      return $_op->emitOpError()
        << "RequiresExplicitBroadcastInterface can only be attached to the "
           "operations that have more than one operand";
    }
    return success();
  }];
}

#endif
