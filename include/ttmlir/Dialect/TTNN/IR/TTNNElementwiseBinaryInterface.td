// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNELEMENTWISEBINARYINTERFACE_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNELEMENTWISEBINARYINTERFACE_TD

include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"

def TTNN_ElementwiseBinary : OpInterface<"ElementwiseBinary"> {
  let cppNamespace = "::mlir::tt::ttnn";

  let verify = [{
    if ($_op->getOperands().size() != 2) {
      return $_op->emitOpError()
        << "ElementwiseBinary interface can only be attached to the "
           "operations that have more than one operand";
    }
    return success();
  }];
}

#endif
