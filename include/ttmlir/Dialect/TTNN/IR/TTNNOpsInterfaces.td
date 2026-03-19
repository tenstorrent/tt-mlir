// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTNN_ElementwiseUnary : OpInterface<"ElementwiseUnary"> {
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_ElementwiseBinary : OpInterface<"ElementwiseBinary"> {
  let cppNamespace = "::mlir::tt::ttnn";
}

#endif
