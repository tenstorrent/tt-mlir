// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMetal_TTMetalPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMetal_TTMetalPASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTTIRToTTMetal: Pass<"convert-ttir-to-ttmetal", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];
}

def ConvertTTMetalKernelsToEmitC: Pass<"convert-ttmetal-kernels-to-cpp", "::mlir::ModuleOp"> {
  let summary = "";
  let description = [{
    todo
  }];
}

#endif
