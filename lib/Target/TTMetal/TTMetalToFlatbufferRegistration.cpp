// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Target/TTMetal/TTMetalToFlatbuffer.h"

using namespace mlir;

namespace mlir::tt::ttmetal {

void registerTTMetalToFlatbuffer() {
  TranslateFromMLIRRegistration reg(
      "ttmetal-to-flatbuffer", "translate ttmetal dialect to flatbuffer",
      translateTTMetalToFlatbuffer /* function */,
      [](DialectRegistry &registry) {
        registry.insert<mlir::tt::TTDialect, mlir::tt::ttmetal::TTMetalDialect,
                        mlir::tt::ttkernel::TTKernelDialect,
                        mlir::func::FuncDialect, mlir::emitc::EmitCDialect>();
      });
}

} // namespace mlir::tt::ttmetal
