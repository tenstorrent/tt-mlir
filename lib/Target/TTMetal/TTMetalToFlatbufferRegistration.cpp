// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Target/TTMetal/TTMetalToFlatbuffer.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir::tt::ttmetal {

void registerTTMetalToFlatbuffer() {
  TranslateFromMLIRRegistration reg(
      "ttmetal-to-flatbuffer", "translate ttmetal dialect to flatbuffer",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateTTMetalToFlatbuffer(op, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<mlir::tt::TTDialect, mlir::tt::ttmetal::TTMetalDialect,
                        mlir::tt::ttkernel::TTKernelDialect,
                        mlir::emitc::EmitCDialect, mlir::memref::MemRefDialect,
                        mlir::func::FuncDialect>();
      });
}

} // namespace mlir::tt::ttmetal
