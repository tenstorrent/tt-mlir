// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir::tt::ttkernel {

void registerTTKernelToCpp() {
  TranslateFromMLIRRegistration reg(
      "ttkernel-to-cpp", "translate ttkernel to C++",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateTopLevelKernelsToCpp(mlir::cast<ModuleOp>(op), os);
      },
      [](DialectRegistry &registry) {
        registry.insert<mlir::tt::ttkernel::TTKernelDialect,
                        mlir::tt::ttmetal::TTMetalDialect,
                        mlir::tt::ttcore::TTCoreDialect,
                        mlir::tt::ttir::TTIRDialect, mlir::emitc::EmitCDialect,
                        mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
      });
}

} // namespace mlir::tt::ttkernel
