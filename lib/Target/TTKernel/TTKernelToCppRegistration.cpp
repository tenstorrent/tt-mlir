// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

namespace mlir::tt::ttkernel {

void registerTTKernelToCpp() {
  TranslateFromMLIRRegistration reg(
      "ttkernel-to-cpp", "translate ttmetal dialect to flatbuffer",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateTTKernelToCpp(op, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<mlir::scf::SCFDialect,
                        mlir::tt::ttkernel::TTKernelDialect, mlir::arith::ArithDialect,
                        mlir::emitc::EmitCDialect, mlir::func::FuncDialect, mlir::tt::TTDialect>();
      });
}

} // namespace mlir::tt::ttkernel
