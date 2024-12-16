// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstddef>
#include <memory>

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Target/Cpp/CppEmitter.h"

#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

namespace mlir::tt::ttkernel {

static void translateModuleToCpp(
    Operation *op, llvm::raw_ostream &os) {
    ModuleOp module = dyn_cast<ModuleOp>(op);
    assert(module && "Expected ModuleOp as top level operation");
    mlir::PassManager pm(op->getContext());

    pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
    pm.addPass(mlir::createConvertArithToEmitC());
    pm.addPass(mlir::createSCFToEmitC());
    pm.addPass(mlir::createConvertFuncToEmitC());

    if (mlir::failed(pm.run(op))) {
      throw std::runtime_error("Failed to lower MLIR to EmitC");
    }

    if ( mlir::failed( mlir::emitc::translateToCpp(op, os) ) ) {
      throw std::runtime_error("Failed to write C++ code to file");
    }
}

LogicalResult translateTTKernelToCpp(
    Operation *op, llvm::raw_ostream &os) {
  translateModuleToCpp(op, os);
  return success();
}

} // namespace mlir::tt::ttkernel
