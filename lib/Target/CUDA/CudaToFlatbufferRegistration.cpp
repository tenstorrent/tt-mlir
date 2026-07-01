// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/CUDA/CudaToFlatbuffer.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir::tt::cuda {

void registerCudaToFlatbuffer() {
  TranslateFromMLIRRegistration reg(
      "ptx-to-flatbuffer", "translate program with ptx to flatbuffer",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateCudaToFlatbuffer(op, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<mlir::tt::ttcore::TTCoreDialect, mlir::gpu::GPUDialect,
                        mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                        mlir::func::FuncDialect, mlir::arith::ArithDialect,
                        mlir::memref::MemRefDialect>();
        registerAllToLLVMIRTranslations(registry);
        registerConvertNVVMToLLVMInterface(registry);
        registerConvertMemRefToLLVMInterface(registry);
        arith::registerConvertArithToLLVMInterface(registry);
        registerConvertFuncToLLVMInterface(registry);
      });
}

} // namespace mlir::tt::cuda
