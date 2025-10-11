// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"

using namespace mlir;

namespace mlir::tt::ttnn {

// Command line option for kernel dump directory
static llvm::cl::opt<std::string>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    kernelDumpDir("kernel-dump-dir",
                  llvm::cl::desc("Directory to dump translated kernel sources "
                                 "during compilation (default: no dumping)"),
                  llvm::cl::init(""));

void registerTTNNToFlatbuffer() {
  TranslateFromMLIRRegistration reg(
      "ttnn-to-flatbuffer", "translate ttnn to flatbuffer",
      [](Operation *op, llvm::raw_ostream &os) -> LogicalResult {
        return translateTTNNToFlatbuffer(op, os, {}, {}, kernelDumpDir);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<mlir::tt::ttcore::TTCoreDialect,
                        mlir::tt::ttnn::TTNNDialect,
                        mlir::tt::ttkernel::TTKernelDialect,
                        mlir::func::FuncDialect,
                        mlir::emitc::EmitCDialect,
                        mlir::LLVM::LLVMDialect,
                        mlir::quant::QuantDialect
                        >();
        // clang-format on
        registerAllToLLVMIRTranslations(registry);
      });
}

} // namespace mlir::tt::ttnn
