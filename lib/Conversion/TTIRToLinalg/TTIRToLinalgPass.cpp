// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/TTIRToLinalg.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOLINALG
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTTIRToLinalgPass
    : public ttir::impl::ConvertTTIRToLinalgBase<ConvertTTIRToLinalgPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<tt::TTDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();


    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToLinalgPatterns(&getContext(), patterns, typeConverter);

    // Apply full conversion
    //
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      llvm::outs() << "failed to apply full TTIR to LinAlg conversion!\n";

      llvm::outs() << "Conversion failed. Remaining IR:\n";
      getOperation()->print(llvm::outs());
      llvm::outs() << "\n";

      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<tt::CPUModuleOp>> createConvertTTIRToLinalgPass() {
  return std::make_unique<ConvertTTIRToLinalgPass>();
}

} // namespace mlir::tt
