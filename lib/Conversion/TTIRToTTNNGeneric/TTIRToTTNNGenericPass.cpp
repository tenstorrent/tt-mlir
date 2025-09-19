// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNNGeneric/TTIRToTTNNGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTNNGENERIC
#include "ttmlir/Conversion/Passes.h.inc" // impl::ConvertTTIRToTTNNGenericBase

// ............................................................................
using namespace llvm;

namespace {

struct ConvertTTIRToTTNNGenericPass final
    : impl::ConvertTTIRToTTNNGenericBase<ConvertTTIRToTTNNGenericPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<ttnn::TTNNDialect>();
    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();

    target.addLegalOp<ttir::StreamLayoutOp>();
    target.addLegalOp<ttir::ViewLayoutOp>();

    target.addLegalOp<memref::AllocOp>();
    target.addLegalOp<memref::DeallocOp>();

    target.addLegalOp<ttir::EmptyOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTNNGenericPatterns(&getContext(), patterns, typeConverter);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
// ............................................................................
namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNGenericPass() {
  return std::make_unique<ttir::ConvertTTIRToTTNNGenericPass>();
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
