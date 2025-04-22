// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTKERNEL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {
struct ConvertTTIRToTTKernel
    : public ttir::impl::ConvertTTIRToTTKernelBase<ConvertTTIRToTTKernel> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<ttmetal::TTMetalDialect>();
    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();

    target.addLegalOp<tt::DeviceOp>();
    target.addLegalOp<ttir::StreamLayoutOp>();
    target.addLegalOp<ttir::ViewLayoutOp>();
    target.addLegalOp<ttir::GenericOp>();
    target.addIllegalOp<memref::StoreOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      if (!op->hasAttr("ttir.thread_type")) {
        return true;
      }
      if (std::all_of(
              op.getArgumentTypes().begin(), op.getArgumentTypes().end(),
              [&](Type type) { return !mlir::isa<MemRefType>(type); })) {
        return true;
      }
      return false;
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTKernelPatterns(&getContext(), patterns, typeConverter);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  };
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTKernelPass() {
  return std::make_unique<ConvertTTIRToTTKernel>();
}

} // namespace mlir::tt
