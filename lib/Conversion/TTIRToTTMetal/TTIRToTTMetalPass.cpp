// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"

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

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTTIRToTTMetal
    : public ttir::impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal> {

  using Base = ttir::impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal>;

  ConvertTTIRToTTMetal() = default;
  ConvertTTIRToTTMetal(
      const mlir::tt::ttir::ConvertTTIRToTTMetalOptions &options)
      : Base(options) {}

  ConvertTTIRToTTMetal(const ConvertTTIRToTTMetal &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->mathFidelity = rhs.mathFidelity;
  }

  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<ttmetal::TTMetalDialect>();
    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();

    target.addLegalOp<ttir::StreamLayoutOp>();
    target.addLegalOp<ttir::ViewLayoutOp>();

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return !mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
          op.getMemref().getType().getMemorySpace());
    });
    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return !mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
          op.getMemref().getType().getMemorySpace());
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTMetalPatterns(&getContext(), patterns, typeConverter,
                                  mathFidelity);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTMetalPass() {
  return std::make_unique<ConvertTTIRToTTMetal>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTMetalPass(
    const ttir::ConvertTTIRToTTMetalOptions &options) {
  return std::make_unique<ConvertTTIRToTTMetal>(options);
}

} // namespace mlir::tt
