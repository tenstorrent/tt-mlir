// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
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

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_CONVERTD2MTOTTMETAL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::d2m

namespace {

struct ConvertD2MToTTMetal
    : public d2m::impl::ConvertD2MToTTMetalBase<ConvertD2MToTTMetal> {

  using Base = d2m::impl::ConvertD2MToTTMetalBase<ConvertD2MToTTMetal>;

  ConvertD2MToTTMetal() = default;
  ConvertD2MToTTMetal(const mlir::tt::d2m::ConvertD2MToTTMetalOptions &options)
      : Base(options) {}

  ConvertD2MToTTMetal(const ConvertD2MToTTMetal &rhs) : Base(rhs) {
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
    target.addIllegalDialect<d2m::D2MDialect>();

    target.addLegalOp<d2m::StreamLayoutOp>();
    target.addLegalOp<d2m::ViewLayoutOp>();

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      // Legal if no memory space (null) OR if it's System memory (host buffer)
      auto memSpace = mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
          op.getMemref().getType().getMemorySpace());
      if (!memSpace)
        return true;  // No memory space = host, legal
      return ttcore::isSystemMemorySpace(memSpace.getValue());  // System memory = host, legal
    });
    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      // Legal if no memory space (null) OR if it's System memory (host buffer)
      auto memSpace = mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
          op.getMemref().getType().getMemorySpace());
      if (!memSpace)
        return true;  // No memory space = host, legal
      return ttcore::isSystemMemorySpace(memSpace.getValue());  // System memory = host, legal
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateD2MToTTMetalPatterns(&getContext(), patterns, typeConverter,
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

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTMetalPass() {
  return std::make_unique<ConvertD2MToTTMetal>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertD2MToTTMetalPass(const d2m::ConvertD2MToTTMetalOptions &options) {
  return std::make_unique<ConvertD2MToTTMetal>(options);
}

} // namespace mlir::tt
