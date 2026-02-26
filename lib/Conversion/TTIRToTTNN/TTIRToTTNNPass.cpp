// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/Debug/IR/Debug.h"
#include "ttmlir/Dialect/Debug/IR/DebugOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {

struct ConvertTTIRToTTNNPass
    : public ttir::impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNNPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<ttnn::TTNNDialect>();
    target.addLegalDialect<quant::QuantDialect>();
    target.addLegalDialect<debug::DebugDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();
    target.addLegalOp<ttcore::DeviceOp>();
    target.addLegalOp<ttcore::OptimizationBarrierOp>();
    target.addLegalOp<ttcore::LoadCachedOp>();

    // Debug dialect ops.
    target.addLegalOp<debug::AnnotateOp>();
    target.addLegalOp<debug::BreakpointOp>();
    target.addLegalOp<debug::PrintOp>();
    target.addLegalOp<debug::MemorySnapshotOp>();
    target.addLegalOp<debug::RegionStartOp>();
    target.addLegalOp<debug::RegionEndOp>();
    target.addIllegalOp<debug::DumpOp>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTNNPatterns(&getContext(), patterns, typeConverter);

    // Apply full conversion
    //
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNPass() {
  return std::make_unique<ConvertTTIRToTTNNPass>();
}

} // namespace mlir::tt
