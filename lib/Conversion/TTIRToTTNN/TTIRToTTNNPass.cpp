// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "../PassDetail.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

struct ConvertTTIRToTTNNPass
    : public ttnn::ConvertTTIRToTTNNBase<ConvertTTIRToTTNNPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<ttnn::TTNNDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();

    TypeConverter typeConverter;
    // All types map 1:1.
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTNNPatterns(&getContext(), patterns, typeConverter);

    // Full conversion requires explicit handling of FuncOp and ModuleOp, which
    // should be passed down unmodified so partial conversion is used.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
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
