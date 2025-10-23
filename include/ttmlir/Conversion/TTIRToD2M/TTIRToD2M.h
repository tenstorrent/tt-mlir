// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOD2M_TTIRTOD2M_H
#define TTMLIR_CONVERSION_TTIRTOD2M_TTIRTOD2M_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#define GEN_PASS_DECL_TTIRTOD2M
#include "ttmlir/Conversion/Passes.h.inc"

void populateTTIRToD2MPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    mlir::tt::ttcore::MemorySpace defaultInputMemSpace,
    mlir::tt::ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
    bool collapseTensors);

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToD2MPass();

std::unique_ptr<OperationPass<ModuleOp>>
createTTIRToD2MPass(const TTIRToD2MOptions &options);

inline bool hasMetalLayout(Value v) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(v.getType());
  return tensorType && mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
                           tensorType.getEncoding());
}

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOD2M_TTIRTOD2M_H
