// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H
#define TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DECL_TTIRTOTTIRGENERIC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir
// ............................................................................
namespace mlir::tt {

void populateTTIRToTTIRGenericPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttcore::MemorySpace defaultInputMemSpace,
    ttcore::MemorySpace defaultOutputMemSpace,
    const llvm::SmallVector<int64_t> &targetGridShape);

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRGenericPass();
std::unique_ptr<OperationPass<ModuleOp>>
createTTIRToTTIRGenericPass(const ttir::TTIRToTTIRGenericOptions &options);

inline bool hasMetalLayout(Value v) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(v.getType());
  return tensorType && mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
                           tensorType.getEncoding());
}

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_TTIRTOTTIRGENERIC_TTIRTOTTIRGENERIC_H
