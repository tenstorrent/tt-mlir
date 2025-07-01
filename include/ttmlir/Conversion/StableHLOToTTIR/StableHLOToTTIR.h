// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOTOTTIR_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOTOTTIR_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt {

#ifdef TTMLIR_ENABLE_STABLEHLO
// Shared TypeConverter for StableHLO to TTIR conversions
class StablehloTypeConverter : public TypeConverter {
public:
  StablehloTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) {
      assert(isa<RankedTensorType>(type) &&
             "only ranked tensor type supported");
      return type;
    });

    // Convert scalars to 1D tensors.
    addConversion([&](RankedTensorType type) -> RankedTensorType {
      if (!type.getShape().empty()) {
        return type;
      }

      return RankedTensorType::get(/*shape=*/{1}, type.getElementType(),
                                   type.getEncoding());
    });
  }
};

void populateStableHLOToTTIRPatterns(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass();
#endif

} // namespace mlir::tt

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_STABLEHLOTOTTIR_H
