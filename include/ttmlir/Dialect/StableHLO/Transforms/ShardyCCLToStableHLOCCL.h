// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYCCLTOSTABLEHLOCCL_H
#define TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYCCLTOSTABLEHLOCCL_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

namespace mlir::tt::stablehlo {

#ifdef TTMLIR_ENABLE_STABLEHLO

class ShardyTypeConverter : public TypeConverter {
public:
  ShardyTypeConverter(MLIRContext *ctx) {
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

void populateShardyCCLToStableHLOCCLPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter);

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYCCLTOSTABLEHLOCCL_H
