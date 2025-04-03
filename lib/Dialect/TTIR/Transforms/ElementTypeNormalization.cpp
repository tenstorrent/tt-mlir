// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/UniformTypeRewriter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_ELEMENTTYPENORMALIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class ElementTypeConverter : public TypeConverter {
public:
  ElementTypeConverter(bool enableFP32) {
    addConversion([](Type type) -> Type { return type; });
    addConversion([enableFP32](RankedTensorType type) -> RankedTensorType {
      Type elementType = type.getElementType();
      size_t bitWidth = type.getElementTypeBitWidth();
      MLIRContext *context = elementType.getContext();

      // Convert bools to bf16, since not all ttnn ops
      // support uint8.
      if (bitWidth == 1) {
        elementType = BFloat16Type::get(context);
      } else if (enableFP32 && isa<FloatType>(elementType) && bitWidth >= 32) {
        elementType = Float32Type::get(context);
      } else {
        elementType =
            dataTypeToElementType(context, elementTypeToDataType(elementType));
      }

      SmallVector<int64_t> shape(type.getShape());
      if (shape.empty()) {
        shape = {1};
      }

      return RankedTensorType::get(shape, elementType);
    });
  }
};

struct ElementTypeNormalization
    : public impl::ElementTypeNormalizationBase<ElementTypeNormalization> {
  using impl::ElementTypeNormalizationBase<
      ElementTypeNormalization>::ElementTypeNormalizationBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    ElementTypeConverter converter(enableFP32);
    patterns.add<UniformTypeRewriter>(converter, &getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttir
