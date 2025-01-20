// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDATATYPESELECTION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRDataTypeConverter : public TypeConverter {
public:
  TTIRDataTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion(
        [ctx](RankedTensorType type) -> Type {
          auto encoding = mlir::cast<MetalLayoutAttr>(type.getEncoding());
          auto memref = encoding.getMemref();
          if (memref.getElementType()) {
            // auto newMemRef = MemRefType::get(type.getShape(), mlir::FloatType::getF32(ctx), memref.getLayout(), memref.getMemorySpace());
            auto newEncoding = MetalLayoutAttr::get(ctx, type, encoding.getMemorySpace(), encoding.getGrid(), mlir::Float32Type::get(ctx), encoding.getMemLayout());
            return RankedTensorType::get(type.getShape(), type.getElementType(), newEncoding);
          }
          return type;
        });
  }
};

class TTIRDataTypeRewriter : public RewritePattern {
public:
  TTIRDataTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        converter(&converter) {}
 
  template <typename ValueRange>
  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    bool updated = false;
    auto result = converter->convertTypes(valueRange.getTypes(), newTypes);
    if (result.failed()) {
      return false;
    }
    for (auto [operand, newType] : llvm::zip(valueRange, newTypes)) {
      if (operand.getType() == newType) {
        continue;
      }
      operand.setType(newType);
      updated = true;
    }
    return updated;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    
    bool updated = false;
    llvm::outs() << op->getName() << "\n";
    SmallVector<Type> operands;
    SmallVector<Type> results;
    updated |= convertTypes(op->getOperands(), operands);
    updated |= convertTypes(op->getResults(), results);

    return updated ? success() : failure();
  }

  const TypeConverter *converter;
};

class TTIRDataTypeSelection
    : public impl::TTIRDataTypeSelectionBase<TTIRDataTypeSelection> {

  using impl::TTIRDataTypeSelectionBase<
      TTIRDataTypeSelection>::TTIRDataTypeSelectionBase;

  void runOnOperation() final {
    TTIRDataTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRDataTypeRewriter>(typeConverter, &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttir 