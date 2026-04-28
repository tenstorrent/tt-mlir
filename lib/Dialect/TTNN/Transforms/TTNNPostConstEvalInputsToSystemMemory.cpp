// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNPOSTCONSTEVALINPUTSTOSYSTEMMEMORY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static int64_t getTiledVolume(ArrayRef<int64_t> shape) {
  llvm::SmallVector<int64_t> paddedShape =
      ttnn::utils::getTilePaddedShape(shape);
  return ttmlir::utils::volume(llvm::ArrayRef<int64_t>(paddedShape));
}

static bool isPureTileLayoutChange(RankedTensorType inputType,
                                   RankedTensorType outputType) {
  auto inputLayout = mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  auto outputLayout = mlir::dyn_cast<TTNNLayoutAttr>(outputType.getEncoding());
  if (!inputLayout || !outputLayout) {
    return false;
  }

  return inputLayout.getBufferType() == outputLayout.getBufferType() &&
         inputLayout.getMemLayout() == outputLayout.getMemLayout() &&
         inputLayout.getDataType() == outputLayout.getDataType() &&
         inputLayout.getLayout() != outputLayout.getLayout() &&
         outputLayout.isTiled();
}

static RankedTensorType withLayout(RankedTensorType type, Layout layout) {
  auto typeLayout = mlir::dyn_cast<TTNNLayoutAttr>(type.getEncoding());
  if (!typeLayout) {
    return type;
  }
  return type.cloneWithEncoding(typeLayout.withLayout(layout, type.getShape()));
}

static int64_t getPaddingWasteBytes(RankedTensorType type) {
  auto layout = mlir::dyn_cast<TTNNLayoutAttr>(type.getEncoding());
  if (!layout || !layout.isTiled()) {
    return 0;
  }
  return getTiledVolume(type.getShape()) -
         ttmlir::utils::volume(type.getShape());
}

static Operation *cloneUnaryOpWithNewInputAndType(PatternRewriter &rewriter,
                                                  Operation *op, Value input,
                                                  RankedTensorType resultType) {
  Operation *newOp = rewriter.clone(*op);
  newOp->setOperand(0, input);
  newOp->getResult(0).setType(resultType);
  return newOp;
}

class SinkTileLayoutPattern : public OpRewritePattern<ttnn::ToLayoutOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ToLayoutOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto tiledInputType =
        mlir::cast<RankedTensorType>(op.getResult().getType());
    if (!isPureTileLayoutChange(inputType, tiledInputType) ||
        !op->hasOneUse()) {
      return failure();
    }

    constexpr int64_t minPaddingReduction = 5LL * 1024LL * 1024LL;
    int64_t currentWaste = getPaddingWasteBytes(tiledInputType);
    if (currentWaste < minPaddingReduction) {
      return failure();
    }

    SmallVector<Operation *> chain;
    int64_t bestIndex = -1;
    int64_t bestWaste = currentWaste;

    Operation *cursor = *op->user_begin();
    while (cursor) {
      if (auto meshShard = mlir::dyn_cast<ttnn::MeshShardOp>(cursor)) {
        if (meshShard.getShardType() != ttcore::MeshShardType::Identity) {
          break;
        }
        chain.push_back(cursor);
      } else if (mlir::isa<ttnn::MeshPartitionOp>(cursor)) {
        chain.push_back(cursor);
      } else if (mlir::isa<ttnn::PermuteOp, ttnn::ReshapeOp>(cursor)) {
        chain.push_back(cursor);
        auto resultType =
            mlir::cast<RankedTensorType>(cursor->getResult(0).getType());
        int64_t candidateWaste = getPaddingWasteBytes(resultType);
        if (candidateWaste + minPaddingReduction <= bestWaste) {
          bestIndex = static_cast<int64_t>(chain.size()) - 1;
          bestWaste = candidateWaste;
        }
      } else {
        break;
      }

      if (!cursor->hasOneUse()) {
        break;
      }
      cursor = *cursor->user_begin();
    }

    if (bestIndex < 0) {
      return failure();
    }

    rewriter.setInsertionPoint(op);

    Value currentValue = op.getInput();
    for (int64_t i = 0; i <= bestIndex; ++i) {
      auto resultType =
          mlir::cast<RankedTensorType>(chain[i]->getResult(0).getType());
      auto movedType = withLayout(
          resultType,
          mlir::cast<TTNNLayoutAttr>(inputType.getEncoding()).getLayout());
      currentValue = cloneUnaryOpWithNewInputAndType(rewriter, chain[i],
                                                     currentValue, movedType)
                         ->getResult(0);
    }

    auto candidateType =
        mlir::cast<RankedTensorType>(chain[bestIndex]->getResult(0).getType());
    auto candidateLayout =
        mlir::cast<TTNNLayoutAttr>(candidateType.getEncoding());
    currentValue =
        utils::createToLayoutOp(
            chain[bestIndex],
            mlir::cast<mlir::TypedValue<RankedTensorType>>(currentValue),
            rewriter, candidateLayout.getLayout(),
            candidateLayout.getBufferType(), candidateLayout.getMemLayout(),
            candidateLayout.getDataType(), "_sink_tile")
            .getResult();

    for (int64_t i = bestIndex + 1; i < static_cast<int64_t>(chain.size());
         ++i) {
      auto resultType =
          mlir::cast<RankedTensorType>(chain[i]->getResult(0).getType());
      currentValue = cloneUnaryOpWithNewInputAndType(rewriter, chain[i],
                                                     currentValue, resultType)
                         ->getResult(0);
    }

    rewriter.replaceOp(chain.back(), currentValue);
    for (int64_t i = static_cast<int64_t>(chain.size()) - 2; i >= 0; --i) {
      rewriter.eraseOp(chain[i]);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class TTNNPostConstEvalInputsToSystemMemory
    : public impl::TTNNPostConstEvalInputsToSystemMemoryBase<
          TTNNPostConstEvalInputsToSystemMemory> {
public:
  TTNNPostConstEvalInputsToSystemMemory() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SinkTileLayoutPattern>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
