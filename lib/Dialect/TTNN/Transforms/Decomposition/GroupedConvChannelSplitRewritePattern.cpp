// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/GroupedConvChannelSplitRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <string>

namespace mlir::tt::ttnn::decomposition {

namespace {

// Extract [begin, end) along `dim`, keeping every other dimension whole.
mlir::Value sliceAlongDim(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value value, int64_t dim, int64_t begin,
                          int64_t end) {
  auto valueType = mlir::cast<mlir::RankedTensorType>(value.getType());
  llvm::ArrayRef<int64_t> shape = valueType.getShape();

  llvm::SmallVector<mlir::Attribute> begins, ends, steps;
  for (int64_t i = 0; i < valueType.getRank(); ++i) {
    begins.push_back(
        rewriter.getI32IntegerAttr(i == dim ? static_cast<int32_t>(begin) : 0));
    ends.push_back(rewriter.getI32IntegerAttr(
        static_cast<int32_t>(i == dim ? end : shape[i])));
    steps.push_back(rewriter.getI32IntegerAttr(1));
  }

  llvm::SmallVector<int64_t> slicedShape(shape);
  slicedShape[dim] = end - begin;

  return rewriter.create<ttnn::SliceStaticOp>(
      loc, utils::RankedTensorTypeFactory::create(valueType, slicedShape),
      value, rewriter.getArrayAttr(begins), rewriter.getArrayAttr(ends),
      rewriter.getArrayAttr(steps));
}

// Pick how many whole groups belong in each chunk.
//
// Only divisors of `groups` are considered, so every chunk comes out the same
// shape and the resulting convs share a single program cache entry. Among
// those, tile-aligned channel counts are preferred over a merely larger chunk,
// since a chunk boundary off a tile edge makes the slices and the final concat
// pay for unaligned copies. Returns 0 when even one group exceeds the budget.
int64_t chooseGroupsPerChunk(int64_t groups, int64_t inChannelsPerGroup,
                             int64_t outChannelsPerGroup,
                             int64_t maxChannelsPerChunk) {
  constexpr int64_t tileWidth = static_cast<int64_t>(ttnn::TILE_WIDTH);

  int64_t widestPerGroup = std::max(inChannelsPerGroup, outChannelsPerGroup);
  int64_t upperBound = std::min(maxChannelsPerChunk / widestPerGroup, groups);

  int64_t largestDivisor = 0;
  for (int64_t candidate = upperBound; candidate >= 1; --candidate) {
    if (groups % candidate != 0) {
      continue;
    }
    if (largestDivisor == 0) {
      largestDivisor = candidate;
    }
    if ((candidate * inChannelsPerGroup) % tileWidth == 0 &&
        (candidate * outChannelsPerGroup) % tileWidth == 0) {
      return candidate;
    }
  }
  return largestDivisor;
}

} // namespace

template <typename ConvOp>
mlir::LogicalResult
GroupedConvChannelSplitRewritePattern<ConvOp>::matchAndRewrite(
    ConvOp srcOp, mlir::PatternRewriter &rewriter) const {
  int64_t groups = srcOp.getGroups();
  int64_t inChannels = srcOp.getInChannels();
  int64_t outChannels = srcOp.getOutChannels();

  // With groups == 1 every output channel reads every input channel, so there
  // is no channel partition to cut along.
  if (groups <= 1) {
    return mlir::failure();
  }

  if (std::max(inChannels, outChannels) <= kMaxGroupedConvChannels) {
    return mlir::failure();
  }

  if (inChannels % groups != 0 || outChannels % groups != 0) {
    return mlir::failure();
  }

  // The weight and bias offsets below assume the unprepared conv weight layout
  // (O, C/G, ...). PrepareConv2dWeights rewrites it to (1, 1, K*K*C/G, O),
  // where output channels are no longer dimension 0.
  auto weightType =
      mlir::cast<mlir::RankedTensorType>(srcOp.getWeight().getType());
  if (weightType.getDimSize(0) != outChannels) {
    return mlir::failure();
  }

  int64_t inChannelsPerGroup = inChannels / groups;
  int64_t outChannelsPerGroup = outChannels / groups;
  int64_t groupsPerChunk = chooseGroupsPerChunk(
      groups, inChannelsPerGroup, outChannelsPerGroup, kMaxGroupedConvChannels);
  if (groupsPerChunk == 0 || groupsPerChunk == groups) {
    return mlir::failure();
  }

  mlir::RankedTensorType resultType = srcOp.getResult().getType();
  int64_t inputChannelDim =
      mlir::cast<mlir::RankedTensorType>(srcOp.getInput().getType()).getRank() -
      1;
  int64_t resultChannelDim = resultType.getRank() - 1;

  int64_t numChunks = groups / groupsPerChunk;
  int64_t inChannelsPerChunk = groupsPerChunk * inChannelsPerGroup;
  int64_t outChannelsPerChunk = groupsPerChunk * outChannelsPerGroup;

  llvm::SmallVector<int64_t> chunkResultShape(resultType.getShape());
  chunkResultShape[resultChannelDim] = outChannelsPerChunk;
  mlir::RankedTensorType chunkResultType =
      utils::RankedTensorTypeFactory::create(resultType, chunkResultShape);

  llvm::SmallVector<mlir::Value> chunkResults;
  chunkResults.reserve(numChunks);

  for (int64_t chunk = 0; chunk < numChunks; ++chunk) {
    mlir::Location chunkLoc = ttmlir::utils::appendLocationSuffix(
        srcOp.getLoc(), "_group_chunk_" + std::to_string(chunk));

    mlir::Value inputSlice = sliceAlongDim(
        rewriter, chunkLoc, srcOp.getInput(), inputChannelDim,
        chunk * inChannelsPerChunk, (chunk + 1) * inChannelsPerChunk);
    mlir::Value weightSlice = sliceAlongDim(
        rewriter, chunkLoc, srcOp.getWeight(), /*dim=*/0,
        chunk * outChannelsPerChunk, (chunk + 1) * outChannelsPerChunk);

    // Bias is (1, 1, 1, O), so it covers the same output channel range as the
    // weight rows. Sliced before cloning the conv so that every operand
    // dominates its use.
    mlir::Value biasSlice;
    if (srcOp.getBias()) {
      int64_t biasChannelDim =
          mlir::cast<mlir::RankedTensorType>(srcOp.getBias().getType())
              .getRank() -
          1;
      biasSlice = sliceAlongDim(rewriter, chunkLoc, srcOp.getBias(),
                                biasChannelDim, chunk * outChannelsPerChunk,
                                (chunk + 1) * outChannelsPerChunk);
    }

    auto chunkConv = mlir::cast<ConvOp>(rewriter.clone(*srcOp.getOperation()));
    chunkConv->setLoc(chunkLoc);
    chunkConv.getInputMutable().assign(inputSlice);
    chunkConv.getWeightMutable().assign(weightSlice);
    if (biasSlice) {
      chunkConv.getBiasMutable().assign(biasSlice);
    }
    chunkConv.setInChannelsAttr(
        rewriter.getI32IntegerAttr(static_cast<int32_t>(inChannelsPerChunk)));
    chunkConv.setOutChannelsAttr(
        rewriter.getI32IntegerAttr(static_cast<int32_t>(outChannelsPerChunk)));
    chunkConv.setGroupsAttr(
        rewriter.getI32IntegerAttr(static_cast<int32_t>(groupsPerChunk)));
    chunkConv.getResult().setType(chunkResultType);

    chunkResults.push_back(chunkConv.getResult());
  }

  rewriter.replaceOpWithNewOp<ttnn::ConcatOp>(
      srcOp, resultType, chunkResults, static_cast<int32_t>(resultChannelDim));

  return mlir::success();
}

template class GroupedConvChannelSplitRewritePattern<ttnn::Conv1dOp>;
template class GroupedConvChannelSplitRewritePattern<ttnn::Conv2dOp>;

} // namespace mlir::tt::ttnn::decomposition
