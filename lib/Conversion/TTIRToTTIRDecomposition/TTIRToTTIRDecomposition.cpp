// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cassert>
#include <numeric>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

//===----------------------------------------------------------------------===//
// IndexOp decomposition
//===----------------------------------------------------------------------===//

// ANCHOR: decomposing_an_op_index_ttir_decompose_pattern
// This transformation adjusts IndexOp attributes so that `begin`, `end`, and
// `step` become arrays, where each array element corresponds to a dimension of
// the input tensor. For dimensions other than the sliced dimension, default
// values are used.
//
namespace {
struct IndexToSliceConversionPattern
    : public OpConversionPattern<ttir::IndexOp> {
  using OpConversionPattern<ttir::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType =
        ::mlir::dyn_cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    if (!inputType || !inputType.hasRank()) {
      return failure();
    }

    int64_t rank = inputType.getRank();
    llvm::SmallVector<mlir::Attribute, 4> begins, ends, steps;

    for (int64_t i = 0; i < rank; ++i) {
      if (i == op.getDim()) {
        begins.push_back(rewriter.getI32IntegerAttr(adaptor.getBegin()));
        ends.push_back(rewriter.getI32IntegerAttr(adaptor.getEnd()));
        steps.push_back(rewriter.getI32IntegerAttr(adaptor.getStep()));
      } else {
        begins.push_back(rewriter.getI32IntegerAttr(0));
        ends.push_back(rewriter.getI32IntegerAttr(inputType.getDimSize(i)));
        steps.push_back(rewriter.getI32IntegerAttr(1));
      }
    }

    auto newOp = rewriter.create<ttir::SliceStaticOp>(
        op.getLoc(), getTypeConverter()->convertType(op.getType()),
        adaptor.getInput(), rewriter.getArrayAttr(begins),
        rewriter.getArrayAttr(ends), rewriter.getArrayAttr(steps));

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};
} // namespace
// ANCHOR_END: decomposing_an_op_index_ttir_decompose_pattern

//===----------------------------------------------------------------------===//
// Reverse Pattern Matching
//===----------------------------------------------------------------------===//

// Decomposing Reverse Op into Gather Op.
// As soon as tenstorrent/tt-metal#16618 is finished, this decomposition can be
// removed.
namespace {
struct ReverseOpConversionPattern
    : public OpConversionPattern<ttir::ReverseOp> {
  using OpConversionPattern<ttir::ReverseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ArrayRef<int64_t> dimensions = adaptor.getDimensions();
    ArrayRef<int64_t> shape = op.getInput().getType().getShape();
    Value currentInput = adaptor.getInput();
    for (int32_t dim : dimensions) {
      SmallVector<int32_t> indices;
      for (int32_t i = shape[dim] - 1; i >= 0; i--) {
        indices.push_back(i);
      }

      auto tensorType =
          RankedTensorType::get({shape[dim]}, rewriter.getI32Type());

      auto denseAttr = DenseIntElementsAttr::get(tensorType, indices);

      Value reversedIndices =
          rewriter.create<ttir::ConstantOp>(op.getLoc(), tensorType, denseAttr);

      SmallVector<int64_t> offsetDims;
      for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); i++) {
        if (i != dim) {
          offsetDims.push_back(i);
        }
      }

      SmallVector<int64_t> sliceSizes(shape.begin(), shape.end());
      sliceSizes[dim] = 1;

      currentInput = rewriter.create<ttir::GatherOp>(
          op.getLoc(), getTypeConverter()->convertType(op.getType()),
          /*input=*/currentInput,
          /*start_indices=*/reversedIndices,
          /*offset_dims=*/offsetDims,
          /*collapsed_slice_dims=*/SmallVector<int64_t>{dim},
          /*operand_batching_dims=*/SmallVector<int64_t>{},
          /*start_indices_batching_dims=*/SmallVector<int64_t>{},
          /*start_index_map=*/SmallVector<int64_t>{dim},
          /*index_vector_dim=*/1,
          /*slice_sizes=*/sliceSizes,
          /*indices_are_sorted=*/false);
    }

    rewriter.replaceOp(op, currentInput);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Gather Pattern Matching
//===----------------------------------------------------------------------===//

namespace {
struct GatherToEmbeddingConversionPattern
    : public OpConversionPattern<ttir::GatherOp> {
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

  /**
   * Validates Gather Op constraints for embedding conversion
   *
   * Enforces constraints on Gather Op to ensure valid embedding
   * transformation:
   * - start indices tensor isn't 1D when we are indexing multiple dims
   * - operandBatchingDims and startIndicesBatchingDims are none
   * - sliceSizes are fullDim for dimensions we are not indexing
   * - for dimensions we are indexing, sliceSizes must fit into one of:
   *   - all sliceSizes are 1
   *   - all sliceSizes are fullDim except one which can be anything
   */

  LogicalResult checkBasicLegality(ttir::GatherOp op,
                                   PatternRewriter &rewriter) const {

    // Get input and start indices tensor shape.
    auto inputShape = op.getInput().getType().getShape();
    auto startIndicesShape = op.getStartIndices().getType().getShape();

    // Get attributes needed for embedding op pattern matching checks.
    auto sliceSizes = op.getSliceSizes();
    auto startIndexMap = op.getStartIndexMap();

    // Check if start indices tensor isn't 1D when we are indexing multiple
    // dimensions because of matmul restrictions.
    if (startIndexMap.size() > 1 && startIndicesShape.size() == 1) {
      return rewriter.notifyMatchFailure(
          op, "Did not satisfy startIndicesShape.size() > 1 when "
              "startIndexMap.size() > 1");
    }

    // Check if there are no batching dims.
    if (!op.getOperandBatchingDims().empty() ||
        !op.getStartIndicesBatchingDims().empty()) {
      return rewriter.notifyMatchFailure(op, "Did not satisfy batching = none");
    }

    // Check slice sizes conditions.
    size_t fullIndexedDims = 0;
    size_t partialIndexedDims = 0;
    size_t singletonIndexedDims = 0;

    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (llvm::is_contained(startIndexMap, i)) {
        if (inputShape[i] == 1) {
          singletonIndexedDims++;
        } else if (sliceSizes[i] == inputShape[i]) {
          fullIndexedDims++;
        } else if (sliceSizes[i] != 1) {
          partialIndexedDims++;
        }
      } else if (sliceSizes[i] != inputShape[i]) {
        return rewriter.notifyMatchFailure(
            op, "Did not satisfy sliceSizes[i] = inputShape[i] for dims not "
                "in startIndexMap");
      }
    }

    size_t remainingIndexedDims =
        startIndexMap.size() - fullIndexedDims - singletonIndexedDims;
    if (partialIndexedDims && (remainingIndexedDims != 1)) {
      return rewriter.notifyMatchFailure(
          op,
          "Did not satisfy slice conditions for dimensions in startIndexMap");
    }

    if (fullIndexedDims &&
        (remainingIndexedDims > 1 ||
         (remainingIndexedDims + singletonIndexedDims) == 0)) {
      return rewriter.notifyMatchFailure(
          op,
          "Did not satisfy slice conditions for dimensions in startIndexMap");
    }

    return success();
  }

  /**
   * Lowers Gather Op into Embedding Op (and applies Reshape and Permute Ops, if
   * necessary)
   *
   * There is no TTNN Gather support.
   * Gather Op is lowered into Embedding Op Op.
   * Torch embeddings are lowered into Gather Op.
   * Most models use Gather Op to implement simple embeddings.
   * If encountered more complicated Gather Op implementations, they can be
   * lowered into slice/ concat/ etc.
   *
   * Embedding Op expects:
   * - weights to be strictly 2D. We index the first dimension of weights, and
   * take slices from the full second dimension.
   * - input can be 1D or 2D
   * - output shape is the shape of input with the last dimension of the
   * weights appended
   *
   *  - Gather Op input becomes Embedding Op weights. Because it can have
   * any number and order of dimensions, it is permuted and reshaped
   * (flattened).
   *  - Gather Op startIndices becomes Embedding Op input. Because it can
   * have any number and order of dimensions, it is permuted and reshaped
   * (flattened).
   * - Embedding Op output needs to be reshaped to recover lost
   * dimensions and permuted as Gather Op output dimensions can be in any
   * order.
   */

  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // GatherOp can be used to implement embedding lookup, check for that case.
    LogicalResult err = checkBasicLegality(op, rewriter);
    if (not err.succeeded()) {
      return err;
    }

    auto inputShape = op.getInput().getType().getShape();
    auto sliceSizes = op.getSliceSizes();
    auto originalStartIndexMap = op.getStartIndexMap();

    // If there are indexed dims that have full slice size, we need to ignore
    // them and slice indices accordingly, which is why we note the
    // actualIndexedDim.
    int64_t actualIndexedDim = -1;

    // If there is an indexed dim with slice size > 1, but not full, we need to
    // expand start indices to contain the implied ones.
    bool needsExpansion = false;

    // Create startIndexMap without dims for which sliceSizes[dim] =
    // inputShape[dim]. If there are dims for which sliceSizes[dim] =
    // inputShape[dim] = 1, they are treated specially:
    // - if there is a partially indexed dim, they are removed
    // - if all other indexed dims are full, one of them is kept
    size_t fullIndexedDims = 0;
    bool partialIndexedDimExists = false;
    for (size_t i = 0; i < originalStartIndexMap.size(); ++i) {
      int64_t dim = originalStartIndexMap[i];
      if (sliceSizes[dim] == inputShape[dim]) {
        fullIndexedDims++;
      } else if (sliceSizes[dim] != 1) {
        partialIndexedDimExists = true;
      }
    }

    llvm::SmallVector<int64_t> startIndexMap;
    for (size_t i = 0; i < originalStartIndexMap.size(); ++i) {
      int64_t dim = originalStartIndexMap[i];
      if (inputShape[dim] == 1) {
        if (fullIndexedDims == originalStartIndexMap.size()) {
          startIndexMap.push_back(dim);
          actualIndexedDim = i;
          break;
        }
        if (partialIndexedDimExists || fullIndexedDims > 0) {
          continue;
        }
      } else if (sliceSizes[dim] == inputShape[dim]) {
        continue;
      }

      startIndexMap.push_back(dim);
      actualIndexedDim = i;
      if (sliceSizes[dim] != 1) {
        needsExpansion = true;
      }
    }
    auto numIndexingDims = startIndexMap.size();

    auto inputPermuted = permuteInput(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_permuteInput"),
        op.getInput(), startIndexMap);
    auto input = reshapeInput(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_reshapeInput"),
        inputPermuted, numIndexingDims);

    // If we are indexing multiple dims, we need to transform indices for the
    // new single (flattened) indexing dim. If the extra indexed dims are full,
    // we need to slice indices.
    auto startIndices = op.getStartIndices();
    if (numIndexingDims > 1) {
      op->emitWarning("End results might be incorrect when indexing multiple "
                      "dimensions of input because of typecast ops.");
      startIndices =
          flattenStartIndices(rewriter, inputPermuted.getType().getShape(), op);
    } else if (originalStartIndexMap.size() != numIndexingDims) {
      startIndices = sliceStartIndices(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op->getLoc(),
                                              "_sliceStartIndices"),
          op.getStartIndices(), op.getIndexVectorDim(), actualIndexedDim);
    }

    if (startIndices.getType().getShape().size() != 2 ||
        (needsExpansion && op.getIndexVectorDim() != 0)) {
      startIndices =
          reshapeStartIndices(rewriter,
                              ttmlir::utils::appendLocationSuffix(
                                  op->getLoc(), "_reshapeStartIndices"),
                              startIndices);
    }

    // If we are indexing a dim with slice size > 1, we need to expand indices
    // to gather all the rows, not just the first one.
    if (needsExpansion) {
      startIndices = expandStartIndices(
          rewriter,
          ttmlir::utils::appendLocationSuffix(op->getLoc(),
                                              "_expandStartIndices"),
          startIndices, op.getIndexVectorDim(),
          sliceSizes[originalStartIndexMap[actualIndexedDim]]);
    }

    // Calculate a new shape for output: this is new start indices shape + last
    // dim of input shape.
    auto startIndicesShape = startIndices.getType().getShape();
    llvm::SmallVector<int64_t> newOutputShape(startIndicesShape.begin(),
                                              startIndicesShape.end());
    newOutputShape.push_back(input.getType().getShape()[1]);

    auto embeddingOutputType = mlir::RankedTensorType::get(
        newOutputShape, input.getType().getElementType(),
        input.getType().getEncoding());
    ttir::EmbeddingOp embeddingOp = rewriter.create<ttir::EmbeddingOp>(
        op.getLoc(), embeddingOutputType, startIndices, input);

    rewriter.replaceOp(op, reshapeAndPermuteOutput(rewriter, embeddingOp,
                                                   startIndexMap[0], op));
    return success();
  }

private:
  // In StableHLO, startIndexMap attribute refers to which dims of input
  // we are indexing (with startIndices). We need these dims to be
  // flattened together to be the first dim of transformed input (that is
  // weights for ttir.embedding). This helper makes these indexing dims
  // the first few dims of input.
  // Example: inputShape = [2, 3, 4, 5], startIndexMap = [1, 3] ->
  // permutedInputShape = [3, 5, 2, 4]
  static ttir::PermuteOp
  permuteInput(ConversionPatternRewriter &rewriter, Location loc,
               ::mlir::TypedValue<::mlir::RankedTensorType> input,
               ::llvm::ArrayRef<int64_t> startIndexMap) {
    auto inputType = input.getType();
    llvm::SmallVector<int64_t> inputPermutation(startIndexMap);
    inputPermutation.append(llvm::filter_to_vector(
        llvm::seq<int64_t>(inputType.getRank()), [&startIndexMap](int64_t idx) {
          return !llvm::is_contained(startIndexMap, idx);
        }));
    auto permutedInputShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), inputPermutation);
    return rewriter.create<ttir::PermuteOp>(
        loc,
        RankedTensorType::get(permutedInputShape, inputType.getElementType(),
                              inputType.getEncoding()),
        input, inputPermutation);
  }

  // This helper flattens the indexing dims to be one, first dim of
  // transformed input, and all the other dims to be the second dim.
  // Example: permutedInputShape = [3, 5, 2, 4], numIndexingDims = 2 ->
  // newIputShape = [15, 8]
  static ttir::ReshapeOp
  reshapeInput(ConversionPatternRewriter &rewriter, Location loc,
               ::mlir::TypedValue<::mlir::RankedTensorType> input,
               size_t numIndexingDims) {
    auto inputShape = input.getType().getShape();
    assert(
        numIndexingDims <= inputShape.size() &&
        "Number of indexing dims can't be greater than number of input dims");
    llvm::SmallVector<int64_t> newInputShape{
        std::accumulate(inputShape.begin(),
                        inputShape.begin() + numIndexingDims, int64_t{1},
                        std::multiplies<>()),
        std::accumulate(inputShape.begin() + numIndexingDims, inputShape.end(),
                        int64_t{1}, std::multiplies<>())};
    return createReshapeOp(rewriter, loc, input, newInputShape);
  }

  // If we are indexing multiple dimes of input, we need to adjust start
  // indices to represent indices that index one flattened dimension.
  // - indexVectorDim represents in what dimension are indices, so first we
  // permute to make sure it is the last dimension
  // - matmul doesn't work with integers (which startIndices are when lowered
  // form SHLO), so a typecast is added
  // - then we add matmul to transform the indices
  // Example: indexingDimsSizes = [3, 5], startIndices[...] = (i, j) ->
  // startIndices[...] = 5 * i + j (because reshaped indexingDimSize is 15)
  static ttir::MatmulOp
  flattenStartIndices(ConversionPatternRewriter &rewriter,
                      ::llvm::ArrayRef<int64_t> inputShape, ttir::GatherOp op) {
    auto startIndices = op.getStartIndices();
    auto startIndicesType = startIndices.getType();
    auto numIndexingDims = op.getStartIndexMap().size();
    auto indexVectorDim = op.getIndexVectorDim();

    llvm::SmallVector<int64_t> startIndicesPermutation = llvm::filter_to_vector(
        llvm::seq<int64_t>(startIndicesType.getRank()),
        [&indexVectorDim](int64_t idx) { return idx != indexVectorDim; });
    startIndicesPermutation.push_back(indexVectorDim);

    auto permutedStartIndicesShape = ttmlir::utils::applyPermutation(
        startIndicesType.getShape(), startIndicesPermutation);
    auto startIndicesPermuted =
        rewriter
            .create<ttir::PermuteOp>(
                ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                                    "_permuteStartIndices"),
                RankedTensorType::get(permutedStartIndicesShape,
                                      startIndicesType.getElementType(),
                                      startIndicesType.getEncoding()),
                startIndices, startIndicesPermutation)
            .getResult();

    // Typecast op because matmul needs float operands.
    auto typecastResultType = startIndicesPermuted.getType().clone(
        mlir::Float32Type::get(op.getContext()));
    ttir::TypecastOp typecastOp = rewriter.create<ttir::TypecastOp>(
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_typecast"),
        typecastResultType, startIndicesPermuted);

    // Const op with correct strides to matmul indices with.
    llvm::SmallVector<float> strides(numIndexingDims);
    int dimensionOffset = 1;
    for (int i = numIndexingDims - 1; i >= 0; i--) {
      strides[i] = dimensionOffset;
      dimensionOffset *= inputShape[i];
    }
    auto tensorType =
        mlir::RankedTensorType::get({static_cast<long>(numIndexingDims), 1},
                                    mlir::Float32Type::get(op.getContext()));
    auto denseAttr =
        mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(strides));
    ttir::ConstantOp constantOp = rewriter.create<ttir::ConstantOp>(
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_constant"),
        tensorType, denseAttr);

    // Return matmul op that transforms indices.
    llvm::SmallVector<int64_t> matmulResultShape = permutedStartIndicesShape;
    matmulResultShape[matmulResultShape.size() - 1] = 1;
    auto matmulResultType = mlir::RankedTensorType::get(
        matmulResultShape, Float32Type::get(op.getContext()));

    return rewriter.create<ttir::MatmulOp>(op.getLoc(), matmulResultType,
                                           typecastOp.getResult(), constantOp);
  }

  // If startIndicesShape[indexVectorDim] > 1, but we are actually slicing only
  // one dim and gathering the other dims fully, we need to slice startIndices
  // to keep only the relevant indices. Example: inputShape = [3, 5],
  // startIndexMap = [0, 1], sliceSizes = [1, 5], startIndices = [[2, 1], [0,
  // 3]], indexVectorDim=1 -> startIndices = [[2], [0]]
  static ttir::SliceStaticOp
  sliceStartIndices(ConversionPatternRewriter &rewriter, Location loc,
                    ::mlir::TypedValue<::mlir::RankedTensorType> startIndices,
                    int64_t indexVectorDim, int64_t actualIndexedDim) {
    auto startIndicesType = startIndices.getType();
    auto startIndicesShape = startIndicesType.getShape();
    int64_t rank = startIndicesType.getRank();

    // Create begins, ends, and steps arrays for slicing
    llvm::SmallVector<int32_t> begins(rank, 0);
    llvm::SmallVector<int32_t> ends(startIndicesShape.begin(),
                                    startIndicesShape.end());
    llvm::SmallVector<int32_t> steps(rank, 1);

    begins[indexVectorDim] = actualIndexedDim;
    ends[indexVectorDim] = actualIndexedDim + 1;

    // Calculate the result shape
    llvm::SmallVector<int64_t> resultShape(startIndicesShape);
    resultShape[indexVectorDim] = 1;

    return rewriter.create<ttir::SliceStaticOp>(
        loc,
        RankedTensorType::get(resultShape, startIndicesType.getElementType(),
                              startIndicesType.getEncoding()),
        startIndices, rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));
  }

  // Helper that reshapes start indices to reduce number of dims, as Embedding
  // Op input needs to be 2D.
  static ttir::ReshapeOp reshapeStartIndices(
      ConversionPatternRewriter &rewriter, Location loc,
      ::mlir::TypedValue<::mlir::RankedTensorType> startIndices) {
    auto startIndicesShape = startIndices.getType().getShape();
    llvm::SmallVector<int64_t, 2> newStartIndicesShape{
        1, std::accumulate(startIndicesShape.begin(), startIndicesShape.end(),
                           int64_t{1}, std::multiplies<>())};
    return createReshapeOp(rewriter, loc, startIndices, newStartIndicesShape);
  }

  // Helper that expands start indices along the index vector dimension when
  // sliceSizes[actualIndexedDim] > 1. This creates additional indices by
  // adding consecutive values to the original indices. Because of earlier
  // reshape, we know startIndices has shape [1, N]. Example: startIndices =
  // [[2, 1]], indexVectorDim=0, sliceSize=3 -> startIndices =
  // [[2, 1], [3, 2], [4, 3]]
  static ttir::AddOp expandStartIndices(ConversionPatternRewriter &rewriter,
                                        Location loc, Value startIndices,
                                        int64_t indexVectorDim,
                                        int64_t sliceSize) {
    auto startIndicesType =
        mlir::cast<RankedTensorType>(startIndices.getType());
    auto startIndicesShape = startIndicesType.getShape();

    // Create NxM matrix where each column is [0, 1, 2, ..., N-1].
    int32_t N = sliceSize;
    int32_t M = startIndicesShape[1];

    llvm::SmallVector<int32_t> matrixData(N * M);
    for (int32_t col = 0; col < M; ++col) {
      for (int32_t row = 0; row < N; ++row) {
        matrixData[row * M + col] = row;
      }
    }
    llvm::SmallVector<int64_t> expandedShape(startIndicesShape);
    expandedShape[0] = sliceSize;

    auto expandedType = mlir::RankedTensorType::get(
        expandedShape, startIndicesType.getElementType(),
        startIndicesType.getEncoding());
    auto offsetAttr =
        mlir::DenseElementsAttr::get(expandedType, llvm::ArrayRef(matrixData));
    auto offsetConstant = rewriter.create<ttir::ConstantOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_offsetConstant"),
        expandedType, offsetAttr);
    // Create broadcast dimensions - all dimensions map directly except the
    // expanded one.
    llvm::SmallVector<int64_t> broadcastDimensions = {sliceSize, 1};

    // Broadcast the original startIndices to the expanded shape.
    auto broadcastedStartIndices = rewriter.create<ttir::BroadcastOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_broadcastStartIndices"),
        expandedType, startIndices,
        rewriter.getDenseI64ArrayAttr(broadcastDimensions));

    // Add the broadcasted tensors to get the final expanded indices.
    return rewriter.create<ttir::AddOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_expandedStartIndices"),
        expandedType, broadcastedStartIndices, offsetConstant);
  }

  // In output, dims other than offsetDims map to startIndices shape, and
  // offsetDims map to input slices. After ttir.embedding all offseDims are
  // flattened to the last dim of output. First we reshape that output to
  // recover lost dims, then we permute them so offset dims are where the
  // attribute states.
  // Example: expectedOutputShape = [2, 3, 4, 5], offsetDims = [1, 3]
  // -> embeddingOutputShape = [2, 4, 15] -reshape-> [2, 4, 3, 5] -permute-> [2,
  // 3, 4, 5]
  static ttir::PermuteOp
  reshapeAndPermuteOutput(ConversionPatternRewriter &rewriter,
                          ::mlir::TypedValue<::mlir::RankedTensorType> output,
                          int64_t indexedDim, ttir::GatherOp op) {
    auto expectedOutputType = op.getType();
    auto expectedOutputShape = expectedOutputType.getShape();
    auto offsetDims = op.getOffsetDims();
    auto collapsedSliceDims = op.getCollapsedSliceDims();
    // Because of permuting input to put the indexing dims first, the output has
    // corresponding dims in front of (other) offsetDims, as well. When size of
    // these dims in output is not 1, we need to move them to their correct
    // spot.
    bool needsOffsetReordering = false;
    size_t numSmallerCollapsedDims = 0;

    if (op.getSliceSizes()[indexedDim] != 1) {
      needsOffsetReordering = true;
      numSmallerCollapsedDims =
          std::lower_bound(collapsedSliceDims.begin(), collapsedSliceDims.end(),
                           indexedDim) -
          collapsedSliceDims.begin();
    }

    llvm::SmallVector<int64_t> outputPermutation;
    if (needsOffsetReordering) {
      outputPermutation.push_back(
          offsetDims[indexedDim - numSmallerCollapsedDims]);
    }
    for (size_t i = 0; i < expectedOutputShape.size(); ++i) {
      if (!llvm::is_contained(offsetDims, i)) {
        outputPermutation.push_back(i);
      }
    }
    for (size_t i = 0; i < offsetDims.size(); ++i) {
      if (!(needsOffsetReordering &&
            i == static_cast<size_t>(indexedDim - numSmallerCollapsedDims))) {
        outputPermutation.push_back(offsetDims[i]);
      }
    }

    auto inverseOutputPermutation =
        ttmlir::utils::inversePermutation(outputPermutation);
    auto permutedOutputShape =
        ttmlir::utils::applyPermutation(expectedOutputShape, outputPermutation);

    auto reshapedOutput = createReshapeOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_reshapeOutput"),
        output, permutedOutputShape);

    return rewriter.create<ttir::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(op->getLoc(), "_permuteOutput"),
        RankedTensorType::get(expectedOutputShape,
                              expectedOutputType.getElementType(),
                              expectedOutputType.getEncoding()),
        reshapedOutput, inverseOutputPermutation);
  }

  static ttir::ReshapeOp
  createReshapeOp(PatternRewriter &rewriter, Location loc, Value input,
                  ::llvm::ArrayRef<int64_t> targetShape) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto shapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(targetShape));

    return rewriter.create<ttir::ReshapeOp>(
        loc, inputType.cloneWith(targetShape, inputType.getElementType()),
        input, shapeAttr);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Embedding to Gather Pattern
//===----------------------------------------------------------------------===//

namespace {
// Converts ttir.embedding to ttir.gather for CPU fallback path.
// This allows us not to use tensor.gather which lacks bufferization support in
// upstream MLIR.
struct EmbeddingToGatherConversionPattern
    : public OpConversionPattern<ttir::EmbeddingOp> {
  using OpConversionPattern<ttir::EmbeddingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    // Cast indices to integer type if needed (gather requires integer indices).
    if (!inputType.getElementType().isIntOrIndex()) {
      auto newInputType =
          RankedTensorType::get(inputType.getShape(), rewriter.getI64Type());
      input =
          rewriter.create<ttir::TypecastOp>(op.getLoc(), newInputType, input);
      inputType = mlir::cast<RankedTensorType>(input.getType());
    }

    auto indicesShape = inputType.getShape();
    int64_t indicesRank = indicesShape.size();
    auto weightShape = weightType.getShape();
    int64_t weightRank = weightType.getRank();

    // Weight is "effectively 2D" with shape (1, 1, ..., 1, vocab_size,
    // embedding_dim). The embedding dimension is always the last dimension.
    int64_t embeddingDim = weightShape[weightRank - 1];
    int64_t vocabDimIndex = weightRank - 2;

    // Add trailing dimension of size 1 to indices for index_vector_dim.
    SmallVector<int64_t> newIndicesShape(indicesShape.begin(),
                                         indicesShape.end());
    newIndicesShape.push_back(1);
    auto reshapedIndicesType =
        RankedTensorType::get(newIndicesShape, inputType.getElementType());

    SmallVector<int32_t> newIndicesShapeI32(newIndicesShape.begin(),
                                            newIndicesShape.end());
    Value reshapedIndices = rewriter.create<ttir::ReshapeOp>(
        op.getLoc(), reshapedIndicesType, input,
        rewriter.getI32ArrayAttr(newIndicesShapeI32));

    // Build gather attributes:
    // - offset_dims: the embedding dimension appears at position indicesRank
    // - collapsed_slice_dims: all dimensions except the last (embedding dim)
    // - start_index_map: points to the vocab dimension (second-to-last)
    // - index_vector_dim: indicesRank (the trailing singleton dimension)
    // - slice_sizes: [1, 1, ..., 1, embedding_dim] matching weight rank
    SmallVector<int64_t> offsetDims{indicesRank};
    SmallVector<int64_t> collapsedSliceDims;
    for (int64_t i = 0; i < weightRank - 1; ++i) {
      collapsedSliceDims.push_back(i);
    }
    SmallVector<int64_t> operandBatchingDims{};
    SmallVector<int64_t> startIndicesBatchingDims{};
    SmallVector<int64_t> startIndexMap{vocabDimIndex};
    int64_t indexVectorDim = indicesRank;
    SmallVector<int64_t> sliceSizes(weightRank, 1);
    sliceSizes[weightRank - 1] = embeddingDim;

    auto gatherOp = rewriter.create<ttir::GatherOp>(
        op.getLoc(), resultType,
        /*input=*/weight,
        /*start_indices=*/reshapedIndices,
        /*offset_dims=*/offsetDims,
        /*collapsed_slice_dims=*/collapsedSliceDims,
        /*operand_batching_dims=*/operandBatchingDims,
        /*start_indices_batching_dims=*/startIndicesBatchingDims,
        /*start_index_map=*/startIndexMap,
        /*index_vector_dim=*/indexVectorDim,
        /*slice_sizes=*/sliceSizes,
        /*indices_are_sorted=*/false);

    rewriter.replaceOp(op, gatherOp);
    return success();
  }
};
} // namespace

namespace {

// Pattern detection - Analyze gather indices to detect replicate padding:
// - Verify single dimension gather
// - Look for pattern like [0, 0, 0, 1, 2, 3] (leading repeated zeros = front
// padding)
// - Or [0, 1, 2, 3, 3, 3] (trailing repeated max = back padding)
// Lower to slice+repeat+concat:
// Example:
// Input: tensor<1x768x4x60x106xbf16>
// index_vector_dim = 2
// Indices: [0, 0, 0, 1, 2, 3] (detected: frontPad=2, backPad=0, dim=2)
// Slice first frame: tensor<1x768x1x60x106xbf16> (dim=2, start=0, end=1)
// Repeat 2x: tensor<1x768x2x60x106xbf16>
// Concat: tensor<1x768x2x60x106xbf16> + tensor<1x768x4x60x106xbf16> ->
// tensor<1x768x6x60x106xbf16>
// This is done to avoid OOM issues when padding is large.
struct GatherToSliceRepeatConcatConversionPattern
    : public OpConversionPattern<ttir::GatherOp> {
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

  // Benefit is 2 because this pattern is more specific than the
  // GatherToEmbeddingConversionPattern.
  GatherToSliceRepeatConcatConversionPattern(TypeConverter &typeConverter,
                                             MLIRContext *context)
      : OpConversionPattern<ttir::GatherOp>(typeConverter, context,
                                            /*benefit=*/2) {}
  LogicalResult checkBasicLegality(ttir::GatherOp op,
                                   PatternRewriter &rewriter) const {
    if (op.getStartIndexMap().size() != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "Start index map must be of size 1");
    }
    auto indices = op.getStartIndices().getDefiningOp<ttir::ConstantOp>();
    if (!indices) {
      return rewriter.notifyMatchFailure(op,
                                         "Start indices must be a constant op");
    }
    return success();
  }
  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult err = checkBasicLegality(op, rewriter);
    if (not err.succeeded()) {
      return err;
    }

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto inputShape = inputType.getShape();
    int64_t indexedDim = op.getStartIndexMap()[0];
    int64_t maxIndex = inputShape[indexedDim] - op.getSliceSizes()[indexedDim];
    int64_t sliceSize = op.getSliceSizes()[indexedDim];
    int32_t starts = 0, ends = 0, lastIndex = 0;
    // It is expected that the indices are consecutive and in ascending order.
    // Like [0,0,0,1,2,3,4,5,5,5,5,5].
    // Number of starts and number of ends are calculated by counting the number
    // of consecutive zeros and max index. In the end, starts and ends are
    // decremented by 1 because they appear once in the original array. In the
    // example above: [0,0,0,1,2,3,4,5,5,5,5,5] = [0,0] + [0,1,2,3,4,5] +
    // [5,5,5,5]

    auto slicedIndices =
        op.getStartIndices().getDefiningOp<ttir::ConstantOp>().getValue();
    for (auto index : slicedIndices.getValues<llvm::APInt>()) {
      if (index == 0) {
        starts++;
      }
      if (index == maxIndex) {
        ends++;
      }
      if (!((index - lastIndex == 1) || (index == lastIndex && index == 0) ||
            (index == lastIndex && index == maxIndex))) {
        return rewriter.notifyMatchFailure(op,
                                           "Indices are not in valid order");
      }
      lastIndex = index.getSExtValue();
    }
    if (lastIndex != maxIndex) {
      return rewriter.notifyMatchFailure(
          op, "Not all indices are present in the original array");
    }

    starts--;
    ends--;

    SmallVector<Value> slicesToConcat;

    slicesToConcat = createSlices(starts, indexedDim, sliceSize, inputShape,
                                  inputType, rewriter, op);

    slicesToConcat.push_back(op.getInput());

    slicesToConcat.append(createSlices(ends, indexedDim, sliceSize, inputShape,
                                       inputType, rewriter, op));

    Value result = rewriter.create<ttir::ConcatOp>(
        op.getLoc(), op.getType(), slicesToConcat,
        rewriter.getSI32IntegerAttr(static_cast<int32_t>(indexedDim)));

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  std::tuple<SmallVector<int32_t>, SmallVector<int32_t>, SmallVector<int32_t>>
  buildSliceArrays(int32_t start, int32_t end, int64_t indexedDim,
                   ArrayRef<int64_t> inputShape) const {
    SmallVector<int32_t> beginsArr(inputShape.size(), 0);
    SmallVector<int32_t> endsArr(inputShape);
    SmallVector<int32_t> stepArr(inputShape.size(), 1);
    beginsArr[indexedDim] = start;
    endsArr[indexedDim] = end;
    return {beginsArr, endsArr, stepArr};
  }

  SmallVector<int64_t>
  computeSliceResultShape(int64_t start, int64_t end, int64_t indexedDim,
                          ArrayRef<int64_t> inputShape) const {
    SmallVector<int64_t> resultShape(inputShape);
    resultShape[indexedDim] = end - start;
    return resultShape;
  }

  SmallVector<Value>
  createSlices(int32_t numberOfRepeats, int32_t indexedDim, int32_t sliceSize,
               ArrayRef<int64_t> inputShape, RankedTensorType inputType,
               ConversionPatternRewriter &rewriter, ttir::GatherOp op) const {
    SmallVector<Value> slices;
    if (numberOfRepeats > 0) {
      auto [begins, endsArr, step] =
          buildSliceArrays(0, sliceSize, indexedDim, inputShape);
      auto sliceShape =
          computeSliceResultShape(0, sliceSize, indexedDim, inputShape);
      auto sliceType = RankedTensorType::get(
          sliceShape, inputType.getElementType(), inputType.getEncoding());

      Value slice = rewriter.create<ttir::SliceStaticOp>(
          op.getLoc(), sliceType, op.getInput(),
          rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(endsArr),
          rewriter.getI32ArrayAttr(step));

      SmallVector<int64_t> repeatShape(sliceShape);
      repeatShape[indexedDim] = numberOfRepeats;
      auto repeatType = RankedTensorType::get(
          repeatShape, inputType.getElementType(), inputType.getEncoding());
      SmallVector<int64_t> repeatDims(inputShape.size(), 1);
      repeatDims[indexedDim] = numberOfRepeats;

      slice = rewriter.create<ttir::RepeatOp>(
          op.getLoc(), repeatType, slice,
          rewriter.getDenseI64ArrayAttr(repeatDims));

      slices.push_back(slice);
    }
    return slices;
  }
};
} // namespace
//===----------------------------------------------------------------------===//
/*
Below is the implementation of the DotGeneralOp decomposition into MatmulOp,
ReshapeOp, and PermuteOp. The DotGeneralOp is a more general form of MatmulOp
where tensors can have arbitrary contract dimensions. Contract dimensions are
the ones along which multiplication happens (typically summed over during the
operation). Previously, DotGeneralOp only supported cases where it directly
mapped to a MatmulOp, which typically involves batch dimensions (e.g., [5, 6, 7]
x [5, 7, 6] where 5 is the batch dimension and multiplication happens along
dimension 7). This decomposition extends the support to more flexible tensor
shapes, such as [5, 6, 7] x [5, 6, 7], where the contract dimension is 6 (or 7)
in both tensors. This allows DotGeneralOp to handle cases beyond the typical
MatmulOp constraints, enabling more complex tensor operations.
*/

namespace {
struct DotGeneralToMatmulConversionPattern
    : public OpConversionPattern<ttir::DotGeneralOp> {
  using OpConversionPattern<ttir::DotGeneralOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value lhs = adaptor.getLhs();
    auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
    int64_t lhsRank = lhsType.getRank();
    SmallVector<int64_t> lhsBatchDims(op.getBatchDimsLhs());
    SmallVector<int64_t> lhsContractDims(op.getContractDimsLhs());

    Value rhs = adaptor.getRhs();
    auto rhsType = mlir::cast<RankedTensorType>(rhs.getType());
    int64_t rhsRank = rhsType.getRank();
    SmallVector<int64_t> rhsBatchDims(op.getBatchDimsRhs());
    SmallVector<int64_t> rhsContractDims(op.getContractDimsRhs());

    Type elementType = op.getType().getElementType();

    SmallVector<int64_t> lhsResultDims =
        getResultDims(lhsBatchDims, lhsContractDims, lhsRank);
    SmallVector<int64_t> rhsResultDims =
        getResultDims(rhsBatchDims, rhsContractDims, rhsRank);

    // Compute permutation for lhs and rhs to get the desired layout.
    // For lhs: (batch dims, result dims, contract dims)
    // For rhs: (batch dims, contract dims, result dims)

    SmallVector<int64_t> lhsPermutation =
        getPermutation(lhsBatchDims, lhsResultDims, lhsContractDims);
    SmallVector<int64_t> rhsPermutation =
        getPermutation(rhsBatchDims, rhsContractDims, rhsResultDims);

    // Apply these permutations to lhs and rhs.

    ttir::PermuteOp lhsPermute = createPermuteOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_permuteLhs"), lhs,
        lhsType, lhsPermutation);
    ttir::PermuteOp rhsPermute = createPermuteOp(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_permuteRhs"), rhs,
        rhsType, rhsPermutation);

    // Compute final shape for lhs and rhs.
    // for lhs (batch dims, prod(result dims), prod(contract dims))
    // for rhs (batch dims, prod(contract dims), prod(result dims))

    SmallVector<int64_t> lhsMatmulInputShape = computeMatmulInputShape(
        rewriter, lhsType, lhsBatchDims, lhsResultDims, lhsContractDims);
    SmallVector<int64_t> rhsMatmulInputShape = computeMatmulInputShape(
        rewriter, rhsType, rhsBatchDims, rhsContractDims, rhsResultDims);

    // Apply this reshape to lhs and rhs to adapt to matmul op.
    // For lhs: (batch dims, prod(result dims), prod(contract dims))
    // For rhs: (batch dims, prod(contract dims), prod(result dims))

    ttir::ReshapeOp lhsMatmulInput = createMatmulFinal(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeLhs"),
        lhsPermute, lhsType, lhsMatmulInputShape);
    ttir::ReshapeOp rhsMatmulInput = createMatmulFinal(
        rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeRhs"),
        rhsPermute, rhsType, rhsMatmulInputShape);
    // Get shape of matmul op result.

    SmallVector<int64_t> matmulDestinationShape;
    for (auto dim : lhsBatchDims) {
      matmulDestinationShape.push_back(lhsType.getShape()[dim]);
    }
    matmulDestinationShape.push_back(
        computeProductOfDims(lhsType.getShape(), lhsResultDims));
    matmulDestinationShape.push_back(
        computeProductOfDims(rhsType.getShape(), rhsResultDims));

    // Perform matmul operation.
    auto matmulOp = rewriter.create<ttir::MatmulOp>(
        op.getLoc(), RankedTensorType::get(matmulDestinationShape, elementType),
        lhsMatmulInput, rhsMatmulInput);

    // Reshape the result by unrolling the prod(lhsResultDims) to original
    // lhsResultDims and likewise for rhsResultDims.

    SmallVector<int64_t> resultShape;
    for (auto dim : lhsBatchDims) {
      resultShape.push_back(lhsType.getShape()[dim]);
    }
    for (auto dim : lhsResultDims) {
      resultShape.push_back(lhsType.getShape()[dim]);
    }
    for (auto dim : rhsResultDims) {
      resultShape.push_back(rhsType.getShape()[dim]);
    }

    llvm::SmallVector<int32_t> finalShapeI32(resultShape.begin(),
                                             resultShape.end());

    auto reshapeOutput = rewriter.replaceOpWithNewOp<ttir::ReshapeOp>(
        op, RankedTensorType::get(resultShape, elementType), matmulOp,
        rewriter.getI32ArrayAttr(finalShapeI32));

    reshapeOutput->setLoc(ttmlir::utils::appendLocationSuffix(
        reshapeOutput->getLoc(), "_reshapeOutput"));

    return success();
  }

private:
  SmallVector<int64_t> getResultDims(const SmallVector<int64_t> &batchDims,
                                     const SmallVector<int64_t> &contractDims,
                                     int64_t rank) const {

    SmallVector<int64_t> allDims;
    for (int64_t i = 0; i < rank; i++) {
      allDims.push_back(i);
    }

    // Remove batch and contract dims.

    for (size_t i = 0; i < batchDims.size(); i++) {
      for (size_t j = 0; j < allDims.size(); j++) {
        if (allDims[j] == batchDims[i]) {
          allDims.erase(allDims.begin() + j);
          break;
        }
      }
    }
    for (size_t i = 0; i < contractDims.size(); i++) {
      for (size_t j = 0; j < allDims.size(); j++) {
        if (allDims[j] == contractDims[i]) {
          allDims.erase(allDims.begin() + j);
          break;
        }
      }
    }

    return allDims;
  }

  SmallVector<int64_t> getPermutation(const SmallVector<int64_t> &batchDims,
                                      const SmallVector<int64_t> &dims1,
                                      const SmallVector<int64_t> &dims2) const {

    SmallVector<int64_t> permutation;
    permutation.append(batchDims);
    permutation.append(dims1);
    permutation.append(dims2);

    return permutation;
  }

  ttir::PermuteOp
  createPermuteOp(PatternRewriter &rewriter, Location loc, Value input,
                  RankedTensorType inputType,
                  const SmallVector<int64_t> &permutation) const {

    SmallVector<int64_t> destinationShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), permutation);

    auto permuteOp = rewriter.create<ttir::PermuteOp>(
        loc,
        RankedTensorType::get(destinationShape, inputType.getElementType(),
                              inputType.getEncoding()),
        input, permutation);

    return permuteOp;
  }

  SmallVector<int64_t>
  computeMatmulInputShape(ConversionPatternRewriter &rewriter,
                          RankedTensorType tensorType,
                          const SmallVector<int64_t> &batchDims,
                          const SmallVector<int64_t> &contractDims,
                          const SmallVector<int64_t> &resultDims) const {

    SmallVector<int64_t> finalShape;

    // Add the batch dimensions.
    for (auto dim : batchDims) {
      finalShape.push_back(tensorType.getShape()[dim]);
    }

    // Add the result and contract product dimensions.
    finalShape.push_back(
        computeProductOfDims(tensorType.getShape(), contractDims));
    finalShape.push_back(
        computeProductOfDims(tensorType.getShape(), resultDims));

    return finalShape;
  }

  ttir::ReshapeOp
  createMatmulFinal(PatternRewriter &rewriter, Location loc, Value input,
                    RankedTensorType type,
                    const SmallVector<int64_t> &finalShape) const {

    llvm::SmallVector<int32_t> finalShapeI32(finalShape.begin(),
                                             finalShape.end());

    auto reshapeOp = rewriter.create<ttir::ReshapeOp>(
        loc,
        RankedTensorType::get(finalShape, type.getElementType(),
                              type.getEncoding()),
        input, rewriter.getI32ArrayAttr(finalShapeI32));

    return reshapeOp;
  }

  int64_t computeProductOfDims(ArrayRef<int64_t> tensorShape,
                               ArrayRef<int64_t> dims) const {
    int64_t product = 1;
    for (auto dim : dims) {
      product *= tensorShape[dim];
    }
    return product;
  }
};
} // namespace

// The following pattern rewriter will replace a pooling op with a FullOp in the
// case where the pooling operation is applied to the result of a FullOp.
namespace {
template <typename Pool2dOp>
class PoolingToFullOp : public OpConversionPattern<Pool2dOp> {
public:
  using OpConversionPattern<Pool2dOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Pool2dOp op, typename Pool2dOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttir::FullOp fullOp =
        dyn_cast_or_null<ttir::FullOp>(op.getInput().getDefiningOp());
    if (!fullOp) {
      return failure();
    }

    std::variant<int64_t, float> fillValue =
        isa<IntegerAttr>(fullOp.getFillValue())
            ? dyn_cast<IntegerAttr>(fullOp.getFillValue())
                  .getValue()
                  .getSExtValue()
            : dyn_cast<FloatAttr>(fullOp.getFillValue())
                  .getValue()
                  .convertToFloat();

    mlir::Attribute fillValueAttr =
        std::holds_alternative<int64_t>(fillValue)
            ? cast<mlir::Attribute>(
                  IntegerAttr::get(IntegerType::get(rewriter.getContext(), 32),
                                   std::get<int64_t>(fillValue)))
            : cast<mlir::Attribute>(
                  FloatAttr::get(Float32Type::get(rewriter.getContext()),
                                 std::get<float>(fillValue)));

    rewriter.replaceOp(
        op, rewriter.create<ttir::FullOp>(op.getLoc(), op.getResult().getType(),
                                          fillValueAttr));

    return success();
  }
};
} // namespace

// IndexSelectOp is converted to a series of SliceStaticOp and potentially a
// ConcatOp if the sliced dimension is sliced multiple times. For example, if
// the input tensor is
//    [[[1, 2, 3],
//      [4, 5, 6],
//      [7, 8, 9],
//      [10, 11, 12],
//      [13, 14, 15],
//      [16, 17, 18]],
//     [[19, 20, 21],
//      [22, 23, 24],
//      [25, 26, 27],
//      [28, 29, 30],
//      [31, 32, 33],
//      [34, 35, 36]]],
//    shape = [2, 6, 3]
// and the IndexSelectOp is dim=1, begin=0, length=2, stride=4, the output
// tensor will be
//    [[[1, 2, 3],
//      [4, 5, 6],
//      [13, 14, 15],
//      [16, 17, 18]],
//     [[19, 20, 21],
//      [22, 23, 24],
//      [31, 32, 33],
//      [34, 35, 36]]],
//    shape = [2, 4, 3]
// In this case 2 slices are created and concatenated to form the output tensor.
// First slice has begins=[0, 0, 0], ends=[2, 2, 3], steps=[1, 1, 1], and the
// second slice has begins=[0, 4, 0], ends=[2, 6, 3], steps=[1, 1, 1].
namespace {
struct SelectToSliceConversionPattern
    : public OpConversionPattern<ttir::IndexSelectOp> {
public:
  using OpConversionPattern<ttir::IndexSelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::IndexSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getType());

    auto inputShape = inputType.getShape();

    int32_t dim =
        op.getDim() < 0 ? inputType.getRank() + op.getDim() : op.getDim();

    int32_t begin = op.getBegin();
    int32_t length = op.getLength();
    int32_t stride = op.getStride();

    int32_t inputDimSize = inputType.getShape()[dim];
    int32_t numSlices = (inputDimSize - begin + stride - 1) / stride;

    llvm::SmallVector<int32_t, 4> begins, ends, steps;
    for (int32_t i = 0; i < inputType.getRank(); ++i) {
      // Always slicing with step 1.
      steps.push_back(1);
      if (i == dim) {
        // Push placeholder values for now which will be updated later.
        begins.push_back(0);
        ends.push_back(0);
        continue;
      }

      // For non-sliced dimensions, begin=0, end=dimSize, step=1.
      begins.push_back(0);
      ends.push_back(inputType.getDimSize(i));
    }

    // Create a slice for each slice of the input tensor. The slices are then
    // concatenated. The slices are created by updating the begin and end values
    // for the sliced dimension.
    llvm::SmallVector<Value> slices;
    for (int32_t i = 0; i < numSlices; ++i) {
      int32_t newBegin = begin + i * stride;
      int32_t newEnd = std::min(newBegin + length, inputDimSize);

      // Make a copy of the input shape and update the dim size.
      llvm::SmallVector<int64_t> resultShape(inputShape);
      resultShape[dim] = newEnd - newBegin;

      begins[dim] = newBegin;
      ends[dim] = newEnd;

      auto newOp = rewriter.create<ttir::SliceStaticOp>(
          op.getLoc(),
          RankedTensorType::get(resultShape, inputType.getElementType(),
                                inputType.getEncoding()),
          adaptor.getInput(), rewriter.getI32ArrayAttr(begins),
          rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));
      slices.push_back(newOp);
    }

    assert(!slices.empty());
    if (slices.size() > 1) {
      auto concatOp =
          rewriter.create<ttir::ConcatOp>(op.getLoc(), outputType, slices, dim);
      rewriter.replaceOp(op, concatOp);
    } else {
      rewriter.replaceOp(op, slices[0]);
    }

    return success();
  }
};
} // namespace

/*
 * This pattern rewrites ArangeOp by forcing the arange_dimension to be
 * rightmost dimension of the output tensor. This is done by replacing the
 * ArangeOp with a new one that has this property, and then transposing out last
 * dimension to the dimension specified by the original ArangeOp, and also
 * inserting a reshape to match the rank of the intended output and broadcasts
 * to repeat the data along the other dimensions.
 *
 * The ArangeOp that is generated here will be equivalent to how ttnn::ArangeOp
 * behaves.
 */
namespace {
struct ArangeForceLastDimensionPattern
    : public OpConversionPattern<ttir::ArangeOp> {
public:
  using OpConversionPattern<ttir::ArangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const RankedTensorType outputType =
        mlir::cast<RankedTensorType>(op.getResult().getType());

    int64_t arangeDimension = adaptor.getArangeDimension();
    int64_t start = adaptor.getStart();
    int64_t end = adaptor.getEnd();
    int64_t step = adaptor.getStep();

    int64_t arangeLength = (end - start) / step;

    const llvm::SmallVector<int64_t, 1> requiredShape{arangeLength};
    ArrayRef<int64_t> ttnnShape(requiredShape);
    if (ttnnShape == outputType.getShape()) {
      return success();
    }

    RankedTensorType arangeOutputType = RankedTensorType::get(
        requiredShape, outputType.getElementType(), outputType.getEncoding());

    Value output =
        rewriter
            .create<ttir::ArangeOp>( // perform arange on the last dimension to
                                     // match how ttnn behaves
                op.getLoc(), arangeOutputType, start, end, step, 0)
            .getResult();

    std::vector<int64_t> outputShape = arangeOutputType.getShape().vec();

    // Must match up the rank of the output with the rank of the intended output
    // from the original arange, with the arangeDimension in the correct
    // position
    if (outputType.getRank() != static_cast<int64_t>(outputShape.size())) {
      std::vector<int64_t> reshapeShape;
      for (uint32_t i = 0; i < outputType.getRank(); i++) {
        i == arangeDimension ? reshapeShape.push_back(arangeLength)
                             : reshapeShape.push_back(1);
      }

      output = rewriter.create<ttir::ReshapeOp>(
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshapeOutput"),
          RankedTensorType::get(reshapeShape, outputType.getElementType(),
                                outputType.getEncoding()),
          output,
          rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(
              reshapeShape.begin(), reshapeShape.end())));

      outputShape = std::move(reshapeShape);
    }

    // Must broadcast the rest of the dimensions.
    SmallVector<Attribute> broadcastDims;
    for (uint32_t i = 0; i < outputShape.size(); i++) {
      if (i != arangeDimension && outputShape[i] != outputType.getShape()[i]) {
        outputShape[i] = outputType.getShape()[i];
        broadcastDims.push_back(rewriter.getI64IntegerAttr(i));
      }
    }
    if (!broadcastDims.empty()) {
      RankedTensorType broadcastType = RankedTensorType::get(
          outputShape, outputType.getElementType(), outputType.getEncoding());

      auto inputShape =
          mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();

      SmallVector<int64_t> broadcastShape =
          ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                         outputShape);

      output = rewriter.create<ttir::BroadcastOp>(
          ttmlir::utils::appendLocationSuffix(op.getLoc(), "_broadcastOutput"),
          broadcastType, output, broadcastShape);

      assert(mlir::cast<RankedTensorType>(output.getType()).getShape() ==
                 outputType.getShape() &&
             "Output shape must match the shape of the input tensor");
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

// TTNN does not support reduction operation for logical and. So this reduction
// is performed by decomposing/converting into reduction product (ttnn.prod op).
// If ttnn.prod output is zero then reduce_and output is false; otherwise the
// output is true.
namespace {
struct ReductionAndPattern : public OpConversionPattern<ttir::ReduceAndOp> {
public:
  using OpConversionPattern<ttir::ReduceAndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType reduceOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    rewriter.replaceOpWithNewOp<ttir::ProdOp>(
        op, reduceOutputType, adaptor.getInput(), op.getKeepDim(),
        op.getDimArgAttr());

    return success();
  }
};
} // namespace

// TTNN does not support reduction operation for logical or. So this reduction
// is performed by decomposing/converting into reduction sum (ttnn.sum op).
// If ttnn.sum output is zero then reduce_or output is false; otherwise the
// output is true.
// This is a performance optimization specific to TTNN backend.
namespace {
struct ReductionOrTTNNPattern : public OpConversionPattern<ttir::ReduceOrOp> {
public:
  using OpConversionPattern<ttir::ReduceOrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType reduceOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    rewriter.replaceOpWithNewOp<ttir::SumOp>(
        op, reduceOutputType, adaptor.getInput(), op.getKeepDim(),
        op.getDimArgAttr());

    return success();
  }
};
} // namespace

// TTNN complete Decomposition for reduce_or Op.
namespace {
struct ReductionOrPattern : public OpConversionPattern<ttir::ReduceOrOp> {
public:
  using OpConversionPattern<ttir::ReduceOrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ReduceOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType reduceOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    mlir::Value sumOp = rewriter.create<ttir::SumOp>(
        op.getLoc(), reduceOutputType, adaptor.getInput(), op.getKeepDim(),
        op.getDimArgAttr());

    // Create zero constant.
    auto elementType = reduceOutputType.getElementType();
    Attribute zerAttr;
    if (mlir::isa<mlir::FloatType>(elementType)) {
      zerAttr = rewriter.getFloatAttr(elementType, 0.0);
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      zerAttr = rewriter.getIntegerAttr(elementType, 0);
    } else {
      return rewriter.notifyMatchFailure(
          op, "reduce_or decomposition only supports floating-point and "
              "integer element types");
    }

    ElementsAttr zeroConstantAttr =
        DenseElementsAttr::get(reduceOutputType, zerAttr);
    mlir::Value zeroConstant = rewriter.create<ttir::ConstantOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_zeroConstant"),
        reduceOutputType, zeroConstantAttr);

    // Compare sum != 0.
    mlir::Value cmpOp = rewriter.create<ttir::NotEqualOp>(
        op.getLoc(), reduceOutputType, sumOp, zeroConstant);

    // Typecast boolean result to float type.
    rewriter.replaceOpWithNewOp<ttir::TypecastOp>(op, reduceOutputType, cmpOp);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// BatchNorm decomposition helpers
//===----------------------------------------------------------------------===//

namespace {
// Helper function that ensures input is in NCHW format by permuting and
// reshaping the input tensor. Returns the transformed value and the normalized
// shape.
static std::pair<mlir::Value, llvm::SmallVector<int64_t>>
normalizeToNCHW(mlir::Value input, uint64_t featureIndex,
                ConversionPatternRewriter &rewriter, mlir::Location loc) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::ArrayRef<int64_t> shape = inputType.getShape();
  mlir::Value newInput = input;
  llvm::SmallVector<int64_t> currentShape(shape.begin(), shape.end());

  // If feature index is not 1, permute the input tensor so that the feature
  // dimension is at index 1 (NCHW format).
  if (featureIndex != 1) {
    // Build permutation to move featureIndex to position 1.
    llvm::SmallVector<int64_t> permutation(currentShape.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[featureIndex], permutation[1]);
    llvm::SmallVector<int64_t> permutedShape = ttmlir::utils::applyPermutation(
        llvm::ArrayRef(currentShape), llvm::ArrayRef(permutation));

    newInput = rewriter.create<mlir::tt::ttir::PermuteOp>(
        loc, RankedTensorType::get(permutedShape, inputType.getElementType()),
        newInput, rewriter.getDenseI64ArrayAttr(permutation));
    currentShape = permutedShape;
  }

  // Reshape to 4D NCHW if needed:
  // If rank is 5, flatten last two dimensions into one.
  // If rank is less than 4, unsqueeze trailing dimensions until rank is 4.
  int64_t rank = currentShape.size();
  if (rank == 5) {
    llvm::SmallVector<int64_t> reshapedShape = {
        currentShape[0], currentShape[1], currentShape[2],
        currentShape[3] * currentShape[4]};
    llvm::SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                                reshapedShape.end());
    newInput = rewriter.create<mlir::tt::ttir::ReshapeOp>(
        loc,
        RankedTensorType::get(reshapedShape, inputType.getElementType(),
                              inputType.getEncoding()),
        newInput, rewriter.getI32ArrayAttr(reshapedShapeI32));
    currentShape = reshapedShape;
  } else if (rank < 4) {
    llvm::SmallVector<int64_t> reshapedShape(currentShape.begin(),
                                             currentShape.end());
    reshapedShape.append(4 - rank, 1);
    llvm::SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                                reshapedShape.end());
    newInput = rewriter.create<mlir::tt::ttir::ReshapeOp>(
        loc,
        RankedTensorType::get(reshapedShape, inputType.getElementType(),
                              inputType.getEncoding()),
        newInput, rewriter.getI32ArrayAttr(reshapedShapeI32));
    currentShape = reshapedShape;
  }

  return {newInput, currentShape};
}

// Helper function to denormalize output back to original layout.
// Forward pass: originalShape -> [permute] -> shapeAfterPermute -> [reshape] ->
// normalizedShape Backward pass: normalizedShape -> [undo reshape] ->
// shapeAfterPermute -> [undo permute] -> originalShape
static mlir::Value denormalizeFromNCHW(mlir::Value output,
                                       llvm::ArrayRef<int64_t> originalShape,
                                       llvm::ArrayRef<int64_t> normalizedShape,
                                       uint64_t originalFeatureIndex,
                                       ConversionPatternRewriter &rewriter,
                                       mlir::Location loc) {
  auto outputType = mlir::cast<mlir::RankedTensorType>(output.getType());
  mlir::Value result = output;

  //  Undo reshape if ranks differ (in reverse order of forward pass)
  if (originalShape.size() != normalizedShape.size()) {
    // Compute the shape after permute but before reshape (the intermediate
    // state). This is what the tensor shape would be if we only applied
    // permutation.
    llvm::SmallVector<int64_t> shapeAfterPermute(originalShape.begin(),
                                                 originalShape.end());
    if (originalFeatureIndex != 1) {
      std::swap(shapeAfterPermute[1], shapeAfterPermute[originalFeatureIndex]);
    }

    llvm::SmallVector<int32_t> shapeAfterPermuteI32(shapeAfterPermute.begin(),
                                                    shapeAfterPermute.end());
    result = rewriter.create<mlir::tt::ttir::ReshapeOp>(
        loc,
        RankedTensorType::get(shapeAfterPermute, outputType.getElementType(),
                              outputType.getEncoding()),
        result, rewriter.getI32ArrayAttr(shapeAfterPermuteI32));
  }

  // Step 2: Undo permutation if featureIndex != 1
  if (originalFeatureIndex != 1) {
    // The inverse permutation is the same as the forward permutation
    // (swapping dimensions 1 and featureIndex is its own inverse)
    llvm::SmallVector<int64_t> permutation(originalShape.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[1], permutation[originalFeatureIndex]);

    result = rewriter.create<mlir::tt::ttir::PermuteOp>(
        loc,
        RankedTensorType::get(originalShape, outputType.getElementType(),
                              outputType.getEncoding()),
        result, rewriter.getDenseI64ArrayAttr(permutation));
  }

  return result;
}

// Helper function to check if input type is valid for BatchNorm weight tensors
static bool isValidBatchNormWeightType(RankedTensorType inputType) {
  if (inputType.getRank() == 1) {
    return true;
  }
  if (inputType.getRank() == 4) {
    auto shape = inputType.getShape();
    return shape[0] == 1 && shape[2] == 1 && shape[3] == 1;
  }
  return false;
}

// Helper function to reshape BatchNorm weight tensors from 1D to 4D [1, C, 1,
// 1]
static mlir::Value getBatchNorm4DTensor(PatternRewriter &rewriter, Location loc,
                                        mlir::Value batchNormInput) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(batchNormInput.getType());

  if (inputType.getRank() == 4) {
    return batchNormInput;
  }

  auto newShape = llvm::SmallVector<int64_t>{1, inputType.getDimSize(0), 1, 1};
  llvm::SmallVector<int32_t> shape32(newShape.begin(), newShape.end());
  auto shapeAttr = rewriter.getI32ArrayAttr(shape32);

  return rewriter.create<ttir::ReshapeOp>(
      loc,
      RankedTensorType::get(newShape, inputType.getElementType(),
                            inputType.getEncoding()),
      batchNormInput, shapeAttr);
}

// Helper function to reshape BatchNorm weight tensors from 4D [1, C, 1, 1] to
// 1D [C]
static mlir::Value reshapeBatchNorm4DTo1D(PatternRewriter &rewriter,
                                          Location loc, mlir::Value input4D,
                                          RankedTensorType target1DType) {
  auto input4DType = mlir::cast<RankedTensorType>(input4D.getType());

  // If already 1D, return as-is
  if (input4DType.getRank() == 1) {
    return input4D;
  }

  // Extract the channel dimension from [1, C, 1, 1] -> [C]
  llvm::SmallVector<int32_t> shape1D = {
      static_cast<int32_t>(target1DType.getDimSize(0))};

  return rewriter.create<ttir::ReshapeOp>(
      loc,
      RankedTensorType::get(target1DType.getShape(),
                            target1DType.getElementType(),
                            target1DType.getEncoding()),
      input4D, rewriter.getI32ArrayAttr(shape1D));
}
} // namespace

//===----------------------------------------------------------------------===//
// BatchNorm decomposition patterns
//===----------------------------------------------------------------------===//

// This pattern reshapes the non input tensors of the BatchNormInferenceOp to 4D
// tensors, by adding additional dimensions of size 1 so that the only
// non-1 dimension is the second dimension. This is done so that the
// op is compatible with ttnn op call.
namespace {
struct BatchNormInferencePattern
    : public OpConversionPattern<ttir::BatchNormInferenceOp> {
public:
  using OpConversionPattern<ttir::BatchNormInferenceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormInferenceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    llvm::ArrayRef<int64_t> originalShape = inputType.getShape();
    uint64_t featureIndex =
        adaptor.getDimensionAttr().getValue().getZExtValue();

    auto meanType = mlir::cast<RankedTensorType>(adaptor.getMean().getType());
    if (!isValidBatchNormWeightType(meanType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp mean must be 1D tensor");
    }

    auto varType =
        mlir::cast<RankedTensorType>(adaptor.getVariance().getType());
    if (!isValidBatchNormWeightType(varType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp var must be 1D or 4D tensor");
    }

    auto weightType =
        mlir::cast<RankedTensorType>(adaptor.getScale().getType());
    if (!isValidBatchNormWeightType(weightType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp weight must be 1D or 4D tensor");
    }

    auto biasType = mlir::cast<RankedTensorType>(adaptor.getOffset().getType());
    if (!isValidBatchNormWeightType(biasType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormInferenceOp bias must be 1D or 4D tensor");
    }

    // Normalize input to NCHW format
    auto [normalizedInput, normalizedShape] =
        normalizeToNCHW(adaptor.getOperand(), featureIndex, rewriter, loc);

    // Reshape weight tensors to 4D (existing logic for TTNN compatibility)
    mlir::Value mean4D = getBatchNorm4DTensor(rewriter, loc, adaptor.getMean());
    mlir::Value variance4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getVariance());
    mlir::Value scale4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getScale());
    mlir::Value offset4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getOffset());

    // Create output type with normalized shape
    auto originalOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto normalizedOutputType = RankedTensorType::get(
        normalizedShape, originalOutputType.getElementType(),
        originalOutputType.getEncoding());

    // After normalization, feature dimension is always at index 1 (NCHW)
    mlir::Type integerType = mlir::IntegerType::get(rewriter.getContext(), 32);
    IntegerAttr dimensionAttr = mlir::IntegerAttr::get(integerType, 1);

    // Create the BatchNorm op with normalized input and 4D weight tensors
    auto batchNormInferenceOp =
        rewriter.create<mlir::tt::ttir::BatchNormInferenceOp>(
            loc, normalizedOutputType, normalizedInput, scale4D, offset4D,
            mean4D, variance4D, adaptor.getEpsilonAttr(), dimensionAttr);

    // Denormalize output back to original layout
    mlir::Value result =
        denormalizeFromNCHW(batchNormInferenceOp.getResult(), originalShape,
                            normalizedShape, featureIndex, rewriter, loc);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// This pattern reshapes the non input tensors of the BatchNormTrainingOp to 4D
// tensors so that the resulting BatchNormTrainingOp is compatible with ttnn op
// call.
namespace {
struct BatchNormTrainingPattern
    : public OpConversionPattern<ttir::BatchNormTrainingOp> {
public:
  using OpConversionPattern<ttir::BatchNormTrainingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::BatchNormTrainingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputType =
        mlir::cast<RankedTensorType>(adaptor.getOperand().getType());
    llvm::ArrayRef<int64_t> originalShape = inputType.getShape();
    uint64_t featureIndex =
        adaptor.getDimensionAttr().getValue().getZExtValue();

    auto scaleType = mlir::cast<RankedTensorType>(adaptor.getScale().getType());
    if (!isValidBatchNormWeightType(scaleType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp scale must be 1D or 4D tensor");
    }

    auto offsetType =
        mlir::cast<RankedTensorType>(adaptor.getOffset().getType());
    if (!isValidBatchNormWeightType(offsetType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp offset must be 1D or 4D tensor");
    }

    auto meanType =
        mlir::cast<RankedTensorType>(adaptor.getRunningMean().getType());
    if (!isValidBatchNormWeightType(meanType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp running_mean must be 1D or 4D tensor");
    }

    auto varType =
        mlir::cast<RankedTensorType>(adaptor.getRunningVariance().getType());
    if (!isValidBatchNormWeightType(varType)) {
      return rewriter.notifyMatchFailure(
          op, "BatchNormTrainingOp running_variance must be 1D or 4D tensor");
    }

    // Normalize input to NCHW format
    auto [normalizedInput, normalizedShape] =
        normalizeToNCHW(adaptor.getOperand(), featureIndex, rewriter, loc);

    // Reshape all weight tensors to 4D (for TTNN compatibility)
    mlir::Value scale4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getScale());
    mlir::Value offset4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getOffset());
    mlir::Value mean4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getRunningMean());
    mlir::Value variance4D =
        getBatchNorm4DTensor(rewriter, loc, adaptor.getRunningVariance());

    // Create output types with normalized shape and 4D weight tensors
    auto originalOutputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto normalizedOutputType = RankedTensorType::get(
        normalizedShape, originalOutputType.getElementType(),
        originalOutputType.getEncoding());

    // running_mean and running_variance results should be 4D [1, C, 1, 1]
    auto mean4DType = mlir::cast<RankedTensorType>(mean4D.getType());
    auto variance4DType = mlir::cast<RankedTensorType>(variance4D.getType());

    // After normalization, feature dimension is always at index 1 (NCHW)
    mlir::Type integerType = mlir::IntegerType::get(rewriter.getContext(), 32);
    IntegerAttr dimensionAttr = mlir::IntegerAttr::get(integerType, 1);

    // Create new BatchNormTrainingOp with normalized input and all 4D weight
    // tensors
    auto batchNormTrainingOp = rewriter.create<ttir::BatchNormTrainingOp>(
        loc, TypeRange{normalizedOutputType, mean4DType, variance4DType},
        normalizedInput, scale4D, offset4D, mean4D, variance4D,
        adaptor.getEpsilonAttr(), dimensionAttr, adaptor.getMomentumAttr());

    // Denormalize the output (first result) back to original layout
    mlir::Value denormalizedOutput =
        denormalizeFromNCHW(batchNormTrainingOp.getResults()[0], originalShape,
                            normalizedShape, featureIndex, rewriter, loc);

    // Reshape batch_mean and batch_variance from 4D [1, C, 1, 1] back to 1D [C]
    auto originalMeanType =
        mlir::cast<RankedTensorType>(op.getBatchMean().getType());
    auto originalVarianceType =
        mlir::cast<RankedTensorType>(op.getBatchVariance().getType());

    mlir::Value reshapedMean = reshapeBatchNorm4DTo1D(
        rewriter, loc, batchNormTrainingOp.getResults()[1], originalMeanType);
    mlir::Value reshapedVariance = reshapeBatchNorm4DTo1D(
        rewriter, loc, batchNormTrainingOp.getResults()[2],
        originalVarianceType);

    // Replace with denormalized output and reshaped mean/variance results
    rewriter.replaceOp(
        op, ValueRange{denormalizedOutput, reshapedMean, reshapedVariance});

    return success();
  }
};
} // namespace

// Utility function to get scale and zero point for quantized types.
static std::pair<mlir::Value, mlir::Value>
getScaleAndZeroPoint(mlir::quant::QuantizedType elementType,
                     ConversionPatternRewriter &rewriter, mlir::Location loc) {
  // Per-tensor quantization.
  if (auto quantPerTensorType =
          mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
    // Create ttir::ConstantOp for scale.
    float scaleValue = quantPerTensorType.getScale();
    mlir::RankedTensorType scaleType =
        mlir::RankedTensorType::get({1}, rewriter.getF32Type());
    mlir::DenseFPElementsAttr scaleDenseAttr =
        mlir::DenseFPElementsAttr::get(scaleType, scaleValue);
    ttir::ConstantOp scaleConstant =
        rewriter.create<ttir::ConstantOp>(loc, scaleType, scaleDenseAttr);

    // Create ttir::ConstantOp for zero point.
    int32_t zeroPoint = static_cast<int32_t>(quantPerTensorType.getZeroPoint());
    mlir::RankedTensorType zeroPointType = mlir::RankedTensorType::get(
        {1}, IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed));
    mlir::DenseIntElementsAttr zeroPointDenseAttr =
        mlir::DenseIntElementsAttr::get(zeroPointType, zeroPoint);
    ttir::ConstantOp zeroPointConstant = rewriter.create<ttir::ConstantOp>(
        loc, zeroPointType, zeroPointDenseAttr);
    return {scaleConstant.getResult(), zeroPointConstant.getResult()};
  }

  // Per-axis quantization.
  if (auto quantPerAxisType =
          mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
              elementType)) {
    // Create ttir::ConstantOp for scale.
    SmallVector<float> scales(
        llvm::to_vector_of<float>(quantPerAxisType.getScales()));
    mlir::RankedTensorType scaleType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(scales.size())}, rewriter.getF32Type());
    mlir::DenseFPElementsAttr scaleDenseAttr =
        mlir::DenseFPElementsAttr::get(scaleType, scales);
    ttir::ConstantOp scaleConstant =
        rewriter.create<ttir::ConstantOp>(loc, scaleType, scaleDenseAttr);

    // Create ttir::ConstantOp for zero point.
    SmallVector<int32_t> zeroPoints(
        llvm::to_vector_of<int32_t>(quantPerAxisType.getZeroPoints()));
    mlir::RankedTensorType zeroPointType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(zeroPoints.size())},
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed));
    mlir::DenseIntElementsAttr zeroPointDenseAttr =
        mlir::DenseIntElementsAttr::get(zeroPointType, zeroPoints);
    ttir::ConstantOp zeroPointConstant = rewriter.create<ttir::ConstantOp>(
        loc, zeroPointType, zeroPointDenseAttr);
    return {scaleConstant.getResult(), zeroPointConstant.getResult()};
  }

  return {nullptr, nullptr};
}

// Utility function to get axis for quantized types.
static IntegerAttr getAxis(mlir::quant::QuantizedType elementType,
                           ConversionPatternRewriter &rewriter) {
  IntegerAttr axis;
  if (auto perAxisType =
          mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
              elementType)) {
    axis = rewriter.getI32IntegerAttr(perAxisType.getQuantizedDimension());
  }
  return axis;
}

// TTNN runtime requires scale and zero point to be treated as input operands
// to quantize and dequantize ops. This reduction creates constant ops for scale
// and zero point and populates the TTIR quantize/dequantize ops with these
// constants as inputs.
namespace {
template <typename QuantizeOpTy, typename QuantizeUnrolledOpTy>
class QuantizationOpConversionPatternBase
    : public OpConversionPattern<QuantizeOpTy> {
public:
  using OpConversionPattern<QuantizeOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QuantizeOpTy op,
                  typename OpConversionPattern<QuantizeOpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::quant::QuantizedType elementType = getQuantizedElementType(op);
    if (!elementType) {
      return failure();
    }
    auto [scale, zeroPoint] =
        getScaleAndZeroPoint(elementType, rewriter, op.getLoc());
    if (!scale) {
      return rewriter.notifyMatchFailure(
          op, "Failed to extract scale and zero point from quantized type.");
    }
    IntegerAttr axisAttr = getAxis(elementType, rewriter);
    mlir::Type quantizeOutputType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<QuantizeUnrolledOpTy>(
        op, quantizeOutputType, adaptor.getInput(), scale, zeroPoint, axisAttr);
    return success();
  }

protected:
  virtual mlir::quant::QuantizedType
  getQuantizedElementType(QuantizeOpTy op) const = 0;
};

struct QuantizeOpPattern
    : public QuantizationOpConversionPatternBase<ttir::QuantizeOp,
                                                 ttir::QuantizeUnrolledOp> {
  using QuantizationOpConversionPatternBase::
      QuantizationOpConversionPatternBase;

protected:
  mlir::quant::QuantizedType
  getQuantizedElementType(ttir::QuantizeOp op) const override {
    mlir::RankedTensorType outputType = op.getResult().getType();
    return mlir::dyn_cast<mlir::quant::QuantizedType>(
        outputType.getElementType());
  }
};

struct DequantizeOpPattern
    : public QuantizationOpConversionPatternBase<ttir::DequantizeOp,
                                                 ttir::DequantizeUnrolledOp> {
  using QuantizationOpConversionPatternBase::
      QuantizationOpConversionPatternBase;

protected:
  mlir::quant::QuantizedType
  getQuantizedElementType(ttir::DequantizeOp op) const override {
    mlir::RankedTensorType inputType = op.getInput().getType();
    return mlir::dyn_cast<mlir::quant::QuantizedType>(
        inputType.getElementType());
  }
};

struct RequantizeOpPattern : public OpConversionPattern<ttir::RequantizeOp> {
public:
  using OpConversionPattern<ttir::RequantizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::RequantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType inputType = op.getInput().getType();
    mlir::RankedTensorType outputType = op.getResult().getType();

    mlir::quant::QuantizedType inputElementType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(inputType.getElementType());
    mlir::quant::QuantizedType outputElementType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(outputType.getElementType());

    if (!inputElementType || !outputElementType) {
      return failure();
    }

    auto [inputScale, inputZeroPoint] =
        getScaleAndZeroPoint(inputElementType, rewriter, op.getLoc());
    if (!inputScale) {
      return rewriter.notifyMatchFailure(
          op,
          "Failed to extract input scale and zero point from quantized type.");
    }

    auto [outputScale, outputZeroPoint] =
        getScaleAndZeroPoint(outputElementType, rewriter, op.getLoc());
    if (!outputScale) {
      return rewriter.notifyMatchFailure(
          op,
          "Failed to extract output scale and zero point from quantized type.");
    }

    IntegerAttr axisAttr = getAxis(inputElementType, rewriter);
    mlir::Type requantizeOutputType =
        this->getTypeConverter()->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<ttir::RequantizeUnrolledOp>(
        op, requantizeOutputType, adaptor.getInput(), inputScale,
        inputZeroPoint, outputScale, outputZeroPoint, axisAttr);
    return success();
  }
};

} // namespace

// TTNN api supports product reduction along one or all dimensions. This
// decomposition will transform product reduction op to multiple reduction ops.
// Each op will perform reduction along one dimension only.
namespace {
struct ReductionProdPattern : public OpConversionPattern<ttir::ProdOp> {
public:
  using OpConversionPattern<ttir::ProdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ProdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimArg = op.getDimArg();
    if (!dimArg) {
      return failure();
    }

    uint64_t rank = op.getInput().getType().getRank();
    uint64_t dimArgSize = dimArg->size();
    if (dimArgSize == 1 || dimArgSize == rank) {
      return failure();
    }

    // Extract reduction dimensions.
    llvm::SmallVector<int32_t> reduceDims(
        llvm::map_to_vector(*dimArg, [](Attribute dim) -> int32_t {
          return mlir::cast<IntegerAttr>(dim).getInt();
        }));

    // Reduce dimensions are sorted in descending order to apply reduction on
    // higher dimension first. This helps to avoid modifying dimArg which will
    // be required in case of applying reduction on lower dimension first.
    llvm::sort(reduceDims, std::greater<>());

    Value runningProdOp = op.getInput();
    llvm::SmallVector<int64_t> shape{op.getInput().getType().getShape()};
    auto elementType = op.getInput().getType().getElementType();
    bool keepDim = op.getKeepDim();

    for (int dim : reduceDims) {
      mlir::ArrayAttr dimArg =
          rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(/*Size=*/1, dim));
      if (keepDim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }

      RankedTensorType outputType = RankedTensorType::get(shape, elementType);
      runningProdOp = rewriter.create<ttir::ProdOp>(
          op.getLoc(), outputType, runningProdOp, op.getKeepDimAttr(), dimArg);
    }

    rewriter.replaceOp(op, runningProdOp);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Conv2d/ConvTranspose2d Channel Layout Decomposition
//===----------------------------------------------------------------------===//
// These patterns add permutes to convert from non-NHWC layouts to NHWC format
// when the dimension indices indicate a non-NHWC layout.

namespace {
template <typename ConvOpType>
struct ConvChannelLastDecompositionPattern
    : public OpConversionPattern<ConvOpType> {
  using OpConversionPattern<ConvOpType>::OpConversionPattern;
  using OpAdaptor = typename ConvOpType::Adaptor;

  LogicalResult
  matchAndRewrite(ConvOpType op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only process ops that are not already in NHWC format.
    if (op.isNHWC()) {
      return failure();
    }

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

    int64_t batchDim = op.getBatchDim();
    int64_t heightDim = op.getHeightDim();
    int64_t widthDim = op.getWidthDim();
    int64_t channelDim = op.getChannelDim();

    llvm::SmallVector<int64_t> toNhwc{batchDim, heightDim, widthDim,
                                      channelDim};

    // Compute inverse permutation from NHWC back to original layout.
    llvm::SmallVector<int64_t> fromNhwc =
        ttmlir::utils::inversePermutation(toNhwc);
    auto permutedInputShape =
        ttmlir::utils::applyPermutation(inputType.getShape(), toNhwc);
    auto permutedInputType =
        RankedTensorType::get(permutedInputShape, inputType.getElementType(),
                              inputType.getEncoding());
    auto permutedInput = rewriter.create<ttir::PermuteOp>(
        op.getLoc(), permutedInputType, adaptor.getInput(), toNhwc);

    // Compute output shape in NHWC format.
    auto permutedOutputShape =
        ttmlir::utils::applyPermutation(outputType.getShape(), toNhwc);
    auto permutedOutputType =
        RankedTensorType::get(permutedOutputShape, outputType.getElementType(),
                              outputType.getEncoding());

    // Permute bias from current layout to NHWC if present.
    Value permutedBias = adaptor.getBias();
    if (permutedBias) {
      auto biasType = mlir::cast<RankedTensorType>(permutedBias.getType());
      auto permutedBiasShape =
          ttmlir::utils::applyPermutation(biasType.getShape(), toNhwc);
      auto permutedBiasType = RankedTensorType::get(
          permutedBiasShape, biasType.getElementType(), biasType.getEncoding());
      permutedBias = rewriter.create<ttir::PermuteOp>(
          op.getLoc(), permutedBiasType, adaptor.getBias(), toNhwc);
    }

    ConvOpType newConv;
    if constexpr (std::is_same_v<ConvOpType, ttir::ConvTranspose2dOp>) {
      newConv = rewriter.create<ttir::ConvTranspose2dOp>(
          op.getLoc(), permutedOutputType, permutedInput, adaptor.getWeight(),
          permutedBias, adaptor.getStride(), adaptor.getPadding(),
          adaptor.getOutputPadding(), adaptor.getDilation(), op.getGroups(),
          adaptor.getFlattenedCompatInfo());
    } else if constexpr (std::is_same_v<ConvOpType, ttir::Conv2dOp>) {
      newConv = rewriter.create<ttir::Conv2dOp>(
          op.getLoc(), permutedOutputType, permutedInput, adaptor.getWeight(),
          permutedBias, adaptor.getStride(), adaptor.getPadding(),
          adaptor.getDilation(), op.getGroups(),
          adaptor.getFlattenedCompatInfo());
    } else {
      static_assert(ttmlir::utils::always_false<ConvOpType>(),
                    "Unsupported ConvOpType");
    }

    // Permute output from NHWC back to original layout.
    auto outputPermute = rewriter.create<ttir::PermuteOp>(
        op.getLoc(), outputType, newConv.getResult(), fromNhwc);

    rewriter.replaceOp(op, outputPermute.getResult());
    return success();
  }
};
} // namespace

namespace {
struct ArgMaxPattern : public OpConversionPattern<ttir::ArgMaxOp> {
  using OpConversionPattern<ttir::ArgMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ArgMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // This pattern works as follows:
    // 1. Permute input tensor, i.e, put all reduction dimensions at the end.
    // 2. Reshape the permuted tensor to make all reduction dimensions into one.
    // 3. Create a new ArgMax op with the reshaped tensor and the new reduction
    //    dimension.
    // 4. Reshape the output tensor to the expected shape in case of keep dim.
    auto tempDimArg = op.getDimArg();
    if (!tempDimArg || tempDimArg->size() <= 1) {
      return failure();
    }

    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto keepDim = op.getKeepDim();

    SmallVector<Attribute> canonicalizedDims;
    canonicalizedDims.reserve(tempDimArg->size());
    int32_t rank = inputType.getRank();

    for (Attribute dimAttr : *tempDimArg) {
      int32_t dim = mlir::cast<IntegerAttr>(dimAttr).getInt();
      if (dim < 0) {
        dim += rank;
      }
      canonicalizedDims.push_back(rewriter.getI32IntegerAttr(dim));
    }

    // Step 1: Permute input tensor.
    // Get the reduce dimensions indices so it's easier to reshape back;
    llvm::SmallDenseSet<int32_t> reduceDimsSet;
    SmallVector<int32_t> reduceDims, nonReduceDims;
    int32_t reshapedDimSize = 1;
    for (size_t dimIdx = 0; dimIdx < canonicalizedDims.size(); ++dimIdx) {
      Attribute dimAttr = canonicalizedDims[dimIdx];
      int32_t dim = mlir::cast<IntegerAttr>(dimAttr).getInt();
      reduceDims.push_back(dim);
      reshapedDimSize *= inputType.getDimSize(dim);
      reduceDimsSet.insert(dim);
    }
    llvm::sort(reduceDims);
    auto *uniqueReduceDims = std::unique(reduceDims.begin(), reduceDims.end());
    reduceDims.erase(uniqueReduceDims, reduceDims.end());
    // Prepare shape for reshaping input tensor.
    for (int32_t dimIdx = 0; dimIdx < inputType.getRank(); ++dimIdx) {
      if (!reduceDimsSet.count(dimIdx)) {
        nonReduceDims.push_back(dimIdx);
      }
    }
    SmallVector<int64_t> permuteShape;
    for (int32_t dimIdx : nonReduceDims) {
      permuteShape.push_back(inputType.getDimSize(dimIdx));
    }
    for (int32_t dimIdx : reduceDims) {
      permuteShape.push_back(inputType.getDimSize(dimIdx));
    }
    auto permuteOpResultType =
        RankedTensorType::get(permuteShape, inputType.getElementType());

    SmallVector<int64_t> permutation;
    for (int32_t dimIdx : nonReduceDims) {
      permutation.push_back(dimIdx);
    }
    for (int32_t dimIdx : reduceDims) {
      permutation.push_back(dimIdx);
    }
    auto permuteOp = rewriter.create<ttir::PermuteOp>(
        op.getLoc(), permuteOpResultType, adaptor.getInput(),
        rewriter.getDenseI64ArrayAttr(permutation));

    // Step 2. Reshape the permuted tensor to make all reduction dimensions into
    // one
    SmallVector<int64_t> tempArgMaxShape;
    for (int32_t dimIdx : nonReduceDims) {
      tempArgMaxShape.push_back(inputType.getDimSize(dimIdx));
    }

    tempArgMaxShape.push_back(reshapedDimSize);
    // RankedTensor need int64_t shape
    auto reshapeOp = rewriter.create<ttir::ReshapeOp>(
        op.getLoc(),
        RankedTensorType::get(tempArgMaxShape, inputType.getElementType()),
        permuteOp.getResult(),
        rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(tempArgMaxShape)));

    // Step 3. Create a new ArgMax op with the reshaped tensor and the new
    // reduction dimension.
    int32_t newDim = static_cast<int32_t>(tempArgMaxShape.size() - 1);
    if (keepDim) {
      tempArgMaxShape.back() = 1; // if keepDim is true, set reduced dim to 1
    } else {
      tempArgMaxShape.pop_back(); // else remove the reduced dim
    }
    auto argMaxOp = rewriter.create<ttir::ArgMaxOp>(
        op.getLoc(),
        RankedTensorType::get(tempArgMaxShape, outputType.getElementType()),
        reshapeOp.getResult(), rewriter.getBoolAttr(keepDim),
        rewriter.getI32ArrayAttr(newDim));

    // Step 4. Reshape the output tensor to the expected shape in case of keep
    // dim
    if (keepDim) {
      SmallVector<int64_t> finalArgMaxShape;
      for (int64_t dimIdx = 0; dimIdx < inputType.getRank(); ++dimIdx) {
        if (!reduceDimsSet.count(dimIdx)) {
          finalArgMaxShape.push_back(inputType.getDimSize(dimIdx));
        } else {
          finalArgMaxShape.push_back(1);
        }
      }
      auto result = rewriter.create<ttir::ReshapeOp>(
          op.getLoc(),
          RankedTensorType::get(finalArgMaxShape, outputType.getElementType()),
          argMaxOp.getResult(),
          rewriter.getI32ArrayAttr(
              llvm::to_vector_of<int32_t>(finalArgMaxShape)));
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, argMaxOp);
    }

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SplitQueryKeyValueAndSplitHeadsOp decomposition
//===----------------------------------------------------------------------===//
// Decomposes ttir.split_query_key_value_and_split_heads into:
//   - ttir.slice_static (to split Q, K, V from fused tensor)
//   - ttir.reshape (to reshape for multi-head attention)
//   - ttir.permute (to reorder dimensions)
//
// For MHA (no kv_input_tensor):
//   Input: [batch, seq, 3 * hidden_size]
//   1. Slice into Q, K, V each [batch, seq, hidden_size]
//   2. Reshape each to [batch, seq, num_heads, head_size]
//   3. Permute each with [0, 2, 1, 3] -> [batch, num_heads, seq, head_size]
//   4. If transpose_key, permute K with [0, 1, 3, 2]
//
// For GQA (with kv_input_tensor and num_kv_heads):
//   input_tensor: [batch, seq, hidden_size] for Q
//   kv_input_tensor: [batch, seq, 2 * kv_hidden_size] for K, V
//   Similar processing with separate num_kv_heads for K, V

namespace {
struct SplitQueryKeyValueAndSplitHeadsDecompositionPattern
    : public OpConversionPattern<ttir::SplitQueryKeyValueAndSplitHeadsOp> {
  using OpConversionPattern<
      ttir::SplitQueryKeyValueAndSplitHeadsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::SplitQueryKeyValueAndSplitHeadsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputTensor = adaptor.getInputTensor();
    Value kvInputTensor = adaptor.getKvInputTensor();

    auto inputType = mlir::cast<RankedTensorType>(inputTensor.getType());
    auto inputShape = inputType.getShape();

    uint32_t numHeads = adaptor.getNumHeads();
    bool transposeKey = adaptor.getTransposeKey();

    // Get output type to determine head_size.
    auto queryResultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getQuery().getType()));

    int64_t batchSize = inputShape[0];
    int64_t seqSize = inputShape[1];
    int64_t headSize = queryResultType.getShape()[3];

    Type elementType = inputType.getElementType();

    // Permutation: [batch, seq, num_heads, head_size] ->
    //              [batch, num_heads, seq, head_size]
    SmallVector<int64_t> qkvPermutation = {0, 2, 1, 3};

    if (kvInputTensor && adaptor.getNumKvHeads()) {
      // GQA case: input_tensor is Q, kv_input_tensor is KV.
      uint32_t numKvHeads = *adaptor.getNumKvHeads();

      // Process Q from input_tensor.
      // Reshape: [batch, seq, hidden] -> [batch, seq, num_heads, head_size].
      SmallVector<int64_t> reshapedQShape = {
          batchSize, seqSize, static_cast<int64_t>(numHeads), headSize};
      auto reshapedQType = RankedTensorType::get(reshapedQShape, elementType);
      Value reshapedQ =
          createReshape(rewriter, loc, inputTensor, reshapedQType);

      // Permute Q.
      SmallVector<int64_t> permutedQShape = {
          batchSize, static_cast<int64_t>(numHeads), seqSize, headSize};
      auto permutedQType = RankedTensorType::get(permutedQShape, elementType);
      Value query = createPermute(rewriter, loc, reshapedQ, permutedQType,
                                  qkvPermutation);

      // Process K and V from kv_input_tensor.
      int64_t kvHiddenSize = static_cast<int64_t>(numKvHeads) * headSize;

      // Slice K: [batch, seq, 0:kv_hidden_size].
      SmallVector<int64_t> kvSliceShape = {batchSize, seqSize, kvHiddenSize};
      auto kvSliceType = RankedTensorType::get(kvSliceShape, elementType);
      Value slicedK =
          createSlice(rewriter, loc, kvInputTensor, kvSliceType, {0, 0, 0},
                      {batchSize, seqSize, kvHiddenSize});

      // Slice V: [batch, seq, kv_hidden_size:2*kv_hidden_size].
      Value slicedV = createSlice(rewriter, loc, kvInputTensor, kvSliceType,
                                  {0, 0, kvHiddenSize},
                                  {batchSize, seqSize, 2 * kvHiddenSize});

      // Reshape K and V: [batch, seq, kv_hidden] ->
      //                  [batch, seq, num_kv_heads, head_size].
      SmallVector<int64_t> reshapedKVShape = {
          batchSize, seqSize, static_cast<int64_t>(numKvHeads), headSize};
      auto reshapedKVType = RankedTensorType::get(reshapedKVShape, elementType);
      Value reshapedK = createReshape(rewriter, loc, slicedK, reshapedKVType);
      Value reshapedV = createReshape(rewriter, loc, slicedV, reshapedKVType);

      // Permute K and V.
      SmallVector<int64_t> permutedKVShape = {
          batchSize, static_cast<int64_t>(numKvHeads), seqSize, headSize};
      auto permutedKVType = RankedTensorType::get(permutedKVShape, elementType);
      Value permutedK = createPermute(rewriter, loc, reshapedK, permutedKVType,
                                      qkvPermutation);
      Value permutedV = createPermute(rewriter, loc, reshapedV, permutedKVType,
                                      qkvPermutation);

      // Optionally transpose K.
      Value key = permutedK;
      if (transposeKey) {
        SmallVector<int64_t> transposedKShape = {
            batchSize, static_cast<int64_t>(numKvHeads), headSize, seqSize};
        auto transposedKType =
            RankedTensorType::get(transposedKShape, elementType);
        key = createPermute(rewriter, loc, permutedK, transposedKType,
                            {0, 1, 3, 2});
      }

      rewriter.replaceOp(op, {query, key, permutedV});
      return success();
    }

    // MHA case: input_tensor contains fused QKV.
    int64_t hiddenSize = inputShape[2] / 3;

    // Slice Q: [batch, seq, 0:hidden_size].
    SmallVector<int64_t> qkvSliceShape = {batchSize, seqSize, hiddenSize};
    auto qkvSliceType = RankedTensorType::get(qkvSliceShape, elementType);

    Value slicedQ = createSlice(rewriter, loc, inputTensor, qkvSliceType,
                                {0, 0, 0}, {batchSize, seqSize, hiddenSize});
    Value slicedK =
        createSlice(rewriter, loc, inputTensor, qkvSliceType,
                    {0, 0, hiddenSize}, {batchSize, seqSize, 2 * hiddenSize});
    Value slicedV = createSlice(rewriter, loc, inputTensor, qkvSliceType,
                                {0, 0, 2 * hiddenSize},
                                {batchSize, seqSize, 3 * hiddenSize});

    // Reshape: [batch, seq, hidden] -> [batch, seq, num_heads, head_size].
    SmallVector<int64_t> reshapedShape = {
        batchSize, seqSize, static_cast<int64_t>(numHeads), headSize};
    auto reshapedType = RankedTensorType::get(reshapedShape, elementType);
    Value reshapedQ = createReshape(rewriter, loc, slicedQ, reshapedType);
    Value reshapedK = createReshape(rewriter, loc, slicedK, reshapedType);
    Value reshapedV = createReshape(rewriter, loc, slicedV, reshapedType);

    // Permute: [batch, seq, num_heads, head_size] ->
    //          [batch, num_heads, seq, head_size].
    SmallVector<int64_t> permutedShape = {
        batchSize, static_cast<int64_t>(numHeads), seqSize, headSize};
    auto permutedType = RankedTensorType::get(permutedShape, elementType);
    Value query =
        createPermute(rewriter, loc, reshapedQ, permutedType, qkvPermutation);
    Value permutedK =
        createPermute(rewriter, loc, reshapedK, permutedType, qkvPermutation);
    Value value =
        createPermute(rewriter, loc, reshapedV, permutedType, qkvPermutation);

    // Optionally transpose K: [batch, num_heads, seq, head_size] ->
    //                         [batch, num_heads, head_size, seq].
    Value key = permutedK;
    if (transposeKey) {
      SmallVector<int64_t> transposedKShape = {
          batchSize, static_cast<int64_t>(numHeads), headSize, seqSize};
      auto transposedKType =
          RankedTensorType::get(transposedKShape, elementType);
      key = createPermute(rewriter, loc, permutedK, transposedKType,
                          {0, 1, 3, 2});
    }

    rewriter.replaceOp(op, {query, key, value});
    return success();
  }

private:
  // Helper to create a slice operation using ttir.slice_static.
  Value createSlice(ConversionPatternRewriter &rewriter, Location loc,
                    Value input, RankedTensorType resultType,
                    ArrayRef<int64_t> begins, ArrayRef<int64_t> ends) const {
    llvm::SmallVector<mlir::Attribute> beginsAttr, endsAttr, stepsAttr;

    for (size_t i = 0; i < begins.size(); ++i) {
      beginsAttr.push_back(rewriter.getI32IntegerAttr(begins[i]));
      endsAttr.push_back(rewriter.getI32IntegerAttr(ends[i]));
      stepsAttr.push_back(rewriter.getI32IntegerAttr(1));
    }

    return rewriter.create<ttir::SliceStaticOp>(
        loc, resultType, input, rewriter.getArrayAttr(beginsAttr),
        rewriter.getArrayAttr(endsAttr), rewriter.getArrayAttr(stepsAttr));
  }

  // Helper to create a reshape operation using ttir.reshape.
  Value createReshape(ConversionPatternRewriter &rewriter, Location loc,
                      Value input, RankedTensorType resultType) const {
    auto newShape = resultType.getShape();
    llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());
    return rewriter.create<ttir::ReshapeOp>(
        loc, resultType, input, rewriter.getI32ArrayAttr(newShapeI32));
  }

  // Helper to create a permute operation using ttir.permute.
  Value createPermute(ConversionPatternRewriter &rewriter, Location loc,
                      Value input, RankedTensorType resultType,
                      ArrayRef<int64_t> permutation) const {
    return rewriter.create<ttir::PermuteOp>(loc, resultType, input,
                                            permutation);
  }
};
} // namespace

namespace {
struct NegativePadOpDecompositionPattern
    : public OpConversionPattern<ttir::PadOp> {
  using OpConversionPattern<ttir::PadOp>::OpConversionPattern;

  // Decomposes ttir.pad operation with negative padding into: ttir.slice_static
  // and ttir.pad with positive padding issue in tt-metal
  // https://github.com/tenstorrent/tt-metal/issues/37475

  LogicalResult
  matchAndRewrite(ttir::PadOp op, ttir::PadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<int32_t> padding = adaptor.getPadding();
    bool needSlice = llvm::any_of(padding, [](int32_t p) { return p < 0; });
    bool needPad = llvm::any_of(padding, [](int32_t p) { return p > 0; });

    Value input = adaptor.getInput();
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto inputShape = inputType.getShape();

    if (!needSlice) {
      return failure();
    }

    SmallVector<int32_t> sliceBegins(inputShape.size(), 0);
    SmallVector<int32_t> sliceEnds(inputShape.begin(), inputShape.end());
    SmallVector<int32_t> sliceSteps(inputShape.size(), 1);

    // Adjust slice parameters for dimensions with negative padding.
    for (size_t i = 0; i < inputShape.size(); i++) {
      int64_t padLow = padding[2 * i];
      int64_t padHigh = padding[2 * i + 1];

      if (padLow < 0) {
        sliceBegins[i] = std::abs(padLow);
      }
      if (padHigh < 0) {
        sliceEnds[i] = inputShape[i] - std::abs(padHigh);
      }
    }

    // Compute the slice result type.
    RankedTensorType sliceResultType;
    // Intermediate type: sliced shape.
    SmallVector<int64_t> slicedShape;
    for (size_t i = 0; i < inputShape.size(); i++) {
      slicedShape.push_back(sliceEnds[i] - sliceBegins[i]);
    }
    sliceResultType =
        RankedTensorType::get(slicedShape, inputType.getElementType());

    ttir::SliceStaticOp sliceOp = rewriter.create<ttir::SliceStaticOp>(
        op.getLoc(), sliceResultType, input,
        rewriter.getI32ArrayAttr(sliceBegins),
        rewriter.getI32ArrayAttr(sliceEnds),
        rewriter.getI32ArrayAttr(sliceSteps));
    input = sliceOp.getResult();

    if (needPad) {
      // Build padding with negative values zeroed out.
      SmallVector<int32_t> posPadding = llvm::to_vector(
          llvm::map_range(padding, [](int32_t p) { return std::max(p, 0); }));

      ttir::PadOp padOp = rewriter.create<ttir::PadOp>(
          op.getLoc(), op.getType(), input,
          rewriter.getDenseI32ArrayAttr(posPadding), adaptor.getValue());
      input = padOp.getResult();
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};
} // namespace

void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter,
                                             DecompMode decompConfig) {
  patterns.add<PoolingToFullOp<ttir::MaxPool2dOp>>(typeConverter, ctx);
  patterns.add<PoolingToFullOp<ttir::AvgPool2dOp>>(typeConverter, ctx);
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<GatherToSliceRepeatConcatConversionPattern>(typeConverter, ctx);
  patterns.add<GatherToEmbeddingConversionPattern>(typeConverter, ctx);
  patterns.add<SelectToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<ArangeForceLastDimensionPattern>(typeConverter, ctx);
  patterns.add<DotGeneralToMatmulConversionPattern>(typeConverter, ctx);
  patterns.add<ReductionAndPattern>(typeConverter, ctx);
  patterns.add<BatchNormInferencePattern>(typeConverter, ctx);
  patterns.add<BatchNormTrainingPattern>(typeConverter, ctx);
  patterns.add<QuantizeOpPattern>(typeConverter, ctx);
  patterns.add<DequantizeOpPattern>(typeConverter, ctx);
  patterns.add<RequantizeOpPattern>(typeConverter, ctx);
  patterns.add<ReductionProdPattern>(typeConverter, ctx);
  patterns.add<ReverseOpConversionPattern>(typeConverter, ctx);
  patterns.add<ArgMaxPattern>(typeConverter, ctx);
  patterns.add<EmbeddingToGatherConversionPattern>(typeConverter, ctx);
  patterns.add<SplitQueryKeyValueAndSplitHeadsDecompositionPattern>(
      typeConverter, ctx);
  patterns.add<NegativePadOpDecompositionPattern>(typeConverter, ctx);

  // Configure which ReductionPattern to add base on the configuration
  switch (decompConfig) {
  case DecompMode::CPUFallback:
    patterns.add<ReductionOrPattern>(typeConverter, ctx);
    break;
  case DecompMode::TTNN:
  case DecompMode::TTMetal:
    patterns.add<ReductionOrTTNNPattern>(typeConverter, ctx);
    break;
  }
  patterns.add<ConvChannelLastDecompositionPattern<ttir::Conv2dOp>>(
      typeConverter, ctx);
  patterns.add<ConvChannelLastDecompositionPattern<ttir::ConvTranspose2dOp>>(
      typeConverter, ctx);
}

} // namespace mlir::tt
