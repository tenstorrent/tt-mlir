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
#include "llvm/Support/raw_ostream.h"

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
    int64_t dim = op.getDim();
    int32_t begin = adaptor.getBegin();
    int32_t end = adaptor.getEnd();
    int64_t dimSize = inputType.getDimSize(dim);

    // Normalize negative bounds so downstream slice folding composes canonical
    // coordinates instead of raw Python-style negative offsets.
    if (begin < 0) {
      begin += dimSize;
    }
    if (end < 0) {
      end += dimSize;
    }

    llvm::SmallVector<mlir::Attribute, 4> begins, ends, steps;

    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim) {
        begins.push_back(rewriter.getI32IntegerAttr(begin));
        ends.push_back(rewriter.getI32IntegerAttr(end));
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

// Decomposing Reverse Op into Embedding Op.
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
    auto inputType = op.getInput().getType();
    ArrayRef<int64_t> shape = inputType.getShape();
    int64_t rank = shape.size();
    Value current = adaptor.getInput();
    Location loc = op.getLoc();

    // Build permutation: reversing dims first (sorted), then non-reversing
    // dims.
    SmallVector<int64_t> permutation(dimensions);
    llvm::sort(permutation);
    for (int64_t i = 0; i < rank; i++) {
      if (!llvm::is_contained(dimensions, i)) {
        permutation.push_back(i);
      }
    }

    // Step 1: Permute reversing dims to front.
    auto permutedShape = ttmlir::utils::applyPermutation(shape, permutation);
    current = rewriter.create<ttir::PermuteOp>(
        loc,
        RankedTensorType::get(permutedShape, inputType.getElementType(),
                              inputType.getEncoding()),
        current, permutation);

    // Step 2: Reshape to 2D [N_reversing, N_non_reversing].
    int32_t nReversing = std::accumulate(
        permutedShape.begin(), permutedShape.begin() + dimensions.size(),
        int32_t{1}, std::multiplies<>());
    int32_t nNonReversing =
        std::accumulate(permutedShape.begin() + dimensions.size(),
                        permutedShape.end(), int64_t{1}, std::multiplies<>());

    SmallVector<int64_t> flatShape{nReversing, nNonReversing};
    auto shapeAttr = rewriter.getI32ArrayAttr(
        SmallVector<int32_t>(flatShape.begin(), flatShape.end()));
    auto flatType = RankedTensorType::get(flatShape, inputType.getElementType(),
                                          inputType.getEncoding());
    current =
        rewriter.create<ttir::ReshapeOp>(loc, flatType, current, shapeAttr);

    // Step 3: Create reversed linear indices [N-1, N-2, ..., 0].
    SmallVector<int32_t> indices(nReversing);
    for (int32_t i = 0; i < nReversing; i++) {
      indices[i] = nReversing - 1 - i;
    }
    auto idxType = RankedTensorType::get(
        {nReversing}, rewriter.getIntegerType(32, /*isSigned=*/true));
    auto idxAttr = DenseIntElementsAttr::get(idxType, indices);
    Value idxConst = rewriter.create<ttir::ConstantOp>(loc, idxType, idxAttr);

    // Step 4: EmbeddingOp to reorder rows.
    current =
        rewriter.create<ttir::EmbeddingOp>(loc, flatType, idxConst, current);

    // Step 5: Reshape back to permuted shape.
    auto permShapeAttr = rewriter.getI32ArrayAttr(
        SmallVector<int32_t>(permutedShape.begin(), permutedShape.end()));
    auto permType = RankedTensorType::get(
        permutedShape, inputType.getElementType(), inputType.getEncoding());
    current =
        rewriter.create<ttir::ReshapeOp>(loc, permType, current, permShapeAttr);

    // Step 6: Inverse permute back to original shape.
    SmallVector<int64_t> invPerm =
        ttmlir::utils::inversePermutation(permutation);
    current = rewriter.create<ttir::PermuteOp>(
        loc,
        RankedTensorType::get(shape, inputType.getElementType(),
                              inputType.getEncoding()),
        current, invPerm);

    rewriter.replaceOp(op, current);

    return success();
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
    // For rank-3 tensors [N, C, S], if S is tile-aligned (according to the
    // default tile width), split it into [N, C, S/tileWidth, tileWidth]
    // rather than appending a trailing 1 ([N, C, S, 1]) to maintain a
    // fully-packed tile layout.
    auto defaultTileShape = ttcore::TileType::getDefaultShape();
    int64_t tileWidth = defaultTileShape[1];
    if (rank == 3 && tileWidth != 0 && reshapedShape.back() % tileWidth == 0) {
      int64_t S = reshapedShape.back();
      reshapedShape.back() = S / tileWidth;
      reshapedShape.push_back(tileWidth);
    } else {
      reshapedShape.append(4 - rank, 1);
    }
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

//===----------------------------------------------------------------------===//
// MoeGPTDecodeOp decomposition
//===----------------------------------------------------------------------===//

namespace {
// Decompose `ttir.moe_gpt_decode` into the three underlying TTIR ops that
// directly map to the tt-metal MoE kernels:
//   1. `ttir.all_to_all_dispatch_metadata` — token dispatch across the ring.
//   2. `ttir.moe_gpt` — fused expert compute (tilize + gate/up + SwiGLU + w2).
//   3. `ttir.selective_reduce_combine` — sparsify and combine back.
//
// The op is kept as a placeholder during sharding propagation. At this point
// we materialize the three-stage pipeline so that the existing TTIR→TTNN
// patterns can lower each stage. Shape adapters (reshape/permute/typecast)
// bridge the 4D decode layout used at the composite boundary with the 2D/3D
// kernel-native layouts expected by the individual TTIR ops.
//
// Decomposes `ttir.moe_gpt_decode` into the dispatch / moe_gpt / combine
// TTIR op triple, mirroring tt-metal's fused decode pipeline. The frontend
// is required to supply preprocessed 6D fused kernel weights via
// `fused_w0_w1` / `fused_w2`; those are bound directly to ttir.moe_gpt's 6D
// weight inputs. The tt-metal kernel consumes only the fused layout, so
// the unfused `gate_up_proj` / `down_proj` experts are not part of this
// op's operand list (a separate non-composite path handles prefill).
struct MoeGPTDecodeDecompositionPattern
    : public OpConversionPattern<ttir::MoeGPTDecodeOp> {
  using OpConversionPattern<ttir::MoeGPTDecodeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MoeGPTDecodeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto hiddenType =
        cast<RankedTensorType>(adaptor.getHiddenStates().getType());
    auto indicesType =
        cast<RankedTensorType>(adaptor.getTopkIndices().getType());
    auto scoresType = cast<RankedTensorType>(adaptor.getTopkScores().getType());
    auto dispatchMappingType =
        cast<RankedTensorType>(adaptor.getDispatchMapping().getType());
    auto moeGptMappingType =
        cast<RankedTensorType>(adaptor.getMoeGptMapping().getType());
    auto fusedW0W1Type =
        cast<RankedTensorType>(adaptor.getFusedW0W1().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    Type bf16 = rewriter.getBF16Type();
    Type ui32 = rewriter.getIntegerType(32, /*isSigned=*/false);

    // Extract logical dimensions from operand shapes.
    // The mapping tensor follows tt-metal's fused decode convention
    // [1, 1, D_total, E] with linearized global device ids (see
    // `build_expert_mapping_linearized` in the Python frontend and
    // `gen_expert_mapping` in tt-metal). D_total spans *all* mesh devices;
    // D_ring (the dispatch-axis ring size) is read from the op attribute.
    // E_local comes from the fused kernel weight layout at
    // fused_w0_w1 shape[2] (see tt-metal `moe_gpt` kernel).
    int64_t B = hiddenType.getShape()[0];
    int64_t S = hiddenType.getShape()[2];
    int64_t H = hiddenType.getShape()[3];
    int64_t KSel = indicesType.getShape()[3];
    int64_t Dtotal = dispatchMappingType.getShape()[2];
    int64_t Eglobal = dispatchMappingType.getShape()[3];
    int64_t Elocal = fusedW0W1Type.getShape()[2];

    int64_t numDevices = static_cast<int64_t>(op.getNumDevices());
    int64_t clusterAxis = static_cast<int64_t>(op.getClusterAxis());
    int64_t numExpertsPerTok = static_cast<int64_t>(op.getNumExpertsPerTok());
    int64_t numExperts = static_cast<int64_t>(op.getNumExperts());
    int64_t Dring = numDevices;
    int64_t Ttokens = Dring * B * S;
    (void)Dtotal;

    auto i32Arr = [&](ArrayRef<int64_t> dims) {
      SmallVector<int32_t> v;
      v.reserve(dims.size());
      for (int64_t d : dims) {
        v.push_back(static_cast<int32_t>(d));
      }
      return rewriter.getI32ArrayAttr(v);
    };

    // Step 1: Reshape routing inputs from the composite 4D layout
    // [B, 1, S, X] to the dispatch 4D layout [1, 1, B*S, X]. Because dim 1 is
    // already size 1 and X is the last dim, this is a pure reshape with the
    // row-major element order preserved.
    auto dispHiddenType =
        RankedTensorType::get({1, 1, B * S, H}, hiddenType.getElementType());
    Value dispHidden = rewriter.create<ttir::ReshapeOp>(
        loc, dispHiddenType, adaptor.getHiddenStates(),
        i32Arr({1, 1, B * S, H}));

    auto dispIndicesType = RankedTensorType::get({1, 1, B * S, KSel},
                                                 indicesType.getElementType());
    Value dispIndices = rewriter.create<ttir::ReshapeOp>(
        loc, dispIndicesType, adaptor.getTopkIndices(),
        i32Arr({1, 1, B * S, KSel}));

    auto dispScoresType =
        RankedTensorType::get({1, 1, B * S, KSel}, scoresType.getElementType());
    Value dispScores = rewriter.create<ttir::ReshapeOp>(
        loc, dispScoresType, adaptor.getTopkScores(),
        i32Arr({1, 1, B * S, KSel}));

    // Step 2: Forward dispatch_mapping directly — the Python side now
    // produces [1, 1, D_total, E] linearized mapping matching
    // AllToAllDispatchMetadataOp's expected [1, 1, D, E] layout
    // (the TTIR→TTNN conversion reshapes it to the [D_total, E] layout the
    // tt-metal kernel requires).
    Value dispMapping = adaptor.getDispatchMapping();

    // Step 3: Emit `ttir.all_to_all_dispatch_metadata`.
    auto dispatchedType = RankedTensorType::get({1, Ttokens, H}, bf16);
    auto dispatchedIdxType =
        RankedTensorType::get({1, Ttokens, KSel}, indicesType.getElementType());
    auto dispatchedScoresType = RankedTensorType::get({1, Ttokens, KSel}, bf16);

    auto dispatchOp = rewriter.create<ttir::AllToAllDispatchMetadataOp>(
        loc, TypeRange{dispatchedType, dispatchedIdxType, dispatchedScoresType},
        dispHidden, dispIndices, dispScores, dispMapping,
        rewriter.getI64IntegerAttr(numDevices),
        rewriter.getI64IntegerAttr(clusterAxis));

    // Step 4: Adapt dispatch outputs to moe_gpt inputs (2D layout, ui16
    // indices/mapping).
    auto moeInputType = RankedTensorType::get({Ttokens, H}, bf16);
    Value moeInput = rewriter.create<ttir::ReshapeOp>(
        loc, moeInputType, dispatchOp.getDispatched(), i32Arr({Ttokens, H}));

    // `ttir.moe_gpt` verifier only checks rank (>=2 for indices, no constraint
    // on mapping element type), and test_moe_ccls.py passes through si32/bf16
    // indices without explicit ui16 casts. Emitting `ttir.typecast` to ui16 is
    // unnecessary and breaks later constraint queries on targets where ui16
    // device buffers are unsupported; keep native element types.
    auto dispatchedIdx2DType =
        RankedTensorType::get({Ttokens, KSel}, indicesType.getElementType());
    Value moeIndices = rewriter.create<ttir::ReshapeOp>(
        loc, dispatchedIdx2DType, dispatchOp.getIndices(),
        i32Arr({Ttokens, KSel}));

    auto moeScoresType = RankedTensorType::get({Ttokens, KSel}, bf16);
    Value moeScores = rewriter.create<ttir::ReshapeOp>(
        loc, moeScoresType, dispatchOp.getScores(), i32Arr({Ttokens, KSel}));

    // moe_gpt device op reads mapping_shape[-1] as experts_total. Use
    // [D_total, E] to mirror tt-metal's `tt_moe_gpt_mapping` layout
    // (generated by `gen_expert_mapping` in fused_decode.py).
    auto moeMapFlatType = RankedTensorType::get(
        {Dtotal, Eglobal}, moeGptMappingType.getElementType());
    Value moeMapping = rewriter.create<ttir::ReshapeOp>(
        loc, moeMapFlatType, adaptor.getMoeGptMapping(),
        i32Arr({Dtotal, Eglobal}));

    // Step 5: Bind the preprocessed 6D fused kernel weights supplied by the
    // frontend. These are required operands on ttir.moe_gpt_decode (see
    // TTIROps.td) because the tt-metal moe_gpt kernel only consumes the
    // fused layout.
    Value w0w1 = adaptor.getFusedW0W1();
    Value w2 = adaptor.getFusedW2();

    // Step 6: Emit `ttir.moe_gpt`.
    // Output sizes follow compute_output_specs() in moe_gpt_device_operation.
    // L1_ALIGN = 16 bytes on Wormhole.
    constexpr int64_t kTileSize = 32;
    auto alignU32 = [](int64_t nBytes) -> int64_t {
      return ((nBytes + 15) / 16) * 16 / 4;
    };
    int64_t tcElems = alignU32(Elocal * 4);
    int64_t actRowStride = alignU32((2 * Elocal + 1) * 4);
    int64_t actElems = (Ttokens + 1) * actRowStride;
    int64_t etRowElems = (Ttokens + 1) * alignU32(4);

    // Worker-core grid count is arch-specific (9*8=72 on Wormhole). The kernel
    // allocates tilize outputs across every worker core. We hard-code 72 here
    // because the TTIR pass has no direct system-desc access; if the target
    // differs this shape can be adjusted in a follow-up.
    constexpr int64_t kNumWorkerCores = 72;

    auto tokenCountsType = RankedTensorType::get({1, tcElems}, ui32);
    auto actRecordsType = RankedTensorType::get({1, actElems}, ui32);
    auto tokenIndicesType = RankedTensorType::get({Elocal, etRowElems}, ui32);
    auto tilizeOutType =
        RankedTensorType::get({kNumWorkerCores, 2, kTileSize, H}, bf16);
    auto tilizeOutRmType = tilizeOutType;

    auto moeGptOp = rewriter.create<ttir::MoeGptOp>(
        loc,
        TypeRange{tokenCountsType, actRecordsType, tokenIndicesType,
                  tilizeOutType, tilizeOutRmType},
        moeInput, moeIndices, moeScores, moeMapping, w0w1, w2,
        /*output_height_shard_dim=*/rewriter.getUI32IntegerAttr(4),
        /*output_width_shard_dim=*/rewriter.getUI32IntegerAttr(3),
        /*hidden_size=*/rewriter.getUI32IntegerAttr(H),
        /*cluster_axis=*/rewriter.getUI32IntegerAttr(clusterAxis));

    // Step 7: Emit `ttir.selective_reduce_combine` producing the original
    // decode result type directly.
    // `batch_size` must equal the post-dispatch total token count per cluster
    // (= Dring * B). The tt-metal kernel derives per-device tokens via
    // `total_tokens = batch_size * seq_size` for stride checks and
    // `total_tokens_per_device = batch_size * seq_size / num_devices_cluster`
    // for the output shape. Passing only B (pre-dispatch) mis-sizes the stride
    // against `activation_records` which is allocated with (Dring*B*S + 1)
    // rows in moe_gpt's compute_output_specs.
    auto combineOp = rewriter.create<ttir::SelectiveReduceCombineOp>(
        loc, resultType, moeGptOp.getTilizeOutRm(),
        moeGptOp.getActivationRecords(), moeGptOp.getTokenIndices(),
        moeGptOp.getTokenCounts(),
        /*hidden_size=*/rewriter.getUI32IntegerAttr(H),
        /*batch_size=*/rewriter.getUI32IntegerAttr(Dring * B),
        /*seq_size=*/rewriter.getUI32IntegerAttr(S),
        /*select_experts_k=*/rewriter.getUI32IntegerAttr(numExpertsPerTok),
        /*experts=*/rewriter.getUI32IntegerAttr(numExperts));

    // Step 8: Emit TP-axis `ttir.all_reduce` to finish the MoE reduction.
    // Mirrors tt-metal's fused_decode.py which wraps combine with
    // `ttnn.all_reduce(cluster_axis=1, topology=Ring, num_links=4)` after the
    // score-weighted sum. Since expert weights are sharded along the
    // non-dispatch cluster axis, each rank's combine output is only a partial
    // sum over its local experts; without this reduction every TP rank would
    // retain only its partial contribution and final logits would be wrong.
    // Placing the all_reduce here (on the raw combine output, with the K
    // dimension still present) is equivalent to placing it after the external
    // score-weighted sum because addition commutes with the sum over K.
    uint32_t tpClusterAxis = (clusterAxis == 0) ? 1 : 0;
    auto allReduceOp = rewriter.create<ttir::AllReduceOp>(
        loc, resultType, combineOp.getResult(), ttcore::ReduceType::Sum,
        tpClusterAxis);

    rewriter.replaceOp(op, allReduceOp.getResult());
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
  patterns.add<MoeGPTDecodeDecompositionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
