// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LinearOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

static bool isBatchedLinearOp(ttnn::LinearOp linearOp) {
  RankedTensorType inputBType = linearOp.getB().getType();
  auto inputBShape = inputBType.getShape();
  int64_t rank = inputBShape.size();

  // if rank <= 2, it cannot be batched. Return false.
  // Check if batched: any dimension before the last 2 has size > 1.
  // i.e. <1x3x128x32xbf16> is batched because of 3.
  // i.e. <1x1x128x32xbf16> is not batched because all dims before last 2 are 1.
  // i.e. <128xbf16> is not batched because rank < 2.
  return rank > 2 && llvm::any_of(inputBShape.drop_back(2),
                                  [](int64_t dim) { return dim > 1; });
}

// The fused bias kernel only broadcasts row 0 of the bias tile. Every dimension
// except the last (feature) dimension must be 1, otherwise the extra rows are
// silently ignored and results are incorrect.
static bool hasNonUnitNonFeatureDims(llvm::ArrayRef<int64_t> shape) {
  return shape.size() > 1 &&
         llvm::any_of(shape.drop_back(1), [](int64_t dim) { return dim > 1; });
}

// Calculate the output shape of a matmul operation following tt-metal's logic.
// Reference: ttnn/cpp/ttnn/operations/matmul/matmul.cpp
static SmallVector<int64_t>
computeMatmulOutputShape(llvm::ArrayRef<int64_t> shapeA, bool transposeA,
                         llvm::ArrayRef<int64_t> shapeB, bool transposeB) {
  int64_t rankA = shapeA.size();
  int64_t rankB = shapeB.size();

  SmallVector<int64_t> outputShape;

  // Handle rank 1 cases
  if (rankA == 1 && rankB == 1) {
    // vector dot vector -> scalar (but represented as 1D tensor with size 1)
    outputShape.push_back(1);
    return outputShape;
  }

  if (rankA == 1) {
    // vector-matrix: (K,) x (..., K, N) -> (..., N)
    // Result shape is all batch dims from B plus the last dim
    outputShape.append(shapeB.begin(), shapeB.end() - 2);
    outputShape.push_back(transposeB ? shapeB[rankB - 2] : shapeB[rankB - 1]);
    return outputShape;
  }

  if (rankB == 1) {
    // matrix-vector: (..., M, K) x (K,) -> (..., M)
    // Result shape is all dims from A except the last (contraction) dim
    if (transposeA) {
      // If A is transposed, the contraction dim is second-to-last, keep last
      outputShape.append(shapeA.begin(), shapeA.end() - 2);
      outputShape.push_back(shapeA[rankA - 1]);
    } else {
      // Normal case: contraction dim is last, keep all but last
      outputShape.append(shapeA.begin(), shapeA.end() - 1);
    }
    return outputShape;
  }

  // Both inputs are at least rank 2
  SmallVector<int64_t> batchShapeA(shapeA.begin(), shapeA.end() - 2);
  SmallVector<int64_t> batchShapeB(shapeB.begin(), shapeB.end() - 2);
  mlir::OpTrait::util::getBroadcastedShape(batchShapeA, batchShapeB,
                                           outputShape);

  // Matmul inner dims: (…, p, q) x (…, q, r) -> (…, p, r)
  if (transposeA) {
    outputShape.push_back(shapeA[rankA - 1]);
  } else {
    outputShape.push_back(shapeA[rankA - 2]);
  }

  if (transposeB) {
    outputShape.push_back(shapeB[rankB - 2]);
  } else {
    outputShape.push_back(shapeB[rankB - 1]);
  }

  return outputShape;
}

// Keep LinearOp only when the bias is safe for tt-metal's fused-bias linear
// path:
//   - RHS/B is not batched.
//   - Bias is effectively a row bias, i.e. every non-feature dimension is 1.
//     The fused bias kernel broadcasts only row 0 of the bias tile, so keeping
//     shapes such as [H, N] with H > 1 would silently ignore the extra rows.
//   - Bias last dimension matches the output feature dimension.
static bool canKeepBiasFusedInLinear(ttnn::LinearOp linearOp,
                                     llvm::ArrayRef<int64_t> matmulShape) {
  if (!linearOp.getBias()) {
    return true;
  }

  if (isBatchedLinearOp(linearOp)) {
    return false;
  }

  llvm::ArrayRef<int64_t> biasShape = linearOp.getBias().getType().getShape();
  if (biasShape.empty()) {
    return false;
  }

  if (hasNonUnitNonFeatureDims(biasShape)) {
    return false;
  }

  if (biasShape.back() != matmulShape.back()) {
    return false;
  }

  return true;
}

// Rewrite Linear op into matmul + add when tt-metal cannot keep the bias fused.
// Follows
// third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul.cpp.
LogicalResult
LinearOpRewritePattern::matchAndRewrite(ttnn::LinearOp srcOp,
                                        PatternRewriter &rewriter) const {

  if (!srcOp.getBias()) {
    return failure();
  }

  RankedTensorType inputAType = srcOp.getA().getType();
  RankedTensorType inputBType = srcOp.getB().getType();
  RankedTensorType outputType = srcOp.getResult().getType();

  // Compute matmul output shape
  SmallVector<int64_t> matmulShape =
      computeMatmulOutputShape(inputAType.getShape(), srcOp.getTransposeA(),
                               inputBType.getShape(), srcOp.getTransposeB());

  if (canKeepBiasFusedInLinear(srcOp, matmulShape)) {
    return failure();
  }

  // Create matmul output type
  auto outputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());
  auto matmulOutputType =
      utils::RankedTensorTypeFactory::create(outputType, matmulShape);

  auto dataTypeAttr = mlir::tt::ttcore::DataTypeAttr::get(
      rewriter.getContext(), outputEncoding.getDataType());

  MatmulOp matmulOp = rewriter.create<ttnn::MatmulOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_decomp_matmul"),
      matmulOutputType, srcOp.getA(), srcOp.getB(), srcOp.getTransposeA(),
      srcOp.getTransposeB(), /*matmul_program_config=*/nullptr,
      /*activation=*/nullptr,
      /*compute_config=*/srcOp.getComputeConfigAttr());

  // Step 2: Create Add operation with bias.
  llvm::SmallVector<int64_t> addShape;
  mlir::OpTrait::util::getBroadcastedShape(matmulOp.getType().getShape(),
                                           srcOp.getBias().getType().getShape(),
                                           addShape);
  auto addOutputType =
      utils::RankedTensorTypeFactory::create(outputType, addShape);
  AddOp addOp = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_decomp_add"),
      addOutputType, matmulOp.getResult(), srcOp.getBias(),
      /*dtype=*/dataTypeAttr,
      /*memory_config=*/ttnn::MemoryConfigAttr());

  // Step 3: Reshape the add result back to original output shape.
  // Reshape op will be no-op if addOp output shape is same as original LinearOp
  // output shape.
  ReshapeOp finalReshape = ttir_to_ttnn::utils::generateReshape(
      addOp.getResult(), srcOp.getType().getShape(), rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp->getLoc(), "_decomp_reshape"));

  // Step 4: If original linear op had activation, apply it to the reshape op
  // result.
  if (srcOp.getActivation()) {
    std::string activationStr = srcOp.getActivation()->str();

    // Map activation string to operation name
    StringAttr opName;
    if (activationStr == "relu") {
      opName = StringAttr::get(rewriter.getContext(), "ttnn.relu");
    } else if (activationStr == "sigmoid") {
      opName = StringAttr::get(rewriter.getContext(), "ttnn.sigmoid");
    } else if (activationStr == "gelu") {
      opName = StringAttr::get(rewriter.getContext(), "ttnn.gelu");
    } else if (activationStr == "tanh") {
      opName = StringAttr::get(rewriter.getContext(), "ttnn.tanh");
    } else if (activationStr == "silu") {
      opName = StringAttr::get(rewriter.getContext(), "ttnn.silu");
    } else {
      matmulOp.emitError()
          << "Unsupported activation type in LinearOp decomposition: "
          << activationStr;
      return failure();
    }

    // Create activation op using generic Operation::create
    Operation *activationOp =
        rewriter.create(ttmlir::utils::appendLocationSuffix(
                            srcOp.getLoc(), "_decomp_" + activationStr),
                        opName,
                        /*operands=*/ValueRange{finalReshape.getResult()},
                        /*types=*/TypeRange{outputType});

    rewriter.replaceOp(srcOp, activationOp->getResult(0));
  } else {
    rewriter.replaceOp(srcOp, finalReshape.getResult());
  }

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
