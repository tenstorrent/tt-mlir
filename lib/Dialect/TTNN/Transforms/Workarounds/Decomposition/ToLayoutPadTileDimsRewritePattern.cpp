// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ToLayoutPadTileDimsRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult ToLayoutPadTileDimsRewritePattern::matchAndRewrite(
    ToLayoutOp srcOp, PatternRewriter &rewriter) const {

  // Only apply when converting TO tile layout.
  if (srcOp.getLayout() != ttnn::Layout::Tile) {
    return failure();
  }

  auto inputType = mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  ArrayRef<int64_t> shape = inputType.getShape();
  int64_t rank = inputType.getRank();

  if (rank < 2) {
    return failure();
  }

  // Only apply to HEIGHT_SHARDED L1 tensors. TTNN handles tile padding
  // internally for interleaved and host tensors.
  // Metal issue: https://github.com/tenstorrent/tt-metal/issues/30541
  if (!inputType.getEncoding()) {
    return failure();
  }
  auto encoding = mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  if (!encoding) {
    return failure();
  }
  if (encoding.getBufferType() != BufferType::L1) {
    return failure();
  }
  auto memLayout = encoding.getMemLayout();
  if (!memLayout || memLayout.getValue() != TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  // Check if the last two dimensions need tile-alignment padding.
  constexpr int64_t tileDim = 32;
  int64_t h = shape[rank - 2];
  int64_t w = shape[rank - 1];
  bool hNeedsPad = (h % tileDim != 0);
  bool wNeedsPad = (w % tileDim != 0);

  if (!hNeedsPad && !wNeedsPad) {
    return failure();
  }

  // Skip guard: if input is already a PadOp, avoid infinite recursion.
  if (srcOp.getInput().getDefiningOp<PadOp>()) {
    return failure();
  }

  // Step 1: Move from sharded L1 to DRAM INTERLEAVED.
  // to_memory_config (ttnn.copy) doesn't support UINT16, so for UINT16 we
  // round-trip through host: from_device → to_device(DRAM).
  auto inputDataType =
      ttcore::elementTypeToDataType(inputType.getElementType());

  auto dramEncoding =
      encoding.withBufferType(BufferType::DRAM)
          .withMemoryLayout(TensorMemoryLayout::Interleaved)
          .withGrid(shape, ttcore::GridAttr::get(srcOp.getContext(), {1, 1}));
  auto dramType =
      RankedTensorType::get(shape, inputType.getElementType(), dramEncoding);

  auto memConfigAttr = MemoryConfigAttr::get(
      srcOp.getContext(),
      TensorMemoryLayoutAttr::get(srcOp.getContext(),
                                  TensorMemoryLayout::Interleaved),
      BufferTypeAttr::get(srcOp.getContext(), BufferType::DRAM),
      /*shard_spec=*/std::nullopt);

  Value currentInput;
  if (inputDataType == ttcore::DataType::UInt16) {
    // from_device: HEIGHT_SHARDED L1 → host (SystemMemory)
    auto hostType = utils::RankedTensorTypeFactory::create(
        inputType, BufferType::SystemMemory);
    Value hostInput = rewriter.create<FromDeviceOp>(srcOp.getLoc(), hostType,
                                                    srcOp.getInput());

    // to_device: host → DRAM INTERLEAVED
    Value device = utils::getOrInsertDevice(rewriter, srcOp);
    currentInput = rewriter.create<ToDeviceOp>(
        srcOp.getLoc(), dramType, hostInput, device, memConfigAttr);
  } else {
    currentInput = rewriter.create<ToMemoryConfigOp>(
        srcOp.getLoc(), dramType, srcOp.getInput(), memConfigAttr);
  }

  // Step 2: Pad to tile-aligned dimensions.
  int64_t paddedH = llvm::divideCeil(h, tileDim) * tileDim;
  int64_t paddedW = llvm::divideCeil(w, tileDim) * tileDim;

  SmallVector<int64_t> paddedShape(shape);
  paddedShape[rank - 2] = paddedH;
  paddedShape[rank - 1] = paddedW;

  SmallVector<int32_t> padding(rank * 2, 0);
  if (hNeedsPad) {
    padding[(rank - 2) * 2 + 1] = paddedH - h;
  }
  if (wNeedsPad) {
    padding[(rank - 1) * 2 + 1] = paddedW - w;
  }

  auto currentInputType = mlir::cast<RankedTensorType>(currentInput.getType());
  auto paddedType =
      utils::RankedTensorTypeFactory::create(currentInputType, paddedShape);

  auto padOp = rewriter.create<PadOp>(
      srcOp.getLoc(), paddedType, currentInput,
      rewriter.getDenseI32ArrayAttr(padding), rewriter.getF32FloatAttr(0.0f),
      /*use_multicore=*/rewriter.getBoolAttr(true),
      /*memory_config=*/nullptr);

  // Step 3: to_layout(TILE) on padded tensor (stay in DRAM).
  // If srcOp performs a dtype conversion (e.g. u16 → si32), the result type
  // must reflect the output dtype so that downstream ops (slice) see
  // consistent element types.
  auto paddedResultType =
      utils::RankedTensorTypeFactory::create(paddedType, ttnn::Layout::Tile);
  if (srcOp.getDtype().has_value()) {
    paddedResultType = utils::RankedTensorTypeFactory::create(
        paddedResultType, srcOp.getDtype().value());
  }
  auto newToLayout = rewriter.create<ToLayoutOp>(
      srcOp.getLoc(), paddedResultType, padOp.getResult(),
      srcOp.getLayoutAttr(), srcOp.getDtypeAttr(),
      /*memory_config=*/memConfigAttr);

  // Step 4: Slice back to original shape.
  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends(shape.begin(), shape.end());
  SmallVector<int32_t> steps(rank, 1);

  auto slicedResultType = utils::RankedTensorTypeFactory::create(
      mlir::cast<RankedTensorType>(newToLayout.getResult().getType()), shape);
  auto sliceOp = rewriter.create<SliceStaticOp>(
      srcOp.getLoc(), slicedResultType, newToLayout.getResult(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(steps));

  // Step 5: Return to the original memory config (HEIGHT_SHARDED L1).
  // to_memory_config (ttnn.copy) doesn't support UINT16, so for UINT16 we
  // round-trip through host: from_device → to_device(target).
  auto origResultType =
      mlir::cast<RankedTensorType>(srcOp.getResult().getType());
  auto origEncoding =
      mlir::dyn_cast<TTNNLayoutAttr>(origResultType.getEncoding());

  ttcore::DeviceAttr deviceAttr =
      ttcore::lookupDevice(srcOp->getBlock()->getParentOp());
  auto returnMemConfigAttr =
      MemoryConfigAttr::get(origEncoding, deviceAttr.getWorkerGrid());

  auto slicedDataType =
      ttcore::elementTypeToDataType(slicedResultType.getElementType());
  Value finalResult;
  if (slicedDataType == ttcore::DataType::UInt16) {
    auto hostType = utils::RankedTensorTypeFactory::create(
        slicedResultType, BufferType::SystemMemory);
    Value hostTensor = rewriter.create<FromDeviceOp>(srcOp.getLoc(), hostType,
                                                     sliceOp.getResult());

    Value device = utils::getOrInsertDevice(rewriter, srcOp);
    finalResult =
        rewriter.create<ToDeviceOp>(srcOp.getLoc(), origResultType, hostTensor,
                                    device, returnMemConfigAttr);
  } else {
    finalResult = rewriter.create<ToMemoryConfigOp>(
        srcOp.getLoc(), origResultType, sliceOp.getResult(),
        returnMemConfigAttr);
  }

  rewriter.replaceOp(srcOp, finalResult);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
