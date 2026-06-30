// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeComputeRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <utility>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
MoeComputeRewritePattern::matchAndRewrite(MoeComputeOp srcOp,
                                          PatternRewriter &rewriter) const {
  if (failed(reshardTilizeInputs(srcOp, rewriter))) {
    return failure();
  }
  // matmul_output (result 4) re-perceives tilize_output (result 3), so the
  // routing/tilize types must be set first.
  setRoutingAndTilizeOutputTypes(srcOp, rewriter);
  setMatmulOutputType(srcOp, rewriter);
  return success();
}

LogicalResult
MoeComputeRewritePattern::reshardTilizeInputs(MoeComputeOp srcOp,
                                              PatternRewriter &rewriter) const {
  // tt-metal's moe_compute kernel globally-allocates the expert-indices/scores
  // circular buffers against a single backing buffer whose data must live on
  // the tilize drain core; validate() never checks this, so we must reshard
  // those two inputs to L1 HEIGHT_SHARDED on the drain core or the kernel reads
  // garbage L1.
  auto indicesType = mlir::cast<RankedTensorType>(
      srcOp.getTilizeExpertIndicesTensor().getType());
  auto indicesEncoding = mlir::cast<TTNNLayoutAttr>(indicesType.getEncoding());

  // Idempotency: once the indices input is L1 HEIGHT_SHARDED we are done.
  if (indicesEncoding.getBufferType() == BufferType::L1 &&
      indicesEncoding.getMemLayout() &&
      indicesEncoding.getMemLayout().getValue() ==
          TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  MLIRContext *ctx = srcOp.getContext();

  // The drain core is the first entry of tt-metal moe_compute
  // get_layout().max_tilize_cores that fits the device's logical worker grid:
  // the program factory filters that list by compute_with_storage_grid_size and
  // takes index 0 (moe_compute_program_factory.cpp). Replicate that here —
  // incl. the {x,y} candidate order — so the reshard lands on the kernel's
  // actual drain core even on harvested grids that drop the leading (y=9)
  // entry.
  ttcore::SystemDescAttr sysDesc = ttcore::getCurrentScopeSystemDesc(srcOp);
  ttcore::ChipDescAttr chip = sysDesc.getChipDesc(0);
  // ChipDescAttr grid is [y, x] (flatbuffer Dim2d), from compute_with_storage.
  llvm::ArrayRef<int64_t> chipGrid = chip.getGrid();
  int64_t gridY = chipGrid[0], gridX = chipGrid[1];
  bool isBlackhole = chip.getArch().getValue() == ttcore::Arch::Blackhole;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> tilizeCandidates =
      isBlackhole ? llvm::SmallVector<std::pair<int64_t, int64_t>, 4>{{10, 9},
                                                                      {10, 8},
                                                                      {9, 9},
                                                                      {9, 8}}
                  : llvm::SmallVector<std::pair<int64_t, int64_t>, 4>{
                        {6, 9}, {6, 8}, {5, 9}, {5, 8}};
  int64_t drainX = -1, drainY = -1;
  for (const auto &[x, y] : tilizeCandidates) {
    if (x < gridX && y < gridY) {
      drainX = x;
      drainY = y;
      break;
    }
  }
  if (drainX < 0) {
    return rewriter.notifyMatchFailure(
        srcOp, "no moe_compute tilize drain core fits the worker grid");
  }

  // The expert-indices/scores CBs are globally allocated against a single
  // backing buffer that must live on the drain core.
  CoreRangeSetAttr tilizeDrainCoreRangeSet = CoreRangeSetAttr::get(
      ctx, CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, drainX, drainY),
                              CoreCoordAttr::get(ctx, drainX, drainY)));

  // Insert a ttnn.to_memory_config converting `oldInput` to
  // L1 HEIGHT_SHARDED ROW_MAJOR on the drain core.
  auto reshardToTilize = [&](Value oldInput) -> Value {
    auto t = mlir::cast<RankedTensorType>(oldInput.getType());
    auto seed = mlir::cast<TTNNLayoutAttr>(t.getEncoding());
    auto newEncoding = TTNNLayoutAttr::Builder(seed, t.getShape())
                           .setBufferType(BufferType::L1)
                           .setMemoryLayout(TensorMemoryLayout::HeightSharded)
                           .setLayout(Layout::RowMajor)
                           .setGridShape({1, 1})
                           .setCoreRangeSet(tilizeDrainCoreRangeSet)
                           .build();
    auto newType =
        RankedTensorType::get(t.getShape(), t.getElementType(), newEncoding);
    return rewriter.create<ttnn::ToLayoutOp>(srcOp.getLoc(), newType, oldInput);
  };

  Value newIndices = reshardToTilize(srcOp.getTilizeExpertIndicesTensor());
  Value newScores = reshardToTilize(srcOp.getTilizeExpertScoresTensor());

  rewriter.modifyOpInPlace(srcOp, [&]() {
    srcOp.getTilizeExpertIndicesTensorMutable().assign(newIndices);
    srcOp.getTilizeExpertScoresTensorMutable().assign(newScores);
  });
  return success();
}

// If a refined result feeds directly into a func.return, the enclosing
// function's result type must be refreshed to match. Other consumers read the
// type through use-def, so they don't need an explicit fix-up.
static void refreshEnclosingFuncReturnType(mlir::Value refinedResult) {
  for (mlir::OpOperand &use : refinedResult.getUses()) {
    auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(use.getOwner());
    if (!retOp) {
      continue;
    }
    auto funcOp = retOp->getParentOfType<mlir::func::FuncOp>();
    if (!funcOp) {
      continue;
    }
    llvm::SmallVector<mlir::Type> newResultTypes(
        funcOp.getFunctionType().getResults().begin(),
        funcOp.getFunctionType().getResults().end());
    newResultTypes[use.getOperandNumber()] = refinedResult.getType();
    funcOp.setFunctionType(mlir::FunctionType::get(
        funcOp.getContext(), funcOp.getFunctionType().getInputs(),
        newResultTypes));
  }
}

// Builds a RankedTensorType carrying a fresh TTNNLayoutAttr, seeded from
// `result`'s current encoding so tensor-mesh metadata is preserved.
static RankedTensorType buildMoeOutputType(
    mlir::Value result, llvm::ArrayRef<int64_t> shape, Type scalarElementType,
    Layout layout, BufferType bufferType, TensorMemoryLayout memLayout,
    llvm::ArrayRef<int64_t> gridShape, CoreRangeSetAttr coreRangeSet) {
  auto seed = mlir::cast<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(result.getType()).getEncoding());
  TTNNLayoutAttr encoding = TTNNLayoutAttr::Builder(seed, shape)
                                .setElementType(scalarElementType)
                                .setBufferType(bufferType)
                                .setMemoryLayout(memLayout)
                                .setGridShape(gridShape)
                                .setLayout(layout)
                                .setCoreRangeSet(coreRangeSet)
                                .build();
  return RankedTensorType::get(shape, scalarElementType, encoding);
}

void MoeComputeRewritePattern::setRoutingAndTilizeOutputTypes(
    MoeComputeOp srcOp, PatternRewriter &rewriter) const {
  MLIRContext *ctx = srcOp.getContext();

  auto inputType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeInputTensor().getType());
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  // total_tokens = input_shape[0] * input_shape[1] (per the host op): tokens
  // per dispatch device flattened across batch and sequence.
  int64_t totalTokens = inputShape[0] * inputShape[1];
  int64_t hiddenSize = inputShape.back();

  // experts_per_device = div_up(experts, num_devices), where experts is the
  // last dim of the expert-mapping tensor and num_devices is the full mesh
  // device count (mesh_view.num_devices()).
  auto mappingType = mlir::cast<RankedTensorType>(
      srcOp.getTilizeExpertMappingTensor().getType());
  int64_t experts = mappingType.getShape().back();
  ttcore::DeviceAttr device = ttcore::lookupDevice(srcOp.getOperation());
  llvm::ArrayRef<int64_t> meshShape = device.getMeshShape();
  int64_t numDevices = 1;
  for (int64_t dim : meshShape) {
    numDevices *= dim;
  }
  int64_t expertsPerDevice = (experts + numDevices - 1) / numDevices;

  // num_cores spans the full compute-with-storage worker grid; the routing and
  // tilize outputs are HEIGHT_SHARDED one shard per core over that range.
  llvm::ArrayRef<int64_t> workerGrid = device.getWorkerGrid().getShape();
  int64_t gridY = workerGrid[0], gridX = workerGrid[1];
  int64_t numCores = gridY * gridX;
  CoreRangeSetAttr fullGrid = CoreRangeSetAttr::get(
      ctx, CoreRangeAttr::get(ctx, CoreCoordAttr::get(ctx, 0, 0),
                              CoreCoordAttr::get(ctx, gridX - 1, gridY - 1)));

  // l1_alignment (= hal::get_l1_alignment()) is the NoC L1 address alignment.
  int64_t l1Align =
      ttcore::getCurrentScopeSystemDesc(srcOp).getNocL1AddressAlignBytes();
  constexpr int64_t kU32Bytes = 4;
  // tt-metal's shared tilize buffer: TOKEN_SIZE (32) tokens double-buffered.
  constexpr int64_t kTokenSize = 32;
  constexpr int64_t kDoubleBufferSize = 2;

  Type u32 = IntegerType::get(ctx, 32, IntegerType::Unsigned);
  Type bf16 = BFloat16Type::get(ctx);

  // Result 0: per-expert total tokens, replicated per core.
  int64_t perExpertRowBytes =
      ttmlir::utils::alignUp(expertsPerDevice * kU32Bytes, l1Align);
  int64_t perExpertRowElems = (perExpertRowBytes + kU32Bytes - 1) / kU32Bytes;
  RankedTensorType perExpertType = buildMoeOutputType(
      srcOp.getPerExpertTotalTokens(), {numCores, perExpertRowElems}, u32,
      Layout::RowMajor, BufferType::L1, TensorMemoryLayout::HeightSharded,
      {numCores, 1}, fullGrid);

  // Result 1: expert activation, a single L1-interleaved DRAM-style page.
  int64_t activationRowElems = (2 * expertsPerDevice) + 1;
  int64_t activationRowBytes =
      ttmlir::utils::alignUp(activationRowElems * kU32Bytes, l1Align);
  int64_t activationTotalElems = totalTokens * activationRowBytes / kU32Bytes;
  RankedTensorType activationType = buildMoeOutputType(
      srcOp.getExpertActivation(), {1, activationTotalElems}, u32,
      Layout::RowMajor, BufferType::L1, TensorMemoryLayout::Interleaved,
      {gridY, gridX}, CoreRangeSetAttr{});

  // Result 2: expert-to-token indices, 1 page per local expert. Each token slot
  // is 16B-aligned for NoC DMA; +1 element for the -1 terminator.
  int64_t eTRowBytes =
      (totalTokens + 1) * ttmlir::utils::alignUp(kU32Bytes, l1Align);
  int64_t eTRowElems = eTRowBytes / kU32Bytes;
  RankedTensorType eTType = buildMoeOutputType(
      srcOp.getExpertToToken(), {expertsPerDevice, eTRowElems}, u32,
      Layout::RowMajor, BufferType::L1, TensorMemoryLayout::Interleaved,
      {gridY, gridX}, CoreRangeSetAttr{});

  // Result 3: shared tilize buffer, double-buffered TOKEN_SIZE chunks per core.
  llvm::SmallVector<int64_t> tilizeShape = {numCores, kDoubleBufferSize,
                                            kTokenSize, hiddenSize};
  RankedTensorType tilizeType = buildMoeOutputType(
      srcOp.getTilizeOutput(), tilizeShape, bf16, Layout::Tile, BufferType::L1,
      TensorMemoryLayout::HeightSharded, {numCores, 1}, fullGrid);

  rewriter.modifyOpInPlace(srcOp, [&]() {
    srcOp.getPerExpertTotalTokens().setType(perExpertType);
    srcOp.getExpertActivation().setType(activationType);
    srcOp.getExpertToToken().setType(eTType);
    srcOp.getTilizeOutput().setType(tilizeType);
  });

  refreshEnclosingFuncReturnType(srcOp.getPerExpertTotalTokens());
  refreshEnclosingFuncReturnType(srcOp.getExpertActivation());
  refreshEnclosingFuncReturnType(srcOp.getExpertToToken());
  refreshEnclosingFuncReturnType(srcOp.getTilizeOutput());
}

void MoeComputeRewritePattern::setMatmulOutputType(
    MoeComputeOp srcOp, PatternRewriter &rewriter) const {
  // Result 4 (matmul_output) is result 3 (tilize_output)'s buffer re-perceived
  // as row-major: identical shape, element type, and L1 HEIGHT_SHARDED
  // placement, only the layout differs. Relies on
  // setRoutingAndTilizeOutputTypes having already refined tilize_output.
  auto tilizeType =
      mlir::cast<RankedTensorType>(srcOp.getTilizeOutput().getType());
  auto seed = mlir::cast<TTNNLayoutAttr>(tilizeType.getEncoding());
  TTNNLayoutAttr matmulEncoding =
      TTNNLayoutAttr::Builder(seed, tilizeType.getShape())
          .setLayout(Layout::RowMajor)
          .build();
  RankedTensorType matmulType = RankedTensorType::get(
      tilizeType.getShape(), tilizeType.getElementType(), matmulEncoding);

  rewriter.modifyOpInPlace(srcOp, [&]() {
    srcOp.getMatmulOutput().setType(matmulType);
    // compute_only produces no combine output; tt-metal returns matmul_output
    // as result 5 (they alias the same device buffer).
    if (srcOp.getComputeOnly()) {
      srcOp.getCombineOutput().setType(matmulType);
    }
  });

  refreshEnclosingFuncReturnType(srcOp.getMatmulOutput());
  if (srcOp.getComputeOnly()) {
    refreshEnclosingFuncReturnType(srcOp.getCombineOutput());
  }
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
