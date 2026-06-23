// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <optional>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDECOMPOSELAYOUTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// This pass lowers a high-level `ttnn.to_layout` into a sequence of primitive
// ops (to_device / from_device / typecast / to_layout / to_memory_config).
// The strategy is intentionally simple and split into two steps:
//
//   Step 1 (layout + dtype): perform the layout (tilize/untilize) and dtype
//     (typecast) changes. If a single `to_layout` can do both (float-family
//     RM -> TILE on device) it is emitted as one op; otherwise a separate
//     `typecast` and `to_layout` are emitted, ordered by the input tilization
//     (input TILE -> typecast then untilize; input RM -> tilize then typecast).
//     The typecast runs on device for any dtype, so as long as one
//     side is device-capable the layout op is ordered to run on that dtype.
//     When neither dtype can tilize/untilize on device (e.g. uint8 <-> uint8),
//     only the tilize/untilize itself round-trips to host; the typecast still
//     runs on device (as a TILE->TILE cast) unless the work is genuinely
//     host-resident (both sides on host, or a CPU-hoisted boundary).
//
//   Step 2 (memory move): a single memory move places the tensor in the output
//     memory. If the input memory is preferred, the move is appended after step
//     1; otherwise it is prepended. When one side is host the move is a
//     to_device / from_device; otherwise it is a to_memory_config.
//
class TTNNDecomposeLayouts
    : public impl::TTNNDecomposeLayoutsBase<TTNNDecomposeLayouts> {

public:
  using impl::TTNNDecomposeLayoutsBase<
      TTNNDecomposeLayouts>::TTNNDecomposeLayoutsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());
    llvm::SmallVector<Operation *> opsToReplace;
    module->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "Found func that didn't have one block!");
      func->walk([&](Operation *op) {
        if (!isa<ttnn::ToLayoutOp>(op)) {
          return;
        }
        opsToReplace.push_back(op);
      });
    });
    for (Operation *op : opsToReplace) {
      if (failed(createLayoutConversionOps(mlir::cast<ttnn::ToLayoutOp>(op),
                                           rewriter))) {
        signalPassFailure();
        return;
      }
      rewriter.eraseOp(op);
    }
  }

private:
  struct LayoutInfo {
    ttnn::BufferType bufferType;
    ttnn::Layout layoutEnum;
    ttcore::DataType dataType;
    ttnn::TensorMemoryLayoutAttr tensorMemoryLayout;
    llvm::SmallVector<int64_t> gridShape;
    ttnn::CoreRangeSetAttr coreRangeSet;

    bool isSharded() const {
      return tensorMemoryLayout &&
             isShardedMemoryLayout(tensorMemoryLayout.getValue());
    }
    bool isL1Sharded() const {
      return bufferType == ttnn::BufferType::L1 && isSharded();
    }
    bool isDramSharded() const {
      return bufferType == ttnn::BufferType::DRAM && isSharded();
    }
    bool isOnHost() const {
      return bufferType == ttnn::BufferType::SystemMemory;
    }
    bool isOnDevice() const { return !isOnHost(); }
    bool isTilized() const { return layoutEnum == ttnn::Layout::Tile; }
  };

  //===--------------------------------------------------------------------===//
  // CPU-hoist boundary detection.
  //===--------------------------------------------------------------------===//

  // True if `value` is the result of a CPU-hoisted function call. Such values
  // live on host; we keep the layout/typecast work on host (before moving to
  // device) to minimize intermediate DRAM/L1 usage.
  bool isOutputFromCPUHoistedFunction(mlir::Value value) const {
    if (auto callOp = value.getDefiningOp<func::CallOp>()) {
      return callOp->hasAttr(ttmlir::utils::g_cpuHoistFuncCallAttrName);
    }
    return false;
  }

  // True if the result of `op` feeds a CPU-hoisted function call. We move to
  // host first and perform the layout/typecast there.
  bool isInputToCPUHoistedFunction(ttnn::ToLayoutOp op) const {
    if (!op.getResult().hasOneUse()) {
      return false;
    }
    if (auto callOp =
            dyn_cast<func::CallOp>(*op.getResult().getUsers().begin())) {
      return callOp->hasAttr(ttmlir::utils::g_cpuHoistFuncCallAttrName);
    }
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Device capability predicates.
  //===--------------------------------------------------------------------===//

  bool canChangeTileLayoutDataTypeOnDevice(ttcore::DataType dataType) const {
    return dataType == ttcore::DataType::BFloat16 ||
           dataType == ttcore::DataType::Float32 ||
           dataType == ttcore::DataType::UInt32 ||
           dataType == ttcore::DataType::UInt16 ||
           dataType == ttcore::DataType::Int32;
  }

  // ttnn.to_layout only performs a real numeric conversion within the float
  // family;
  bool isFloatFamily(ttcore::DataType dataType) const {
    switch (dataType) {
    case ttcore::DataType::UInt32:
    case ttcore::DataType::UInt16:
    case ttcore::DataType::UInt8:
    case ttcore::DataType::Int32:
    case ttcore::DataType::Bool:
      return false;
    default:
      return true;
    }
  }

  //===--------------------------------------------------------------------===//
  // Change predicates and memory preference.
  //===--------------------------------------------------------------------===//

  bool hasLayoutChange(const LayoutInfo &input,
                       const LayoutInfo &output) const {
    return input.layoutEnum != output.layoutEnum;
  }

  bool hasDtypeChange(const LayoutInfo &input, const LayoutInfo &output) const {
    return input.dataType != output.dataType;
  }

  bool hasMemoryChange(const LayoutInfo &input,
                       const LayoutInfo &output) const {
    if (input.bufferType != output.bufferType) {
      return true;
    }
    // Same buffer type. If both are on host there is nothing more to compare.
    if (input.isOnHost()) {
      return false;
    }
    if (input.tensorMemoryLayout != output.tensorMemoryLayout) {
      return true;
    }
    // Reshard if either the virtual grid or the physical placement (CRS)
    // changes. Same CRS can host different virtual grids and same gridShape can
    // sit on different CRSes; both demand a reshard.
    if (input.isSharded() && output.isSharded()) {
      return input.gridShape != output.gridShape ||
             input.coreRangeSet != output.coreRangeSet;
    }
    return false;
  }

  // Decide whether the layout/dtype change runs before the memory move (in the
  // input memory) or after it (in the output memory). The memory move is always
  // placed last unless tt-metal's sharded (un)tilize constraints (or a host
  // boundary) require it first.
  bool layoutChangeRunsBeforeMemoryMove(const LayoutInfo &input,
                                        const LayoutInfo &output,
                                        bool needLayout) const {
    // Host boundary: move onto the device first (host input), or finish the
    // work on device before reading it back (host output).
    if (input.isOnHost() || output.isOnHost()) {
      return !input.isOnHost();
    }
    if (needLayout) {
      // Tilize (RM -> TILE): run the tilize first, then reshard. Tilizing a
      // tensor that was first resharded into the (sharded) output memory can
      // produce a non-tile-aligned physical shard that tt-metal rejects.
      if (output.isTilized()) {
        return true;
      }
      // Untilize (TILE -> RM): deshard first when the input is L1-sharded (or
      // when moving L1 -> DRAM), then untilize the interleaved tensor;
      // otherwise untilize in place and move afterwards.
      bool deshardFirst =
          input.isL1Sharded() || (input.bufferType == ttnn::BufferType::L1 &&
                                  output.bufferType == ttnn::BufferType::DRAM);
      return !deshardFirst;
    }
    // Pure typecast / memory move: typecast in place, memory config last.
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Layout info extraction.
  //===--------------------------------------------------------------------===//

  std::pair<LayoutInfo, LayoutInfo>
  getInputOutputLayouts(ttnn::ToLayoutOp op) const {
    LayoutInfo input, output;

    auto inputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(op.getInput().getType().getEncoding());
    auto outputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(op.getResult().getType().getEncoding());

    input.bufferType = inputLayoutAttr.getBufferType();
    output.bufferType = outputLayoutAttr.getBufferType();

    input.layoutEnum = inputLayoutAttr.getLayout();
    output.layoutEnum = outputLayoutAttr.getLayout();

    input.dataType = inputLayoutAttr.getDataType();
    output.dataType = outputLayoutAttr.getDataType();

    input.tensorMemoryLayout = inputLayoutAttr.getMemLayout();
    output.tensorMemoryLayout = outputLayoutAttr.getMemLayout();

    input.gridShape =
        llvm::SmallVector<int64_t>(inputLayoutAttr.getGridShape());
    output.gridShape =
        llvm::SmallVector<int64_t>(outputLayoutAttr.getGridShape());

    input.coreRangeSet = inputLayoutAttr.getCoreRangeSet();
    output.coreRangeSet = outputLayoutAttr.getCoreRangeSet();

    TTMLIR_DEBUG(ttmlir::LogComponent::General,
                 "Decompose layouts pass for op {} \nInput layout: {} \nOutput "
                 "layout: {} \n",
                 op, inputLayoutAttr, outputLayoutAttr);

    return {input, output};
  }

  //===--------------------------------------------------------------------===//
  // Primitive op creation. These unconditionally create the requested op.
  //===--------------------------------------------------------------------===//

  template <typename OpType, typename... Args>
  mlir::Value createOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                       RankedTensorType resultType, mlir::Value input,
                       Args &&...args) const {
    rewriter.setInsertionPoint(op);
    return rewriter.create<OpType>(op.getLoc(), resultType, input,
                                   std::forward<Args>(args)...);
  }

  // Encode `target`'s memory (buffer / memory layout / grid / CRS) onto the
  // current tensor type, keeping the current layout and dtype.
  RankedTensorType withTargetMemory(RankedTensorType currentType,
                                    const LayoutInfo &target) const {
    TTNNLayoutAttr encoding = TTNNLayoutAttr::Builder(currentType)
                                  .setBufferType(target.bufferType)
                                  .setMemoryLayout(target.tensorMemoryLayout)
                                  .setGridShape(target.gridShape)
                                  .setCoreRangeSet(target.coreRangeSet)
                                  .build();
    return utils::RankedTensorTypeFactory::create(currentType, encoding);
  }

  mlir::Value createToDeviceOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                               mlir::Value current,
                               const LayoutInfo &target) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType = withTargetMemory(currentType, target);
    mlir::Value device = utils::getOrInsertDevice(rewriter, op);
    return createOp<ttnn::ToDeviceOp>(op, rewriter, resultType, current,
                                      device);
  }

  mlir::Value createFromDeviceOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                 mlir::Value current) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType = utils::RankedTensorTypeFactory::create(
        currentType, ttnn::BufferType::SystemMemory);
    return createOp<ttnn::FromDeviceOp>(op, rewriter, resultType, current);
  }

  // Create a to_layout op. When `dtype` is set, the result also changes data
  // type (a fused tilize/untilize + cast).
  mlir::Value
  createToLayoutOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                   mlir::Value current, ttnn::Layout targetLayout,
                   std::optional<ttcore::DataType> dtype = std::nullopt) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType =
        utils::RankedTensorTypeFactory::create(currentType, targetLayout);
    if (dtype) {
      resultType = utils::RankedTensorTypeFactory::create(resultType, *dtype);
    }
    return createOp<ttnn::ToLayoutOp>(op, rewriter, resultType, current);
  }

  mlir::Value createTypecastOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                               mlir::Value current,
                               ttcore::DataType targetDtype) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType =
        utils::RankedTensorTypeFactory::create(currentType, targetDtype);
    return createOp<ttnn::TypecastOp>(op, rewriter, resultType, current);
  }

  mlir::Value createToMemoryConfigOp(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                     mlir::Value current,
                                     const LayoutInfo &target) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType = withTargetMemory(currentType, target);
    return createOp<ttnn::ToMemoryConfigOp>(op, rewriter, resultType, current);
  }

  //===--------------------------------------------------------------------===//
  // tt-metal#30541 sharded tilize pad workaround.
  //===--------------------------------------------------------------------===//

  // ttnn.tilize does not support HEIGHT_SHARDED L1 tensors with non-tile-
  // aligned shard shapes. Detect that case from the tensor about to be tilized.
  bool needsShardedTilizePadWorkaround(RankedTensorType type) const {
    auto encoding = mlir::dyn_cast<TTNNLayoutAttr>(type.getEncoding());
    if (!encoding || encoding.getBufferType() != ttnn::BufferType::L1) {
      return false;
    }
    auto memLayout = encoding.getMemLayout();
    if (!memLayout ||
        memLayout.getValue() != TensorMemoryLayout::HeightSharded) {
      return false;
    }
    ArrayRef<int64_t> shape = type.getShape();
    int64_t rank = type.getRank();
    if (rank < 2) {
      return false;
    }
    return (shape[rank - 2] % TILE_HEIGHT != 0) ||
           (shape[rank - 1] % TILE_WIDTH != 0);
  }

  // Workaround for tt-metal#30541: unshard -> pad -> tilize -> slice.
  // Returns a DRAM INTERLEAVED, tilized, original-shape tensor. The caller is
  // responsible for any subsequent typecast and the final memory config.
  mlir::Value handleShardedTilizeWithPadding(ttnn::ToLayoutOp op,
                                             IRRewriter &rewriter,
                                             mlir::Value current,
                                             ttnn::Layout targetLayout) const {
    auto inputType = mlir::cast<RankedTensorType>(current.getType());
    TTNNLayoutAttr inputEncoding =
        mlir::cast<TTNNLayoutAttr>(inputType.getEncoding());
    ArrayRef<int64_t> shape = inputType.getShape();
    int64_t rank = inputType.getRank();

    // Step 1: Unshard to DRAM INTERLEAVED.
    TTNNLayoutAttr dramEncoding =
        TTNNLayoutAttr::Builder(inputEncoding, shape)
            .setBufferType(BufferType::DRAM)
            .setMemoryLayout(TensorMemoryLayout::Interleaved)
            .setGridShape({1, 1})
            .build();
    RankedTensorType dramType =
        RankedTensorType::get(shape, inputType.getElementType(), dramEncoding);

    rewriter.setInsertionPoint(op);
    current = rewriter.create<ToMemoryConfigOp>(op.getLoc(), dramType, current);

    // Step 2: Pad to tile-aligned dimensions.
    int64_t h = shape[rank - 2];
    int64_t w = shape[rank - 1];
    int64_t paddedH =
        llvm::divideCeil(h, static_cast<int64_t>(TILE_HEIGHT)) * TILE_HEIGHT;
    int64_t paddedW =
        llvm::divideCeil(w, static_cast<int64_t>(TILE_WIDTH)) * TILE_WIDTH;

    SmallVector<int64_t> paddedShape(shape);
    paddedShape[rank - 2] = paddedH;
    paddedShape[rank - 1] = paddedW;

    SmallVector<int32_t> padding(rank * 2, 0);
    if (h % TILE_HEIGHT != 0) {
      padding[(rank - 2) * 2 + 1] = paddedH - h;
    }
    if (w % TILE_WIDTH != 0) {
      padding[(rank - 1) * 2 + 1] = paddedW - w;
    }

    auto currentInputType = mlir::cast<RankedTensorType>(current.getType());
    auto paddedType =
        utils::RankedTensorTypeFactory::create(currentInputType, paddedShape);
    current = rewriter.create<PadOp>(
        op.getLoc(), paddedType, current,
        rewriter.getDenseI32ArrayAttr(padding), rewriter.getF32FloatAttr(0.0f),
        /*use_multicore=*/rewriter.getBoolAttr(true));

    // Step 3: Tilize on padded DRAM tensor.
    RankedTensorType paddedTiledType = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(current.getType()), targetLayout);
    current = rewriter.create<ttnn::ToLayoutOp>(op.getLoc(), paddedTiledType,
                                                current);

    // Step 4: Slice back to original shape.
    SmallVector<int32_t> begins(rank, 0);
    SmallVector<int32_t> ends(shape.begin(), shape.end());
    SmallVector<int32_t> steps(rank, 1);
    RankedTensorType slicedType = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(current.getType()), shape);
    current = rewriter.create<SliceStaticOp>(
        op.getLoc(), slicedType, current, rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

    return current;
  }

  //===--------------------------------------------------------------------===//
  // Cross-orientation reshard guard.
  //===--------------------------------------------------------------------===//

  // A TILE sharded->sharded reshard that crosses L1<->DRAM silently corrupts
  // (or raises) for certain orientation changes; route it through an
  // interleaved buffer instead.
  bool needsReshardViaInterleaved(TTNNLayoutAttr currentEncoding,
                                  const LayoutInfo &target) const {
    if (currentEncoding.getLayout() != ttnn::Layout::Tile) {
      return false;
    }
    auto currentML = currentEncoding.getMemLayout();
    bool currentSharded =
        currentML && isShardedMemoryLayout(currentML.getValue());
    if (!currentSharded || !target.isSharded()) {
      return false;
    }
    BufferType cb = currentEncoding.getBufferType();
    BufferType tb = target.bufferType;
    bool crossesL1Dram = (cb == BufferType::L1 && tb == BufferType::DRAM) ||
                         (cb == BufferType::DRAM && tb == BufferType::L1);
    if (!crossesL1Dram) {
      return false;
    }
    TensorMemoryLayout co = currentML.getValue();
    TensorMemoryLayout to = target.tensorMemoryLayout.getValue();
    bool eitherHeightSharded = co == TensorMemoryLayout::HeightSharded ||
                               to == TensorMemoryLayout::HeightSharded;
    bool bothWidthSharded = co == TensorMemoryLayout::WidthSharded &&
                            to == TensorMemoryLayout::WidthSharded;
    return (eitherHeightSharded && co != to) || bothWidthSharded;
  }

  // Move the current tensor into `target`'s memory. Picks to_device /
  // from_device when a host side is involved, otherwise to_memory_config.
  mlir::Value createMemoryMove(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                               mlir::Value current,
                               const LayoutInfo &target) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    auto currentEncoding =
        mlir::cast<TTNNLayoutAttr>(currentType.getEncoding());
    bool currentOnHost =
        currentEncoding.getBufferType() == ttnn::BufferType::SystemMemory;
    bool targetOnHost = target.isOnHost();

    if (currentOnHost && !targetOnHost) {
      return createToDeviceOp(op, rewriter, current, target);
    }
    if (!currentOnHost && targetOnHost) {
      return createFromDeviceOp(op, rewriter, current);
    }
    if (currentOnHost && targetOnHost) {
      return current;
    }

    // Device to device.
    if (needsReshardViaInterleaved(currentEncoding, target)) {
      TTNNLayoutAttr interEncoding =
          TTNNLayoutAttr::Builder(currentEncoding, currentType.getShape())
              .setBufferType(target.bufferType)
              .setMemoryLayout(TensorMemoryLayout::Interleaved)
              .setGridShape({1, 1})
              .build();
      RankedTensorType interType = RankedTensorType::get(
          currentType.getShape(), currentType.getElementType(), interEncoding);
      current =
          createOp<ttnn::ToMemoryConfigOp>(op, rewriter, interType, current);
    }
    return createToMemoryConfigOp(op, rewriter, current, target);
  }

  // True if the current tensor already lives in `target`'s memory.
  bool memoryMatchesTarget(mlir::Value current,
                           const LayoutInfo &target) const {
    auto encoding = mlir::cast<TTNNLayoutAttr>(
        mlir::cast<RankedTensorType>(current.getType()).getEncoding());
    if (encoding.getBufferType() != target.bufferType) {
      return false;
    }
    if (target.isOnHost()) {
      return true;
    }
    if (encoding.getMemLayout() != target.tensorMemoryLayout) {
      return false;
    }
    bool currentSharded =
        encoding.getMemLayout() &&
        isShardedMemoryLayout(encoding.getMemLayout().getValue());
    if (currentSharded && target.isSharded()) {
      if (ArrayRef<int64_t>(encoding.getGridShape()) !=
          ArrayRef<int64_t>(target.gridShape)) {
        return false;
      }
      if (encoding.getCoreRangeSet() != target.coreRangeSet) {
        return false;
      }
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Layout + dtype transformations.
  //===--------------------------------------------------------------------===//

  // Apply just the layout change (tilize/untilize), keeping the current dtype.
  mlir::Value createLayoutStep(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                               mlir::Value current,
                               const LayoutInfo &output) const {
    if (output.isTilized() &&
        needsShardedTilizePadWorkaround(
            mlir::cast<RankedTensorType>(current.getType()))) {
      return handleShardedTilizeWithPadding(op, rewriter, current,
                                            output.layoutEnum);
    }
    return createToLayoutOp(op, rewriter, current, output.layoutEnum);
  }

  // Perform the layout and dtype changes. `onDevice` indicates whether the
  // chain runs on device (enables the fused layout+dtype op for the float
  // family) or on host (order is irrelevant, so typecast then layout).
  mlir::Value
  layoutAndDtypeTransformations(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                mlir::Value current, const LayoutInfo &input,
                                const LayoutInfo &output, bool onDevice) const {
    bool needLayout = hasLayoutChange(input, output);
    bool needDtype = hasDtypeChange(input, output);
    if (!needLayout && !needDtype) {
      return current;
    }

    bool tilizing = needLayout && output.isTilized();
    bool padWorkaround = onDevice && tilizing &&
                         needsShardedTilizePadWorkaround(
                             mlir::cast<RankedTensorType>(current.getType()));

    // A single to_layout can do both layout and dtype only for a float-family
    // RM -> TILE conversion on device (and not when the pad workaround is in
    // play, which displaces the tensor to DRAM interleaved).
    if (onDevice && needLayout && needDtype && tilizing && !input.isTilized() &&
        isFloatFamily(input.dataType) && isFloatFamily(output.dataType) &&
        !padWorkaround) {
      return createToLayoutOp(op, rewriter, current, output.layoutEnum,
                              output.dataType);
    }

    if (needLayout && !needDtype) {
      return createLayoutStep(op, rewriter, current, output);
    }
    if (!needLayout && needDtype) {
      return createTypecastOp(op, rewriter, current, output.dataType);
    }

    // Both changes, emitted as separate ops.
    if (!onDevice) {
      // On host the order is irrelevant; typecast first then change layout.
      current = createTypecastOp(op, rewriter, current, output.dataType);
      return createLayoutStep(op, rewriter, current, output);
    }

    // On device the tilize/untilize must run on a device-capable dtype. The
    // typecast runs on device for any dtype, so order the two ops to place the
    // layout change on whichever side (input or output dtype) is
    // device-capable, keeping the typecast on device too.
    if (input.isTilized()) {
      // Untilize. Untilize on the output dtype (typecast first) when it can run
      // on device; otherwise untilize on the input dtype first, then typecast.
      if (canChangeTileLayoutDataTypeOnDevice(output.dataType)) {
        current = createTypecastOp(op, rewriter, current, output.dataType);
        return createLayoutStep(op, rewriter, current, output);
      }
      current = createLayoutStep(op, rewriter, current, output);
      return createTypecastOp(op, rewriter, current, output.dataType);
    }
    // Tilize. Tilize on the input dtype (tilize first) when it can run on
    // device; otherwise typecast to the output dtype first, then tilize.
    if (canChangeTileLayoutDataTypeOnDevice(input.dataType)) {
      current = createLayoutStep(op, rewriter, current, output);
      return createTypecastOp(op, rewriter, current, output.dataType);
    }
    current = createTypecastOp(op, rewriter, current, output.dataType);
    return createLayoutStep(op, rewriter, current, output);
  }

  // The tilize/untilize itself must run on host when neither the input nor the
  // output dtype can (un)tilize on device. (The typecast may still run on
  // device; see canKeepTypecastOnDevice.) The tilize/untilize can run on either
  // the input or output dtype depending on op ordering, so the work stays on
  // device as long as at least one side is device-capable.
  bool layoutChangeNeedsHost(const LayoutInfo &input, const LayoutInfo &output,
                             bool needLayout) const {
    if (!needLayout) {
      return false;
    }
    return !canChangeTileLayoutDataTypeOnDevice(input.dataType) &&
           !canChangeTileLayoutDataTypeOnDevice(output.dataType);
  }

  // For a device-involved transfer whose tilize/untilize must run on host, the
  // typecast can still run on device when the device side holds the tensor in
  // TILE form for an (always-supported) TILE->TILE typecast:
  //   - untilize: the (TILE) input is on device, so typecast there first;
  //   - tilize:   the (TILE) output is on device, so typecast there last.
  bool canKeepTypecastOnDevice(const LayoutInfo &input,
                               const LayoutInfo &output) const {
    return input.isTilized() ? input.isOnDevice() : output.isOnDevice();
  }

  // Run the tilize/untilize on host while keeping the typecast on device. The
  // device typecast is a TILE->TILE op; the host op only changes layout (the
  // dtype is preserved across the host round-trip). Requires
  // canKeepTypecastOnDevice(input, output).
  mlir::Value hostLayoutWithDeviceTypecast(ttnn::ToLayoutOp op,
                                           IRRewriter &rewriter,
                                           mlir::Value current,
                                           const LayoutInfo &input,
                                           const LayoutInfo &output) const {
    bool needDtype = hasDtypeChange(input, output);
    if (input.isTilized()) {
      // Untilize: typecast on device (TILE) first, untilize on host, then land
      // in the output memory.
      if (needDtype) {
        current = createTypecastOp(op, rewriter, current, output.dataType);
      }
      current = createFromDeviceOp(op, rewriter, current);
      current = createLayoutStep(op, rewriter, current, output);
      if (output.isOnDevice()) {
        current = createToDeviceOp(op, rewriter, current, output);
      }
      return current;
    }
    // Tilize: tilize on host (dtype preserved), move to the output memory, then
    // typecast on device (TILE).
    if (input.isOnDevice()) {
      current = createFromDeviceOp(op, rewriter, current);
    }
    current = createLayoutStep(op, rewriter, current, output);
    current = createToDeviceOp(op, rewriter, current, output);
    if (needDtype) {
      current = createTypecastOp(op, rewriter, current, output.dataType);
    }
    return current;
  }

  //===--------------------------------------------------------------------===//
  // Orchestration.
  //===--------------------------------------------------------------------===//

  mlir::LogicalResult createLayoutConversionOps(ttnn::ToLayoutOp op,
                                                IRRewriter &rewriter) const {
    auto [input, output] = getInputOutputLayouts(op);

    bool needLayout = hasLayoutChange(input, output);
    bool needDtype = hasDtypeChange(input, output);
    bool needMemory = hasMemoryChange(input, output);

    if (!needLayout && !needDtype && !needMemory) {
      op->emitError(
          "Redundant ttnn::ToLayoutOp - no ttnn layout ops needed, this may be "
          "due to the forcing of tile/row major layouts.");
      return failure();
    }

    mlir::Value current = op.getInput();

    // CPU-hoisted boundaries keep the layout/typecast work on host.
    bool cpuHoistOnHost = isOutputFromCPUHoistedFunction(current) ||
                          isInputToCPUHoistedFunction(op);
    bool bothOnHost = input.isOnHost() && output.isOnHost();
    bool layoutNeedsHost = layoutChangeNeedsHost(input, output, needLayout);

    // The typecast runs on host only for genuinely host-resident work: both
    // sides on host, a CPU-hoisted boundary, or a host (un)tilize whose device
    // side cannot hold the TILE tensor for an on-device typecast.
    bool fullyOnHost =
        bothOnHost || cpuHoistOnHost ||
        (layoutNeedsHost && !canKeepTypecastOnDevice(input, output));

    if (fullyOnHost) {
      // Bring the tensor to host (if needed), do the layout/dtype changes
      // there, then move to the output memory (if it is on device). to_device
      // carries the full target memory config, so no separate to_memory_config
      // is required.
      if (input.isOnDevice()) {
        current = createFromDeviceOp(op, rewriter, current);
      }
      current = layoutAndDtypeTransformations(op, rewriter, current, input,
                                              output, /*onDevice=*/false);
      if (output.isOnDevice()) {
        current = createToDeviceOp(op, rewriter, current, output);
      }
    } else if (layoutNeedsHost) {
      // The tilize/untilize must run on host, but the typecast stays on device.
      current =
          hostLayoutWithDeviceTypecast(op, rewriter, current, input, output);
    } else if (layoutChangeRunsBeforeMemoryMove(input, output, needLayout)) {
      // The layout/dtype changes run in the input memory; the memory move (if
      // any) comes after.
      current = layoutAndDtypeTransformations(op, rewriter, current, input,
                                              output, /*onDevice=*/true);
      if (!memoryMatchesTarget(current, output)) {
        current = createMemoryMove(op, rewriter, current, output);
      }
    } else {
      // Move to the output memory first, then run the layout/dtype changes.
      current = createMemoryMove(op, rewriter, current, output);
      current = layoutAndDtypeTransformations(op, rewriter, current, input,
                                              output, /*onDevice=*/true);
    }

    op.getResult().replaceAllUsesWith(current);
    return success();
  }
};
} // namespace mlir::tt::ttnn
