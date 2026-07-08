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

  bool
  isOnDeviceLayoutChangeSupportedForDataType(ttcore::DataType dataType) const {
    return dataType == ttcore::DataType::BFloat16 ||
           dataType == ttcore::DataType::Float32 ||
           dataType == ttcore::DataType::UInt32 ||
           dataType == ttcore::DataType::UInt16 ||
           dataType == ttcore::DataType::Int32;
  }

  // ttnn.to_layout only performs a real numeric conversion within the float
  // family.
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

  bool hasLayoutChange(TTNNLayoutAttr input, TTNNLayoutAttr output) const {
    return input.getLayout() != output.getLayout();
  }

  bool hasDtypeChange(TTNNLayoutAttr input, TTNNLayoutAttr output) const {
    return input.getDataType() != output.getDataType();
  }

  bool hasMemoryChange(TTNNLayoutAttr input, TTNNLayoutAttr output) const {
    if (input.getBufferType() != output.getBufferType()) {
      return true;
    }
    // Same buffer type. If both are on host there is nothing more to compare.
    if (input.isSystemBufferType()) {
      return false;
    }
    if (input.getMemLayout() != output.getMemLayout()) {
      return true;
    }
    // Reshard if either the virtual grid or the physical placement (CRS)
    // changes. Same CRS can host different virtual grids and same gridShape can
    // sit on different CRSes; both demand a reshard.
    if (input.hasShardedTensorMemoryLayout() &&
        output.hasShardedTensorMemoryLayout()) {
      return input.getGridShape() != output.getGridShape() ||
             input.getCoreRangeSet() != output.getCoreRangeSet();
    }
    return false;
  }

  // Decide whether the layout/dtype change runs before the memory move (in the
  // input memory) or after it (in the output memory). The memory move is always
  // placed last unless tt-metal's sharded (un)tilize constraints (or a host
  // boundary) require it first.
  bool
  shouldLayoutAndDtypeChangesRunBeforeMemoryMove(TTNNLayoutAttr input,
                                                 TTNNLayoutAttr output) const {
    // Host boundary: move onto the device first (host input), or finish the
    // work on device before reading it back (host output).
    if (input.isSystemBufferType() || output.isSystemBufferType()) {
      return !input.isSystemBufferType();
    }
    if (hasLayoutChange(input, output)) {
      // Tilize (RM -> TILE): run the tilize first, then reshard. Tilizing a
      // tensor that was first resharded into the (sharded) output memory can
      // produce a non-tile-aligned physical shard that tt-metal rejects.
      if (output.isTiled()) {
        return true;
      }
      // Untilize (TILE -> RM): deshard first when the input is L1-sharded (or
      // when moving L1 -> DRAM), then untilize the interleaved tensor;
      // otherwise untilize in place and move afterwards.
      bool deshardFirst = input.hasShardedL1TensorMemoryLayout() ||
                          (input.getBufferType() == ttnn::BufferType::L1 &&
                           output.getBufferType() == ttnn::BufferType::DRAM);
      return !deshardFirst;
    }
    // Pure typecast / memory move: typecast in place, memory config last.
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Layout info extraction.
  //===--------------------------------------------------------------------===//

  std::pair<TTNNLayoutAttr, TTNNLayoutAttr>
  getInputOutputLayouts(ttnn::ToLayoutOp op) const {
    auto inputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(op.getInput().getType().getEncoding());
    auto outputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(op.getResult().getType().getEncoding());

    TTMLIR_DEBUG(ttmlir::LogComponent::General,
                 "Decompose layouts pass for op {} \nInput layout: {} \nOutput "
                 "layout: {} \n",
                 op, inputLayoutAttr, outputLayoutAttr);

    return {inputLayoutAttr, outputLayoutAttr};
  }

  //===--------------------------------------------------------------------===//
  // Primitive op creation. These unconditionally create the requested op.
  //===--------------------------------------------------------------------===//

  template <typename OpType, typename... Args>
  mlir::Value createOp(IRRewriter &rewriter, RankedTensorType resultType,
                       mlir::Value input, Args &&...args) const {
    return rewriter.create<OpType>(input.getLoc(), resultType, input,
                                   std::forward<Args>(args)...);
  }

  // Encode `target`'s memory (buffer / memory layout / grid / CRS) onto the
  // current tensor type, keeping the current layout and dtype.
  RankedTensorType withTargetMemory(RankedTensorType currentType,
                                    TTNNLayoutAttr target) const {
    TTNNLayoutAttr encoding = TTNNLayoutAttr::Builder(currentType)
                                  .setBufferType(target.getBufferType())
                                  .setMemoryLayout(target.getMemLayout())
                                  .setGridShape(target.getGridShape())
                                  .setCoreRangeSet(target.getCoreRangeSet())
                                  .build();
    return utils::RankedTensorTypeFactory::create(currentType, encoding);
  }

  mlir::Value createToDeviceOp(IRRewriter &rewriter, mlir::Value current,
                               TTNNLayoutAttr target) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType = withTargetMemory(currentType, target);
    mlir::Value device =
        utils::getOrInsertDevice(rewriter, rewriter.getInsertionBlock());
    return createOp<ttnn::ToDeviceOp>(rewriter, resultType, current, device);
  }

  mlir::Value createFromDeviceOp(IRRewriter &rewriter,
                                 mlir::Value current) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType = utils::RankedTensorTypeFactory::create(
        currentType, ttnn::BufferType::SystemMemory);
    return createOp<ttnn::FromDeviceOp>(rewriter, resultType, current);
  }

  // Create a to_layout op. When `dtype` is set, the result also changes data
  // type (a fused tilize/untilize + cast).
  mlir::Value
  createToLayoutOp(IRRewriter &rewriter, mlir::Value current,
                   ttnn::Layout targetLayout,
                   std::optional<ttcore::DataType> dtype = std::nullopt) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType =
        utils::RankedTensorTypeFactory::create(currentType, targetLayout);
    if (dtype) {
      resultType = utils::RankedTensorTypeFactory::create(resultType, *dtype);
    }
    return createOp<ttnn::ToLayoutOp>(rewriter, resultType, current);
  }

  mlir::Value createTypecastOp(IRRewriter &rewriter, mlir::Value current,
                               ttcore::DataType targetDtype) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType =
        utils::RankedTensorTypeFactory::create(currentType, targetDtype);
    return createOp<ttnn::TypecastOp>(rewriter, resultType, current);
  }

  mlir::Value createToMemoryConfigOp(IRRewriter &rewriter, mlir::Value current,
                                     TTNNLayoutAttr target) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    RankedTensorType resultType = withTargetMemory(currentType, target);
    return createOp<ttnn::ToMemoryConfigOp>(rewriter, resultType, current);
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
  mlir::Value handleShardedTilizeWithPadding(IRRewriter &rewriter,
                                             mlir::Value current,
                                             ttnn::Layout targetLayout) const {
    auto inputType = mlir::cast<RankedTensorType>(current.getType());
    TTNNLayoutAttr inputEncoding =
        mlir::cast<TTNNLayoutAttr>(inputType.getEncoding());
    ArrayRef<int64_t> shape = inputType.getShape();
    int64_t rank = inputType.getRank();
    Location loc = current.getLoc();

    // Step 1: Unshard to DRAM INTERLEAVED.
    TTNNLayoutAttr dramEncoding =
        TTNNLayoutAttr::Builder(inputEncoding, shape)
            .setBufferType(BufferType::DRAM)
            .setMemoryLayout(TensorMemoryLayout::Interleaved)
            .setGridShape({1, 1})
            .build();
    RankedTensorType dramType =
        RankedTensorType::get(shape, inputType.getElementType(), dramEncoding);

    current = rewriter.create<ToMemoryConfigOp>(loc, dramType, current);

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
        loc, paddedType, current, rewriter.getDenseI32ArrayAttr(padding),
        rewriter.getF32FloatAttr(0.0f),
        /*use_multicore=*/rewriter.getBoolAttr(true));

    // Step 3: Tilize on padded DRAM tensor.
    RankedTensorType paddedTiledType = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(current.getType()), targetLayout);
    current = rewriter.create<ttnn::ToLayoutOp>(loc, paddedTiledType, current);

    // Step 4: Slice back to original shape.
    SmallVector<int32_t> begins(rank, 0);
    SmallVector<int32_t> ends(shape.begin(), shape.end());
    SmallVector<int32_t> steps(rank, 1);
    RankedTensorType slicedType = utils::RankedTensorTypeFactory::create(
        mlir::cast<RankedTensorType>(current.getType()), shape);
    current = rewriter.create<SliceStaticOp>(
        loc, slicedType, current, rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));

    return current;
  }

  //===--------------------------------------------------------------------===//
  // Cross-orientation reshard guard.
  //===--------------------------------------------------------------------===//

  // A TILE sharded->sharded reshard that crosses L1<->DRAM silently corrupts
  // (or raises) for certain orientation changes; route it through an
  // interleaved buffer instead.
  // TT-metal issue: https://github.com/tenstorrent/tt-metal/issues/49224
  bool needsReshardViaInterleaved(TTNNLayoutAttr currentEncoding,
                                  TTNNLayoutAttr target) const {
    if (currentEncoding.getLayout() != ttnn::Layout::Tile) {
      return false;
    }
    auto currentML = currentEncoding.getMemLayout();
    bool currentSharded =
        currentML && isShardedMemoryLayout(currentML.getValue());
    if (!currentSharded || !target.hasShardedTensorMemoryLayout()) {
      return false;
    }
    BufferType currentBufferType = currentEncoding.getBufferType();
    BufferType targetBufferType = target.getBufferType();
    bool crossesL1Dram = (currentBufferType == BufferType::L1 &&
                          targetBufferType == BufferType::DRAM) ||
                         (currentBufferType == BufferType::DRAM &&
                          targetBufferType == BufferType::L1);
    if (!crossesL1Dram) {
      return false;
    }
    TensorMemoryLayout currentShardLayout = currentML.getValue();
    TensorMemoryLayout targetShardLayout = target.getMemLayout().getValue();
    // Empirically (see the metal issue) the direct reshard is only safe for a
    // same-orientation height reshard (HS->HS) and block->width (BS->WS).
    // Everything else corrupts or raises, which reduces to two rules:
    //   - a width-sharded source is never safe (WS->{HS,WS,BS}), and
    //   - height-sharding on either side combined with an orientation change
    //     (HS<->WS, HS<->BS) is never safe.
    bool sourceWidthSharded =
        currentShardLayout == TensorMemoryLayout::WidthSharded;
    bool eitherHeightSharded =
        currentShardLayout == TensorMemoryLayout::HeightSharded ||
        targetShardLayout == TensorMemoryLayout::HeightSharded;
    return sourceWidthSharded ||
           (eitherHeightSharded && currentShardLayout != targetShardLayout);
  }

  // Move the current tensor into `target`'s memory. Picks to_device /
  // from_device when a host side is involved, otherwise to_memory_config.
  mlir::Value createMemoryMove(IRRewriter &rewriter, mlir::Value current,
                               TTNNLayoutAttr target) const {
    RankedTensorType currentType =
        mlir::cast<RankedTensorType>(current.getType());
    auto currentEncoding =
        mlir::cast<TTNNLayoutAttr>(currentType.getEncoding());
    bool currentOnHost =
        currentEncoding.getBufferType() == ttnn::BufferType::SystemMemory;
    bool targetOnHost = target.isSystemBufferType();

    if (currentOnHost && !targetOnHost) {
      return createToDeviceOp(rewriter, current, target);
    }
    if (!currentOnHost && targetOnHost) {
      return createFromDeviceOp(rewriter, current);
    }
    if (currentOnHost && targetOnHost) {
      return current;
    }

    // Device to device.
    if (needsReshardViaInterleaved(currentEncoding, target)) {
      TTNNLayoutAttr interEncoding =
          TTNNLayoutAttr::Builder(currentEncoding, currentType.getShape())
              .setBufferType(BufferType::DRAM)
              .setMemoryLayout(TensorMemoryLayout::Interleaved)
              .setGridShape({1, 1})
              .build();
      RankedTensorType interType = RankedTensorType::get(
          currentType.getShape(), currentType.getElementType(), interEncoding);
      current = createOp<ttnn::ToMemoryConfigOp>(rewriter, interType, current);
    }
    return createToMemoryConfigOp(rewriter, current, target);
  }

  //===--------------------------------------------------------------------===//
  // Layout + dtype transformations.
  //===--------------------------------------------------------------------===//

  // Apply just the layout change (tilize/untilize), keeping the current dtype.
  mlir::Value createLayoutStep(IRRewriter &rewriter, mlir::Value current,
                               TTNNLayoutAttr output,
                               bool padWorkaround) const {
    if (padWorkaround) {
      return handleShardedTilizeWithPadding(rewriter, current,
                                            output.getLayout());
    }
    return createToLayoutOp(rewriter, current, output.getLayout());
  }

  // Perform the layout and dtype changes. `onDevice` indicates whether the
  // chain runs on device (enables the fused layout+dtype op for the float
  // family) or on host (order is irrelevant, so typecast then layout).
  mlir::Value layoutAndDtypeTransformations(IRRewriter &rewriter,
                                            mlir::Value current,
                                            TTNNLayoutAttr input,
                                            TTNNLayoutAttr output,
                                            bool onDevice) const {
    bool needsLayoutChange = hasLayoutChange(input, output);
    bool needDtype = hasDtypeChange(input, output);
    if (!needsLayoutChange && !needDtype) {
      return current;
    }

    bool tilizing = needsLayoutChange && output.isTiled();
    bool padWorkaround = onDevice && tilizing &&
                         needsShardedTilizePadWorkaround(
                             mlir::cast<RankedTensorType>(current.getType()));

    // A single to_layout can do both layout and dtype only for a float-family
    // RM -> TILE conversion on device (and not when the pad workaround is in
    // play, which displaces the tensor to DRAM interleaved).
    if (onDevice && needsLayoutChange && needDtype && tilizing &&
        !input.isTiled() && isFloatFamily(input.getDataType()) &&
        isFloatFamily(output.getDataType()) && !padWorkaround) {
      return createToLayoutOp(rewriter, current, output.getLayout(),
                              output.getDataType());
    }

    if (needsLayoutChange && !needDtype) {
      return createLayoutStep(rewriter, current, output, padWorkaround);
    }
    if (!needsLayoutChange && needDtype) {
      // ttnn.typecast on device produces wrong results in row-major layout, so
      // never typecast a row-major tensor on device: tilize around the cast so
      // it runs on a TILE tensor, then untilize back.
      if (onDevice && !output.isTiled() &&
          isOnDeviceLayoutChangeSupportedForDataType(input.getDataType()) &&
          isOnDeviceLayoutChangeSupportedForDataType(output.getDataType())) {
        current = createToLayoutOp(rewriter, current, ttnn::Layout::Tile);
        current = createTypecastOp(rewriter, current, output.getDataType());
        return createToLayoutOp(rewriter, current, ttnn::Layout::RowMajor);
      }
      return createTypecastOp(rewriter, current, output.getDataType());
    }

    // Both changes, emitted as separate ops.
    if (!onDevice) {
      // On host the order is irrelevant; typecast first then change layout.
      current = createTypecastOp(rewriter, current, output.getDataType());
      return createLayoutStep(rewriter, current, output, padWorkaround);
    }

    // On device the tilize/untilize must run on a device-capable dtype. The
    // typecast runs on device for any dtype, so order the two ops to place the
    // layout change on whichever side (input or output dtype) is
    // device-capable, keeping the typecast on device too.
    if (input.isTiled()) {
      // Untilize on the output dtype (typecast first) when it can run on
      // device.
      if (isOnDeviceLayoutChangeSupportedForDataType(output.getDataType())) {
        current = createTypecastOp(rewriter, current, output.getDataType());
        return createLayoutStep(rewriter, current, output, padWorkaround);
      }
      // Otherwise untilize on the input dtype first, then typecast.
      current = createLayoutStep(rewriter, current, output, padWorkaround);
      return createTypecastOp(rewriter, current, output.getDataType());
    }
    // Tilize on the input dtype (tilize first) when it can run on device.
    if (isOnDeviceLayoutChangeSupportedForDataType(input.getDataType())) {
      current = createLayoutStep(rewriter, current, output, padWorkaround);
      return createTypecastOp(rewriter, current, output.getDataType());
    }
    // Otherwise typecast to the output dtype first, then tilize.
    current = createTypecastOp(rewriter, current, output.getDataType());
    return createLayoutStep(rewriter, current, output, padWorkaround);
  }

  // The tilize/untilize itself must run on host when neither the input nor the
  // output dtype can (un)tilize on device. (The typecast may still run on
  // device; see canKeepTypecastOnDevice.) The tilize/untilize can run on either
  // the input or output dtype depending on op ordering, so the work stays on
  // device as long as at least one side is device-capable.
  bool layoutChangeNeedsHost(TTNNLayoutAttr input,
                             TTNNLayoutAttr output) const {
    if (!hasLayoutChange(input, output)) {
      return false;
    }
    return !isOnDeviceLayoutChangeSupportedForDataType(input.getDataType()) &&
           !isOnDeviceLayoutChangeSupportedForDataType(output.getDataType());
  }

  // For a device-involved transfer whose tilize/untilize must run on host, the
  // typecast can still run on device when the device side holds the tensor in
  // TILE form for an (always-supported) TILE->TILE typecast:
  //   - untilize: the (TILE) input is on device, so typecast there first;
  //   - tilize:   the (TILE) output is on device, so typecast there last.
  bool canKeepTypecastOnDevice(TTNNLayoutAttr input,
                               TTNNLayoutAttr output) const {
    return input.isTiled() ? input.isDeviceBufferType()
                           : output.isDeviceBufferType();
  }

  // A combined layout+dtype change run on device emits a ttnn.typecast on a
  // row-major tensor (which ttnn miscomputes) unless the tensor can be kept in
  // TILE across the cast:
  //   - untilize (input TILE): typecast TILE->TILE first, then untilize in the
  //     OUTPUT dtype - which needs the output dtype to (un)tilize on device;
  //   - tilize (input RM): tilize in the INPUT dtype, then typecast TILE->TILE
  //     - which needs the input dtype to (un)tilize on device (the fused
  //     float-family tilize+cast single op also avoids the row-major cast).
  // When neither holds, the on-device path casts a row-major tensor. Pushing
  // the (un)tilize to host (with the typecast staying on device as TILE->TILE)
  // avoids it. Complements layoutChangeNeedsHost, which covers the (un)tilize
  // that must run on host regardless of any dtype change.
  bool onDeviceTypecastWouldBeRowMajor(TTNNLayoutAttr input,
                                       TTNNLayoutAttr output) const {
    if (!hasLayoutChange(input, output) || !hasDtypeChange(input, output)) {
      return false;
    }
    if (input.isTiled()) {
      return !isOnDeviceLayoutChangeSupportedForDataType(output.getDataType());
    }
    if (isFloatFamily(input.getDataType()) &&
        isFloatFamily(output.getDataType())) {
      return false;
    }
    return !isOnDeviceLayoutChangeSupportedForDataType(input.getDataType());
  }

  //===--------------------------------------------------------------------===//
  // Orchestration.
  //===--------------------------------------------------------------------===//

  mlir::LogicalResult createLayoutConversionOps(ttnn::ToLayoutOp op,
                                                IRRewriter &rewriter) const {
    auto [inputLayout, outputLayout] = getInputOutputLayouts(op);

    bool needsLayoutChange = hasLayoutChange(inputLayout, outputLayout);
    bool needDtype = hasDtypeChange(inputLayout, outputLayout);
    bool needMemory = hasMemoryChange(inputLayout, outputLayout);

    if (!needsLayoutChange && !needDtype && !needMemory) {
      op->emitError(
          "Redundant ttnn::ToLayoutOp - no ttnn layout ops needed, this may be "
          "due to the forcing of tile/row major layouts.");
      return failure();
    }

    mlir::Value current =
        buildLayoutConversion(op, rewriter, inputLayout, outputLayout);

    op.getResult().replaceAllUsesWith(current);
    return success();
  }

  // Emits the layout/dtype/memory conversion ops and returns the final value.
  mlir::Value buildLayoutConversion(ttnn::ToLayoutOp op, IRRewriter &rewriter,
                                    TTNNLayoutAttr inputLayout,
                                    TTNNLayoutAttr outputLayout) const {
    mlir::Value current = op.getInput();

    rewriter.setInsertionPoint(op);

    // CPU-hoisted boundaries keep the layout/typecast work on host.
    bool isCpuHoistedInputOrResult = isOutputFromCPUHoistedFunction(current) ||
                                     isInputToCPUHoistedFunction(op);
    bool bothOnHost =
        inputLayout.isSystemBufferType() && outputLayout.isSystemBufferType();
    // Run the (un)tilize on host when it cannot run on device at all
    // (layoutChangeNeedsHost) or when running it on device would force the
    // typecast onto a row-major tensor (onDeviceTypecastWouldBeRowMajor). In
    // both cases the typecast can still stay on device as a TILE->TILE op.
    bool layoutNeedsHost =
        layoutChangeNeedsHost(inputLayout, outputLayout) ||
        onDeviceTypecastWouldBeRowMajor(inputLayout, outputLayout);

    // The typecast runs on host only for genuinely host-resident work: both
    // sides on host, a CPU-hoisted boundary, or a host (un)tilize whose device
    // side cannot hold the TILE tensor for an on-device typecast.
    bool fullyOnHost = bothOnHost || isCpuHoistedInputOrResult ||
                       (layoutNeedsHost &&
                        !canKeepTypecastOnDevice(inputLayout, outputLayout));

    if (fullyOnHost) {
      // Bring the tensor to host (if needed), do the layout/dtype changes
      // there, then move to the output memory (if it is on device). to_device
      // carries the full target memory config, so no separate to_memory_config
      // is required.
      if (inputLayout.isDeviceBufferType()) {
        current = createFromDeviceOp(rewriter, current);
      }
      current = layoutAndDtypeTransformations(rewriter, current, inputLayout,
                                              outputLayout,
                                              /*onDevice=*/false);
      if (outputLayout.isDeviceBufferType()) {
        current = createToDeviceOp(rewriter, current, outputLayout);
      }
      return current;
    }

    if (layoutNeedsHost) {
      // The tilize/untilize must run on host, but the typecast stays on device
      // as a TILE->TILE op (the dtype is preserved across the host round-trip).
      // Reached only when canKeepTypecastOnDevice(input, output) holds.
      bool needDtype = hasDtypeChange(inputLayout, outputLayout);
      if (inputLayout.isTiled()) {
        // Untilize: typecast on device (TILE) first, untilize on host, then
        // land in the output memory.
        if (needDtype) {
          current =
              createTypecastOp(rewriter, current, outputLayout.getDataType());
        }
        current = createFromDeviceOp(rewriter, current);
        current = createLayoutStep(rewriter, current, outputLayout,
                                   /*padWorkaround=*/false);
        if (outputLayout.isDeviceBufferType()) {
          current = createToDeviceOp(rewriter, current, outputLayout);
        }
        return current;
      }
      // Tilize: tilize on host (dtype preserved), move to the output memory,
      // then typecast on device (TILE).
      if (inputLayout.isDeviceBufferType()) {
        current = createFromDeviceOp(rewriter, current);
      }
      current = createLayoutStep(rewriter, current, outputLayout,
                                 /*padWorkaround=*/false);
      current = createToDeviceOp(rewriter, current, outputLayout);
      if (needDtype) {
        current =
            createTypecastOp(rewriter, current, outputLayout.getDataType());
      }
      return current;
    }

    if (shouldLayoutAndDtypeChangesRunBeforeMemoryMove(inputLayout,
                                                       outputLayout)) {
      // The layout/dtype changes run in the input memory; the memory move (if
      // any) comes after.
      current = layoutAndDtypeTransformations(rewriter, current, inputLayout,
                                              outputLayout, /*onDevice=*/true);
      auto currentEncoding = mlir::cast<TTNNLayoutAttr>(
          mlir::cast<RankedTensorType>(current.getType()).getEncoding());
      if (hasMemoryChange(currentEncoding, outputLayout)) {
        current = createMemoryMove(rewriter, current, outputLayout);
      }
      return current;
    }

    // Move to the output memory first, then run the layout/dtype changes.
    current = createMemoryMove(rewriter, current, outputLayout);
    return layoutAndDtypeTransformations(rewriter, current, inputLayout,
                                         outputLayout, /*onDevice=*/true);
  }
};
} // namespace mlir::tt::ttnn
