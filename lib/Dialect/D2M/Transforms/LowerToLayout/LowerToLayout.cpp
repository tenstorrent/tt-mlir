// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/LowerToLayout/Plan.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Utils/AffineMapUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <string>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERTOLAYOUT
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Helper struct to encapsulate tensor info; this allows us to package
// MetalLayoutAttr as optional gracefully.
namespace {
struct TensorInfo {
  RankedTensorType type;
  std::optional<ttcore::MetalLayoutAttr> layout;

  static TensorInfo from(Value val) {
    return fromType(mlir::cast<RankedTensorType>(val.getType()));
  }

  static TensorInfo fromType(RankedTensorType type) {
    auto layout =
        mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(type.getEncoding());
    return {type, layout ? std::optional(layout) : std::nullopt};
  }

  bool hasLayout() const { return layout.has_value(); }

  ttcore::MemorySpace getMemorySpace() const {
    return layout ? layout->getMemorySpace() : ttcore::MemorySpace::System;
  }

  bool isL1() const {
    return hasLayout() &&
           layout->getMemorySpace() == ttcore::MemorySpace::DeviceL1;
  }

  bool isDRAM() const {
    return hasLayout() &&
           layout->getMemorySpace() == ttcore::MemorySpace::DeviceDRAM;
  }

  bool isSystem() const {
    return !hasLayout() ||
           layout->getMemorySpace() == ttcore::MemorySpace::System;
  }

  ArrayRef<int64_t> getGridShape() const {
    assert(hasLayout() && "Cannot get grid shape without layout");
    return layout->getGridShape(type);
  }
};

} // namespace

namespace {

// ============================================================================
// Helper functions for building GenericOp regions with RemoteLoad/RemoteStore
// ============================================================================

// Extract the shard type from an operand allocation value, including CB block
// arguments passed to generic regions.
static Type getShardTypeFromCB(Value operandAlloc) {
  return operandAlloc.getType();
}

// Build identity grid indices for a given grid rank.
static SmallVector<Value>
buildIdentityGridIndices(OpBuilder &builder, Location loc, size_t gridRank) {
  AffineMap indexingMap = builder.getMultiDimIdentityMap(gridRank);
  return d2m::utils::buildGridIndices(builder, loc, indexingMap);
}

// Create a RemoteLoadOp in implicit form.
static Value createRemoteLoad(OpBuilder &builder, Location loc, Type shardType,
                              Value source, ArrayRef<Value> indices) {
  // RemoteLoadOp writes into an explicitly allocated local buffer.
  auto tensorType = mlir::cast<RankedTensorType>(shardType);
  auto bufferOp = builder.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                  tensorType.getElementType());
  Value buffer = bufferOp.getResult();
  return builder.create<RemoteLoadOp>(loc, shardType, buffer, source, indices)
      .getResult();
}

// Create a tensor.empty with identical result type.
static Value createTensorEmpty(OpBuilder &builder, Location loc,
                               Type shardType) {
  auto tensorType = mlir::cast<RankedTensorType>(shardType);
  return builder
      .create<tensor::EmptyOp>(loc, tensorType.getShape(),
                               tensorType.getElementType())
      .getResult();
}

// Create a RemoteStoreOp in implicit form.
static Value createRemoteStore(OpBuilder &builder, Location loc,
                               Value destination, ArrayRef<Value> indices,
                               Value localBuffer) {
  return builder
      .create<RemoteStoreOp>(loc, destination.getType(), destination, indices,
                             localBuffer)
      .getResult();
}

// Complete identity load-store pattern: load from input, acquire output buffer,
// and return both along with the indices. This is useful for operations that
// need to perform transformations between load and store (e.g., tilize, mask).
struct IdentityLoadStoreResult {
  Value src;
  Value dst;
  SmallVector<Value> indices;
};

static IdentityLoadStoreResult
buildIdentityLoadStore(OpBuilder &builder, Location loc, Value inputCBBlockArg,
                       Value outputCBBlockArg, Value input, Value output,
                       int64_t outputOperandIndex) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto inputLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
  size_t gridRank = inputLayout.getGridShape(inputType).size();

  Type inputShardType = getShardTypeFromCB(inputCBBlockArg);
  Type outputShardType = getShardTypeFromCB(outputCBBlockArg);
  SmallVector<Value> indices = buildIdentityGridIndices(builder, loc, gridRank);

  Value src = createRemoteLoad(builder, loc, inputShardType, input, indices);
  Value dst = createTensorEmpty(builder, loc, outputShardType);

  return {src, dst, indices};
}

class D2MLowerToLayoutRewriter : public OpRewritePattern<ToLayoutOp> {

public:
  D2MLowerToLayoutRewriter(MLIRContext *context,
                           ArrayRef<int64_t> targetGridShape)
      : OpRewritePattern(context, PatternBenefit(1)),
        targetGridShape(targetGridShape) {}

  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  // Lower mapping transformations (grid redistribution, padding changes,
  // collapse changes, index map transformations) to ViewLayoutOp + DMA generic.
  // The ViewLayoutOp represents the transformation as an affine map, and the
  // DMA generic materializes the data movement for L1→L1 transformations.
  static Value lowerMappingChange(PatternRewriter &rewriter, Value input,
                                  Value output, Location loc) {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    // Precondition: both operands must have layouts, be in the same memory
    // space, and have the same element type. These are guaranteed by the
    // compound splitting logic upstream.
    assert((inputInfo.hasLayout() && outputInfo.hasLayout()) &&
           "Mapping change requires both input and output to have layouts");
    assert(inputInfo.getMemorySpace() == outputInfo.getMemorySpace() &&
           "Mapping change should not change memory space");
    assert(inputInfo.type.getElementType() ==
               outputInfo.type.getElementType() &&
           "Mapping change should not change element type");

    auto inputLayout = *inputInfo.layout;
    auto outputLayout = *outputInfo.layout;

    // Simple tilized reblocking can use a direct device-space map, avoiding
    // tile-unaligned logical-space mappings.
    bool isSimpleReblocking =
        (inputLayout.getLogicalShape() == outputLayout.getLogicalShape() &&
         inputLayout.getDimAlignments() == outputLayout.getDimAlignments() &&
         inputLayout.getCollapsedIntervals() ==
             outputLayout.getCollapsedIntervals());

    bool bothTilized =
        ttcore::isTiled(inputInfo.type) && ttcore::isTiled(outputInfo.type);

    AffineMap viewMap;

    if (isSimpleReblocking && bothTilized) {
      // Fast path: pure grid reblocking on tilized tensors.
      // Use calculateReblockMap which works directly on device shapes without
      // going through logical space (avoids tile alignment issues).
      viewMap = ttmlir::utils::calculateReblockMap(inputInfo.type.getShape(),
                                                   outputInfo.type.getShape(),
                                                   rewriter.getContext());
    } else {
      // Complex layout changes are represented through shared logical space.
      viewMap = ttcore::utils::buildLayoutTransformMap(
          inputLayout, inputInfo.type, outputLayout, outputInfo.type);
    }

    auto newLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), outputLayout.getLogicalShape(),
        outputLayout.getDimAlignments(), outputLayout.getCollapsedIntervals(),
        outputLayout.getMemorySpace(), outputLayout.getMemoryLayout());

    auto viewType =
        RankedTensorType::get(outputInfo.type.getShape(),
                              outputInfo.type.getElementType(), newLayout);

    // Pass the transformation map via the remapping attribute.
    Value viewOp = rewriter.create<ViewLayoutOp>(loc, viewType, input, viewMap,
                                                 /*reinterpretLayout=*/false);

    // Materialize L1→L1 transformations with a DMA generic that performs the
    // actual data movement according to the view's affine map.
    if (!inputInfo.isDRAM() && !outputInfo.isDRAM()) {
      auto gridShape = outputInfo.getGridShape();
      const size_t gridRank = gridShape.size();

      // Build identity indexing maps for the generic operation. The view's
      // affine map handles all address transformations.
      ArrayAttr indexingMaps, iteratorTypes;
      std::tie(indexingMaps, iteratorTypes) =
          GenericOp::buildParallelAffineMapsAndIteratorTypes(
              rewriter, /*arity=*/2, gridRank);
      auto indexingMapAttr = mlir::cast<AffineMapAttr>(indexingMaps[0]);
      AffineMap indexingMap = indexingMapAttr.getValue();

      return rewriter
          .create<GenericOp>(
              loc, viewOp, output, /*additionalArgs=*/ValueRange(),
              [&](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
                // Load from input, store to output (load+store pair for proper
                // CB association)
                Type inputShardType = getShardTypeFromCB(blockArgs[0]);
                SmallVector<Value> indices = d2m::utils::buildGridIndices(
                    builder, innerLoc, indexingMap);

                // Load-store idiom
                Value loadedData = createRemoteLoad(
                    builder, innerLoc, inputShardType, viewOp, indices);
                Value storeResult = createRemoteStore(builder, innerLoc, output,
                                                      indices, loadedData);
                builder.create<YieldOp>(innerLoc, storeResult);
              },
              ThreadType::Unified)
          .getResult(0);
    }
    // DRAM operations use the view directly without immediate
    // materialization.
    return viewOp;
  }

  static Value lowerSystemLayoutChange(PatternRewriter &rewriter, Value input,
                                       Value output, Location loc) {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    assert(inputInfo.isSystem() != outputInfo.isSystem() &&
           "one of input or output must be system for now");

    // Use the layout of whichever side has a layout (input or output).
    auto deviceLayout =
        inputInfo.isSystem() ? outputInfo.layout : inputInfo.layout;
    assert(deviceLayout.has_value() && "Device side must have a layout");

    // Emit dedicated host transfer ops based on direction.
    if (inputInfo.isSystem()) {
      // Host → Device: use ToDeviceOp.
      return rewriter.create<ToDeviceOp>(loc, input, output, *deviceLayout)
          .getResult(0);
    }
    // Device → Host: use ToHostOp.
    return rewriter.create<ToHostOp>(loc, input, output, *deviceLayout)
        .getResult(0);
  }

  // Return true if the input operand to a ToLayoutOp is itself a result of a
  // device->device memspace ToLayoutOp.
  static bool producerMustBeLoweredFirst(ToLayoutOp op) {
    if (auto producer = op.getInput().getDefiningOp<ToLayoutOp>()) {
      auto producerInputInfo = TensorInfo::from(producer.getInput());
      auto producerOutputInfo = TensorInfo::from(producer.getOutput());

      // Check if both producer's input and output are on device
      // (i.e., both have layouts and neither is system memory).
      if (producerInputInfo.hasLayout() && producerOutputInfo.hasLayout() &&
          !producerInputInfo.isSystem() && !producerOutputInfo.isSystem()) {
        return true;
      }
    }
    return false;
  }

  Value lowerDatamovementGeneric(PatternRewriter &rewriter, Value input,
                                 Value output, Location loc) const {
    auto inputInfo = TensorInfo::from(input);
    auto outputInfo = TensorInfo::from(output);

    if (inputInfo.isSystem() || outputInfo.isSystem()) {
      return lowerSystemLayoutChange(rewriter, input, output, loc);
    }

    // Both input and output must have layouts at this point.
    assert(inputInfo.hasLayout() && outputInfo.hasLayout());

    Value viewInput = input;

    bool isSrcDramOrReblock =
        inputInfo.isDRAM() ||
        (!outputInfo.isDRAM() &&
         (inputInfo.getGridShape() != outputInfo.getGridShape()));

    assert(!(isSrcDramOrReblock && outputInfo.isDRAM()) &&
           "input and output cannot both be remote");

    auto buildConcreteView = [&](Value fromVal, RankedTensorType fromTy,
                                 RankedTensorType toTy) -> Value {
      auto *ctx = rewriter.getContext();
      auto baseLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(fromTy.getEncoding());
      auto targetLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(toTy.getEncoding());

      AffineMap map;
      if (ttmlir::utils::volume<int64_t>(fromTy.getShape()) ==
          ttmlir::utils::volume<int64_t>(toTy.getShape())) {
        map = ttmlir::utils::calculateReblockMap(fromTy.getShape(),
                                                 toTy.getShape(), ctx);
      } else {
        map = ttcore::utils::buildLayoutTransformMap(baseLayout, fromTy,
                                                     targetLayout, toTy);
      }

      auto enc = ttcore::MetalLayoutAttr::get(
          ctx, baseLayout.getLogicalShape(), baseLayout.getDimAlignments(),
          baseLayout.getCollapsedIntervals(), baseLayout.getMemorySpace(),
          baseLayout.getMemoryLayout());
      auto resultTy =
          RankedTensorType::get(toTy.getShape(), toTy.getElementType(), enc);
      return rewriter
          .create<ViewLayoutOp>(loc, resultTy, fromVal, map,
                                /*reinterpretLayout=*/false)
          .getResult();
    };

    if (isSrcDramOrReblock) {
      viewInput = buildConcreteView(input, inputInfo.type, outputInfo.type);
    }

    Value viewOutput = output;
    ttcore::GridAttr grid;
    if (outputInfo.isDRAM()) {
      viewOutput = buildConcreteView(output, outputInfo.type, inputInfo.type);
      auto gridShape = llvm::to_vector(inputInfo.getGridShape());
      if (auto maps =
              utils::getGridMapsFromVirtualGridMapping(input, gridShape)) {
        grid = rewriter.getAttr<ttcore::GridAttr>(gridShape, maps->first,
                                                  maps->second);
      }
    }

    auto viewOutputType = mlir::cast<RankedTensorType>(viewOutput.getType());
    auto viewOutputLayout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(
        viewOutputType.getEncoding());
    const size_t gridRank =
        viewOutputLayout ? viewOutputLayout.getShardShape(viewOutputType).size()
                         : viewOutputType.getShape().size();

    ArrayAttr indexingMaps, iteratorTypes;
    std::tie(indexingMaps, iteratorTypes) =
        GenericOp::buildParallelAffineMapsAndIteratorTypes(
            rewriter, /*arity=*/2, gridRank);
    auto indexingMapAttr = mlir::cast<AffineMapAttr>(indexingMaps[0]);
    AffineMap indexingMap = indexingMapAttr.getValue();

    auto result =
        rewriter
            .create<GenericOp>(
                loc, viewInput, viewOutput, /*additionalArgs=*/ValueRange(),
                [&](OpBuilder &builder, Location innerLoc,
                    ValueRange blockArgs) {
                  Type inputShardType = getShardTypeFromCB(blockArgs[0]);
                  SmallVector<Value> indices = d2m::utils::buildGridIndices(
                      builder, innerLoc, indexingMap);

                  // Use load+store idiom for proper CB association
                  Value loadedData = createRemoteLoad(
                      builder, innerLoc, inputShardType, viewInput, indices);
                  Value storeResult = createRemoteStore(
                      builder, innerLoc, viewOutput, indices, loadedData);
                  builder.create<YieldOp>(innerLoc, storeResult);
                },
                ThreadType::Unified, grid)
            .getResult(0);
    if (outputInfo.isDRAM() && result.getType() != output.getType()) {
      return buildConcreteView(result,
                               mlir::cast<RankedTensorType>(result.getType()),
                               outputInfo.type);
    }
    return result;
  }

  Value lowerFormatConversionGeneric(PatternRewriter &rewriter, Value input,
                                     Value output, Location loc) const {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto outputType = mlir::cast<RankedTensorType>(output.getType());
    bool inputTiled = ttcore::isTiled(inputType);
    bool outputTiled = ttcore::isTiled(outputType);
    assert(inputTiled != outputTiled &&
           "one of input or output must be tiled for now");
    assert(TensorInfo::from(input).getGridShape() ==
               TensorInfo::from(output).getGridShape() &&
           "format conversion generic requires matching input/output grids");

    return rewriter
        .create<GenericOp>(
            loc, input, output, /*additionalArgs=*/ValueRange(),
            [=](OpBuilder &builder, Location innerLoc, ValueRange blockArgs) {
              auto [src, dst, indices] =
                  buildIdentityLoadStore(builder, innerLoc, blockArgs[0],
                                         blockArgs[1], input, output, 1);

              Value result;
              if (inputTiled) {
                result = builder
                             .create<TileUntilizeBlockOp>(
                                 innerLoc, dst.getType(), src, dst)
                             .getResult();
              } else {
                result = builder
                             .create<TileTilizeBlockOp>(innerLoc, dst.getType(),
                                                        src, dst)
                             .getResult();
              }

              Value storeResult =
                  createRemoteStore(builder, innerLoc, output, indices, result);
              builder.create<YieldOp>(innerLoc, storeResult);
            },
            ThreadType::Unified)
        .getResult(0);
  }

  static bool matchesOutputSpec(Value value, const OutputBufferSpec &spec) {
    if (!value || value.getType() != spec.type) {
      return false;
    }

    AffineMap currentForward =
        utils::getVirtualGridForwardMapping(value).value_or(AffineMap());
    AffineMap currentInverse =
        utils::getVirtualGridInverseMapping(value).value_or(AffineMap());
    // View remappings are semantic view metadata. Do not reuse such values as
    // generic outs buffers, even when their type and VGM match the spec.
    if (utils::getAssociatedRemapping(value)) {
      return false;
    }
    return currentForward == spec.vgmForward &&
           currentInverse == spec.vgmInverse;
  }

  Value createEmpty(PatternRewriter &rewriter, Location loc,
                    const OutputBufferSpec &spec, Value reusableOutput = {},
                    bool allowReuse = true) const {
    if (allowReuse && matchesOutputSpec(reusableOutput, spec)) {
      return reusableOutput;
    }

    AffineMapAttr invAttr =
        spec.vgmInverse ? AffineMapAttr::get(spec.vgmInverse) : nullptr;
    AffineMapAttr fwdAttr =
        spec.vgmForward ? AffineMapAttr::get(spec.vgmForward) : nullptr;
    if (!invAttr && !fwdAttr) {
      if (auto layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
              spec.type.getEncoding())) {
        return rewriter
            .create<d2m::EmptyOp>(loc, spec.type.getShape(),
                                  spec.type.getElementType(), layout,
                                  targetGridShape)
            .getResult();
      }
    }

    return rewriter.create<d2m::EmptyOp>(loc, spec.type, invAttr, fwdAttr)
        .getResult();
  }

  // `d2m.mask` is created before LowerToLayout, so its input and output types
  // initially match the pre-lowered ToLayout result. Lowering may replace that
  // input with a value that has a different concrete physical shape, for
  // example after tilization chooses a tile-aligned buffer. Since MaskOp
  // requires input/output/result to have identical types, rebuild the mask
  // output to match the rewritten input while preserving the old output's
  // virtual-grid metadata.
  void repairMaskAfterInputRewrite(PatternRewriter &rewriter,
                                   MaskOp mask) const {
    auto inputType =
        mlir::dyn_cast<RankedTensorType>(mask.getInput().getType());
    if (!inputType || mask.getOutput().getType() == inputType) {
      return;
    }

    Value oldOutput = mask.getOutput();
    d2m::EmptyOp oldOutputEmpty = oldOutput.getDefiningOp<d2m::EmptyOp>();

    OutputBufferSpec outputSpec{inputType};
    if (oldOutputEmpty) {
      if (auto attr = oldOutputEmpty.getVirtualGridForwardMappingAttr()) {
        outputSpec.vgmForward = attr.getValue();
      }
      if (auto attr = oldOutputEmpty.getVirtualGridInverseMappingAttr()) {
        outputSpec.vgmInverse = attr.getValue();
      }
    }

    rewriter.setInsertionPoint(mask);
    Value newOutput = createEmpty(rewriter, mask.getLoc(), outputSpec);
    auto newMask =
        rewriter.create<MaskOp>(mask.getLoc(), mask.getInput(), newOutput,
                                mask.getLogicalShape(), mask.getFillValue());
    rewriter.replaceOp(mask, newMask.getResult());

    if (oldOutputEmpty && oldOutput.use_empty()) {
      rewriter.eraseOp(oldOutputEmpty);
    }
  }

  // Pull the planning-relevant metadata off a Value: its type plus any
  // remapping / virtual-grid-mapping attached via producing view/empty ops.
  static PlanState extractPlanState(Value v) {
    PlanState state;
    state.type = mlir::cast<RankedTensorType>(v.getType());
    state.remapping = utils::getAssociatedRemapping(v).value_or(AffineMap());
    state.vgmForward =
        utils::getVirtualGridForwardMapping(v).value_or(AffineMap());
    state.vgmInverse =
        utils::getVirtualGridInverseMapping(v).value_or(AffineMap());
    return state;
  }

  // Materialize `plan` as IR by applying each Step to the value produced by the
  // previous Step.
  Value emit(PatternRewriter &rewriter, ToLayoutOp op, const Plan &plan) const {
    Value currentValue = op.getInput();
    Location loc = op.getLoc();

    auto createStepOutput = [&](const OutputBufferSpec &spec,
                                bool allowReuse = true) -> Value {
      return this->createEmpty(rewriter, loc, spec, op.getOutput(), allowReuse);
    };

    for (const Step &step : plan) {
      if (const auto *s = std::get_if<HostToDeviceStep>(&step)) {
        currentValue = lowerSystemLayoutChange(
            rewriter, currentValue, createStepOutput(s->output), loc);
      } else if (const auto *s = std::get_if<DeviceToHostStep>(&step)) {
        currentValue = lowerSystemLayoutChange(
            rewriter, currentValue,
            createStepOutput(OutputBufferSpec{s->outputType}), loc);
      } else if (const auto *s = std::get_if<L1ToDRAMStep>(&step)) {
        currentValue = lowerDatamovementGeneric(
            rewriter, currentValue, createStepOutput(s->output), loc);
      } else if (const auto *s = std::get_if<DRAMToL1Step>(&step)) {
        currentValue = lowerDatamovementGeneric(
            rewriter, currentValue, createStepOutput(s->output), loc);
      } else if (const auto *s = std::get_if<TilizeStep>(&step)) {
        currentValue = lowerFormatConversionGeneric(
            rewriter, currentValue, createStepOutput(s->output), loc);
      } else if (const auto *s = std::get_if<UntilizeStep>(&step)) {
        currentValue = lowerFormatConversionGeneric(
            rewriter, currentValue, createStepOutput(s->output), loc);
      } else if (const auto *s = std::get_if<RebufferStep>(&step)) {
        currentValue = lowerDatamovementGeneric(
            rewriter, currentValue, createStepOutput(s->output), loc);
      } else if (const auto *s = std::get_if<ReshardStep>(&step)) {
        currentValue = lowerMappingChange(rewriter, currentValue,
                                          createStepOutput(s->output), loc);
      } else if (const auto *s = std::get_if<RemapStep>(&step)) {
        currentValue = rewriter
                           .create<ViewLayoutOp>(loc, s->outputType,
                                                 currentValue, s->remapping,
                                                 /*reinterpretLayout=*/false)
                           .getResult();
      }
    }
    return currentValue;
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    if (producerMustBeLoweredFirst(op)) {
      return failure();
    }
    PlanState src = extractPlanState(op.getInput());
    PlanState tgt = extractPlanState(op.getOutput());
    Plan plan = minimize(
        canonicalize(src, tgt, targetGridShape, rewriter.getContext()));
    if (plan.empty()) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    // Capture direct mask users before replacing the ToLayout result. The
    // replacement updates the mask input in-place, and that can leave the mask
    // output/result type stale until repaired below.
    SmallVector<MaskOp> maskUsers;
    for (OpOperand &use : op.getResult(0).getUses()) {
      if (use.getOperandNumber() == 0) {
        if (auto mask = dyn_cast<MaskOp>(use.getOwner())) {
          maskUsers.push_back(mask);
        }
      }
    }

    Value result = emit(rewriter, op, plan);
    rewriter.replaceOp(op, result);
    for (MaskOp mask : maskUsers) {
      repairMaskAfterInputRewrite(rewriter, mask);
    }
    return success();
  }

  ArrayRef<int64_t> getTargetGridShape() const { return targetGridShape; }

private:
  llvm::SmallVector<int64_t> targetGridShape;
};
} // namespace

namespace {
class D2MLowerToLayout : public impl::D2MLowerToLayoutBase<D2MLowerToLayout> {
public:
  using impl::D2MLowerToLayoutBase<D2MLowerToLayout>::D2MLowerToLayoutBase;

  llvm::SmallVector<int64_t> getTargetGridShape() {
    ::mlir::ModuleOp moduleOp = getOperation();
    mlir::tt::ttcore::DeviceAttr device =
        mlir::tt::ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    // Use the physical worker grid as the target for virtual-grid legality and
    // materialization decisions.
    llvm::SmallVector<int64_t> targetGridShape = getTargetGridShape();

    patterns.add<D2MLowerToLayoutRewriter>(&getContext(), targetGridShape);
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
