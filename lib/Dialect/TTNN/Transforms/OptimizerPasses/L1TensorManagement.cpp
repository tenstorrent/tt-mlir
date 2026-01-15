// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNL1TENSORMANAGEMENT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Information about a live tensor in L1.
struct L1TensorInfo {
  Value value;
  Operation *definingOp;
  TTNNLayoutAttr layout;
  uint64_t sizeBytes;
  Operation *lastUseOp;
  int64_t lastUsePosition;
  bool isForkOutput;
  int numRemainingUses;
};

// Context for processing a function - holds all shared state.
struct ProcessingContext {
  IRRewriter &rewriter;
  const LivenessBlockInfo *blockInfo;
  const llvm::SmallVector<Operation *> &schedule;
  const llvm::DenseMap<Operation *, int64_t> &schedulePos;
  llvm::DenseMap<Value, L1TensorInfo> *liveTensors;
  uint64_t *currentL1Usage;
  uint64_t l1Budget;

  ProcessingContext(IRRewriter &rewriter, const LivenessBlockInfo *blockInfo,
                    const llvm::SmallVector<Operation *> &schedule,
                    const llvm::DenseMap<Operation *, int64_t> &schedulePos,
                    llvm::DenseMap<Value, L1TensorInfo> &liveTensors,
                    uint64_t &currentL1Usage, uint64_t l1Budget)
      : rewriter(rewriter), blockInfo(blockInfo), schedule(schedule),
        schedulePos(schedulePos), liveTensors(&liveTensors),
        currentL1Usage(&currentL1Usage), l1Budget(l1Budget) {}
};

} // namespace

class TTNNL1TensorManagement
    : public impl::TTNNL1TensorManagementBase<TTNNL1TensorManagement> {
public:
  using impl::TTNNL1TensorManagementBase<
      TTNNL1TensorManagement>::TTNNL1TensorManagementBase;

  void runOnOperation() final {
#ifndef TTMLIR_ENABLE_OPMODEL
    llvm::llvm_unreachable_internal(
        "TTNNL1TensorManagement pass requires OpModel support to be enabled.");
#else
    op_model::ScopedSingletonDeviceGuard deviceGuard;

    ModuleOp moduleOp = getOperation();

    // Get L1 budget from system description.
    ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
        moduleOp->getAttr(ttcore::SystemDescAttr::name));
    ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
    uint64_t l1Budget =
        static_cast<uint64_t>(chipDesc.getUsableL1Size() * tensorL1UsageCap);

    // Get device grid for L1 interleaved layout creation.
    deviceGrid = ttcore::lookupDevice(moduleOp).getWorkerGrid();

    // Process each function.
    moduleOp->walk([&](func::FuncOp funcOp) {
      if (ttmlir::utils::isConstEvalFunc(funcOp)) {
        return;
      }
      processFunction(funcOp, l1Budget);
    });
#endif
  }

private:
  ttcore::GridAttr deviceGrid;

  //===--------------------------------------------------------------------===//
  // Utility functions
  //===--------------------------------------------------------------------===//

  bool isForkPoint(Operation *op) {
    if (op->getNumResults() == 0) {
      return false;
    }
    Value result = op->getResult(0);
    int userCount = 0;
    for ([[maybe_unused]] auto &use : result.getUses()) {
      userCount++;
    }
    return userCount > 1;
  }

  int countUsers(Value value) {
    int count = 0;
    for ([[maybe_unused]] auto &use : value.getUses()) {
      count++;
    }
    return count;
  }

  bool hasDeallocateActivation(Operation *op, Value input) {
    auto checkConvOp = [&](auto convOp) -> bool {
      if (convOp.getInput() != input) {
        return false;
      }
      auto config = convOp.getConv2dConfigAttr();
      if (!config) {
        return false;
      }
      auto deallocAttr = config.getDeallocateActivation();
      return deallocAttr && deallocAttr.getValue();
    };

    if (auto conv2d = dyn_cast<Conv2dOp>(op)) {
      return checkConvOp(conv2d);
    }
    if (auto convTranspose2d = dyn_cast<ConvTranspose2dOp>(op)) {
      return checkConvOp(convTranspose2d);
    }
    return false;
  }

  // Check if an op's result is directly returned from the function.
  // We should not promote such results to L1 as they need to match the
  // function's return type signature (typically DRAM).
  bool isResultReturnedFromFunction(Operation *op) {
    if (op->getNumResults() == 0) {
      return false;
    }
    Value result = op->getResult(0);
    for (Operation *user : result.getUsers()) {
      if (isa<func::ReturnOp>(user)) {
        return true;
      }
    }
    return false;
  }

  // Create an L1 interleaved layout for an op's output using the device's
  // worker grid.
  TTNNLayoutAttr createL1InterleavedLayout(Operation *op,
                                           RankedTensorType outputType,
                                           TTNNLayoutAttr baseLayout) {
    return baseLayout.withBufferType(BufferType::L1)
        .withMemoryLayout(TensorMemoryLayout::Interleaved)
        .withGrid(outputType, deviceGrid, {{0, -1}});
  }

  // Try to validate an op with L1 interleaved output and return the L1 size.
  // Returns 0 if validation fails or the op doesn't support L1 interleaved.
  // Uses actual input layouts (doesn't change them) and accounts for L1 pressure.
  uint64_t tryGetL1InterleavedSize(Operation *op, uint64_t currentL1Usage) {
    if (op->getNumResults() == 0) {
      return 0;
    }

    auto outputType =
        mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outputType) {
      return 0;
    }

    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());
    if (!currentLayout) {
      return 0;
    }

    // Create L1 interleaved layout for validation.
    TTNNLayoutAttr l1Layout =
        createL1InterleavedLayout(op, outputType, currentLayout);

    // Use actual input layouts (don't convert them).
    std::vector<TTNNLayoutAttr> inputLayouts = utils::extractInputLayouts(op);

    OpConfig l1Config;
    l1Config.outputLayout = l1Layout;
    populateOpSpecificAttrs(op, l1Config);

    op_constraint_validation::ValidationResult result =
        op_constraint_validation::validateOperation(op, inputLayouts, l1Config,
                                                    currentL1Usage);

    return result.isSuccess() ? result.outputL1Usage : 0;
  }

  // Check if layout is L1 interleaved.
  bool isL1Interleaved(TTNNLayoutAttr layout) {
    if (!layout) {
      return false;
    }
    return layout.getBufferType() == BufferType::L1 &&
           layout.getMemLayout().getValue() == TensorMemoryLayout::Interleaved;
  }

  // Check if layout is DRAM interleaved.
  bool isDRAMInterleaved(TTNNLayoutAttr layout) {
    if (!layout) {
      return false;
    }
    return layout.getBufferType() == BufferType::DRAM &&
           layout.getMemLayout().getValue() == TensorMemoryLayout::Interleaved;
  }

  // Get L1 size in bytes for a tensor (sharded or interleaved).
  // Returns 0 if tensor is not in L1.
  uint64_t getL1SizeBytes(Operation *op) {
    if (op->getNumResults() == 0) {
      return 0;
    }
    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorType) {
      return 0;
    }
    auto layout = dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());
    if (!layout) {
      return 0;
    }

    // For L1 sharded, use getOpOutputL1Usage.
    if (layout.hasShardedL1TensorMemoryLayout()) {
      return utils::getOpOutputL1Usage(layout);
    }

    // For L1 interleaved, validate to get the actual L1 usage.
    if (isL1Interleaved(layout)) {
      return tryGetL1InterleavedSize(op, /*currentL1Usage=*/0);
    }

    return 0;
  }

  TTNNLayoutAttr createDRAMInterleavedLayout(TTNNLayoutAttr baseLayout) {
    // withBufferType(DRAM) already sets memLayout to Interleaved and grid to 1x1.
    return baseLayout.withBufferType(BufferType::DRAM);
  }

  MemoryConfigAttr createDRAMInterleavedMemoryConfig(MLIRContext *ctx) {
    return MemoryConfigAttr::get(
        ctx,
        TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::Interleaved),
        BufferTypeAttr::get(ctx, BufferType::DRAM),
        /*shardSpec=*/nullptr);
  }

  // Try to promote an op's output from DRAM interleaved to L1 interleaved.
  // Returns the L1 size if successful, 0 otherwise.
  // Takes current L1 usage to validate with proper L1 pressure.
  uint64_t tryPromoteToL1Interleaved(Operation *op, uint64_t currentL1Usage) {
    if (op->getNumResults() == 0) {
      return 0;
    }

    // Skip layout ops - these should keep their specified layouts.
    if (isa<ToLayoutOp, ToMemoryConfigOp>(op)) {
      return 0;
    }

    // Skip ops whose results are returned from the function.
    // These must match the function's return type signature (typically DRAM).
    if (isResultReturnedFromFunction(op)) {
      return 0;
    }

    auto outputType =
        mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!outputType) {
      return 0;
    }

    auto currentLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());
    if (!currentLayout || !isDRAMInterleaved(currentLayout)) {
      return 0;
    }

    // Try to validate with L1 interleaved output, accounting for L1 pressure.
    uint64_t l1Size = tryGetL1InterleavedSize(op, currentL1Usage);
    if (l1Size == 0) {
      return 0;
    }

    // Update the op's output type to L1 interleaved.
    TTNNLayoutAttr l1Layout =
        createL1InterleavedLayout(op, outputType, currentLayout);
    auto newType = RankedTensorType::get(outputType.getShape(),
                                         outputType.getElementType(), l1Layout);
    op->getResult(0).setType(newType);

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] Op {}: promoted DRAM -> L1 interleaved "
                 "(size: {} bytes, L1 pressure: {} bytes)",
                 op->getName(), l1Size, currentL1Usage);

    return l1Size;
  }

  // Populate op-specific attrs for Conv2d, Matmul, and similar ops.
  void populateOpSpecificAttrs(Operation *op, OpConfig &config) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<Conv2dOp, ConvTranspose2dOp>([&config](auto convOp) {
          Conv2dAttrs attrs;
          attrs.conv2dConfig = convOp.getConv2dConfigAttr();
          attrs.deviceComputeKernelConfig = convOp.getComputeConfigAttr();
          config.opSpecificAttrs = attrs;
        })
        .Case<MatmulOp, LinearOp>([&config](auto matmulOp) {
          MatmulAttrs attrs;
          attrs.matmulProgramConfig = matmulOp.getMatmulProgramConfig();
          config.opSpecificAttrs = attrs;
        });
  }

  //===--------------------------------------------------------------------===//
  // Spill candidate selection
  //===--------------------------------------------------------------------===//

  Value findSpillCandidate(
      const llvm::DenseMap<Value, L1TensorInfo> *liveTensors,
      int64_t currentPos) {
    Value candidate;
    int64_t maxRemainingLife = -1;

    // First pass: prefer non-fork tensors.
    for (const auto &[value, info] : *liveTensors) {
      int64_t remaining = info.lastUsePosition - currentPos;
      if (!info.isForkOutput && remaining > maxRemainingLife) {
        maxRemainingLife = remaining;
        candidate = value;
      }
    }

    // If all are fork tensors, spill one anyway.
    if (!candidate) {
      maxRemainingLife = -1;
      for (const auto &[value, info] : *liveTensors) {
        int64_t remaining = info.lastUsePosition - currentPos;
        if (remaining > maxRemainingLife) {
          maxRemainingLife = remaining;
          candidate = value;
        }
      }
    }

    return candidate;
  }

  Operation *findFirstUseAfter(
      Value tensor, int64_t currentPos,
      const llvm::DenseMap<Operation *, int64_t> &schedulePos) {
    Operation *firstUser = nullptr;
    int64_t earliestPos = INT64_MAX;

    for (Operation *user : tensor.getUsers()) {
      auto it = schedulePos.find(user);
      if (it != schedulePos.end() && it->second > currentPos) {
        if (it->second < earliestPos) {
          earliestPos = it->second;
          firstUser = user;
        }
      }
    }
    return firstUser;
  }

  //===--------------------------------------------------------------------===//
  // Liveness helpers
  //===--------------------------------------------------------------------===//

  Operation *findLastUseOp(Value value, const LivenessBlockInfo *blockInfo,
                           Operation *definingOp,
                           const llvm::DenseMap<Operation *, int64_t> &schedulePos) {
    Operation *lastUseOp = blockInfo->getEndOperation(value, definingOp);

    // Check if any user is a conv2d with deallocateActivation=true.
    for (Operation *user : value.getUsers()) {
      if (hasDeallocateActivation(user, value)) {
        if (schedulePos.count(user) && schedulePos.count(lastUseOp)) {
          if (schedulePos.at(user) < schedulePos.at(lastUseOp)) {
            lastUseOp = user;
          }
        }
      }
    }

    return lastUseOp;
  }

  //===--------------------------------------------------------------------===//
  // Conv2d config modification
  //===--------------------------------------------------------------------===//

  void disableDeallocateActivation(Operation *op) {
    auto setNewConfig = [](auto convOp) {
      auto config = convOp.getConv2dConfigAttr();
      if (!config) {
        return;
      }
      auto newConfig = config.withDeallocateActivation(false);
      convOp.setConv2dConfigAttr(newConfig);
    };

    if (auto conv2d = dyn_cast<Conv2dOp>(op)) {
      setNewConfig(conv2d);
    } else if (auto convTranspose2d = dyn_cast<ConvTranspose2dOp>(op)) {
      setNewConfig(convTranspose2d);
    }
  }

  // Post-process all Conv2d ops to disable deallocate_activation when input
  // still has multiple users after tensor management. This runs AFTER spilling
  // so we account for the actual IR state - if fork edges were spilled, Conv2d
  // may now be the only user and can safely deallocate.
  void fixConvDeallocateActivationForForks(func::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      Value input;
      Conv2dConfigAttr config;

      if (auto conv2d = dyn_cast<Conv2dOp>(op)) {
        input = conv2d.getInput();
        config = conv2d.getConv2dConfigAttr();
      } else if (auto convTranspose2d = dyn_cast<ConvTranspose2dOp>(op)) {
        input = convTranspose2d.getInput();
        config = convTranspose2d.getConv2dConfigAttr();
      } else {
        return;
      }

      // Check if deallocate_activation is set to true.
      if (!config || !config.getDeallocateActivation() ||
          !config.getDeallocateActivation().getValue()) {
        return;
      }

      // Check if input still has multiple users after tensor management.
      // If fork edges were spilled, Conv2d may now be the only user.
      if (countUsers(input) > 1) {
        disableDeallocateActivation(op);
        TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                     "[L1TensorManagement] Op {}: disabled deallocate_activation"
                     " (input still has multiple users after tensor management)",
                     op->getName());
      }
    });
  }

  //===--------------------------------------------------------------------===//
  // Spilling
  //===--------------------------------------------------------------------===//

  void spillTensor(Value tensor, ProcessingContext &ctx, int64_t currentPos) {
    auto &info = (*ctx.liveTensors)[tensor];

    // Find where to insert the spill op - before the first remaining use.
    Operation *insertPoint =
        findFirstUseAfter(tensor, currentPos, ctx.schedulePos);
    if (!insertPoint) {
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] Tensor from op {} has no remaining "
                   "uses after position {}, just freeing",
                   info.definingOp->getName(), currentPos);
      *ctx.currentL1Usage -= info.sizeBytes;
      ctx.liveTensors->erase(tensor);
      return;
    }

    // Get the input tensor type and layout.
    auto inputTensorType = cast<RankedTensorType>(tensor.getType());
    TTNNLayoutAttr inputLayout =
        dyn_cast_or_null<TTNNLayoutAttr>(inputTensorType.getEncoding());
    if (!inputLayout) {
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] Cannot spill tensor from op {} - "
                   "no valid layout, just freeing",
                   info.definingOp->getName());
      *ctx.currentL1Usage -= info.sizeBytes;
      ctx.liveTensors->erase(tensor);
      return;
    }

    // Create DRAM interleaved layout.
    // withBufferType(DRAM) sets memLayout to Interleaved and grid to 1x1.
    TTNNLayoutAttr dramLayout = inputLayout.withBufferType(BufferType::DRAM);

    // Create the output type using the factory for consistency.
    RankedTensorType newType =
        utils::RankedTensorTypeFactory::create(inputTensorType, dramLayout);

    // Create the memory config from the layout for consistency.
    MemoryConfigAttr memConfig =
        MemoryConfigAttr::get(dramLayout, deviceGrid);

    ctx.rewriter.setInsertionPoint(insertPoint);
    auto spillOp = ctx.rewriter.create<ToMemoryConfigOp>(
        info.definingOp->getLoc(), newType, tensor, memConfig);

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] Spilling tensor from op {} to DRAM "
                 "(size: {} bytes, remaining life: {}) before op {}",
                 info.definingOp->getName(), info.sizeBytes,
                 info.lastUsePosition - currentPos, insertPoint->getName());

    // Replace uses of the original tensor after the spill point.
    tensor.replaceUsesWithIf(spillOp.getResult(), [&](OpOperand &use) {
      Operation *user = use.getOwner();
      auto it = ctx.schedulePos.find(user);
      return it != ctx.schedulePos.end() && it->second >= currentPos;
    });

    *ctx.currentL1Usage -= info.sizeBytes;
    ctx.liveTensors->erase(tensor);
  }

  //===--------------------------------------------------------------------===//
  // Range validation with L1 reservation
  //===--------------------------------------------------------------------===//

  // Check if an operation uses a specific value as one of its operands.
  bool opUsesValue(Operation *op, Value tensor) {
    for (Value operand : op->getOperands()) {
      if (operand == tensor) {
        return true;
      }
    }
    return false;
  }

  // Get total L1 usage from active reservations at a given position.
  uint64_t getActiveL1Usage(int64_t pos, ProcessingContext &ctx) {
    uint64_t total = 0;
    for (const auto &[value, info] : *ctx.liveTensors) {
      // Tensor is active if pos is within [definingPos, lastUsePosition]
      auto defPosIt = ctx.schedulePos.find(info.definingOp);
      if (defPosIt == ctx.schedulePos.end()) {
        continue;
      }
      int64_t defPos = defPosIt->second;
      if (pos >= defPos && pos <= info.lastUsePosition) {
        total += info.sizeBytes;
      }
    }
    return total;
  }

  // Validate that all operations in [startPos, endPos] can execute with
  // an additional tensor of size `additionalL1` occupying L1.
  // This is used to proactively check if keeping a tensor in L1 is feasible.
  //
  // The `tensor` parameter identifies which tensor we're checking. When
  // validating ops that consume this tensor directly, we don't add its size
  // as additional L1 pressure since it's already part of the op's inputLayouts.
  bool validateScheduleRangeWithReservation(ProcessingContext &ctx,
                                            int64_t startPos, int64_t endPos,
                                            uint64_t additionalL1,
                                            Value tensor) {
    if (startPos > endPos) {
      return true;
    }

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] validateScheduleRange: [{}, {}] with "
                 "additionalL1={} bytes",
                 startPos, endPos, additionalL1);

    for (int64_t pos = startPos; pos <= endPos; ++pos) {
      if (pos < 0 || pos >= static_cast<int64_t>(ctx.schedule.size())) {
        continue;
      }

      Operation *op = ctx.schedule[static_cast<size_t>(pos)];

      // Skip non-compute ops.
      if (isa<ToLayoutOp, ToMemoryConfigOp, EmptyOp, GetDeviceOp>(op)) {
        continue;
      }

      // Check if op has results.
      if (op->getNumResults() == 0) {
        continue;
      }

      auto outputType =
          dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!outputType) {
        continue;
      }

      auto outputLayout =
          dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());

      // Validate ALL ops regardless of output location or sharding.
      // - Ops may have internal L1 requirements (circular buffers, etc.)
      //   even when their output goes to DRAM.
      // - L1 sharded ops were validated by L1LayoutPropagation, but without
      //   considering additional L1 pressure from other live tensors.

      // Calculate total L1 pressure at this position.
      uint64_t existingL1 = getActiveL1Usage(pos, ctx);

      // If this op consumes the tensor we're checking, don't add its size
      // as additional L1 - it's already part of the op's input layout.
      bool opConsumesTensor = opUsesValue(op, tensor);
      uint64_t tensorL1ForThisOp = opConsumesTensor ? 0 : additionalL1;
      uint64_t totalAdditionalL1 = existingL1 + tensorL1ForThisOp;

      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] Range pos {}: Validating op {} with "
                   "L1 pressure: existing={} + additional={} = {} bytes{}",
                   pos, op->getName(), existingL1, tensorL1ForThisOp,
                   totalAdditionalL1,
                   opConsumesTensor ? " (op consumes tensor)" : "");

      // Validate op with total L1 pressure.
      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(op);

      OpConfig config;
      config.outputLayout = outputLayout;
      populateOpSpecificAttrs(op, config);

      auto result = op_constraint_validation::validateOperation(
          op, inputLayouts, config, totalAdditionalL1);

      if (!result.isSuccess()) {
        TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                     "[L1TensorManagement] Range validation FAILED at op {} "
                     "(pos {}): L1 pressure {} + {} = {}, error: {}",
                     op->getName(), pos, existingL1, tensorL1ForThisOp,
                     totalAdditionalL1, result.errorMessage);
        return false;
      }
    }

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] Range validation PASSED for [{}, {}]",
                 startPos, endPos);
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Per-operation processing steps
  //===--------------------------------------------------------------------===//

  // Step 1: Free tensors that are no longer live at this position.
  void freeExpiredTensors(ProcessingContext &ctx, int64_t pos) {
    llvm::SmallVector<Value> toFree;
    for (auto &[value, info] : *ctx.liveTensors) {
      if (info.lastUsePosition <= pos) {
        toFree.push_back(value);
      }
    }
    for (Value v : toFree) {
      *ctx.currentL1Usage -= (*ctx.liveTensors)[v].sizeBytes;
      ctx.liveTensors->erase(v);
    }
  }

  // Step 2: Handle conv2d ops with deallocateActivation.
  void handleConvDeallocateActivation(Operation *op, ProcessingContext &ctx) {
    for (Value operand : op->getOperands()) {
      if (!hasDeallocateActivation(op, operand) ||
          !ctx.liveTensors->count(operand)) {
        continue;
      }

      auto &inputInfo = (*ctx.liveTensors)[operand];

      if (inputInfo.isForkOutput && inputInfo.numRemainingUses > 1) {
        // Fork output with remaining uses - disable deallocateActivation.
        inputInfo.numRemainingUses--;
        disableDeallocateActivation(op);
        TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                     "[L1TensorManagement] Conv2d {} consumes fork output "
                     "with {} remaining uses - disabled deallocateActivation",
                     op->getName(), inputInfo.numRemainingUses);
      } else {
        // Safe to deallocate - remove from live set.
        *ctx.currentL1Usage -= inputInfo.sizeBytes;
        ctx.liveTensors->erase(operand);
      }
    }
  }

  // Step 3: Validate current op with L1 pressure and spill if needed.
  // Returns true if validation passed, false otherwise.
  //
  // NOTE: We validate ALL ops including those with L1 sharded output.
  // L1LayoutPropagation validated them but without considering cumulative L1
  // pressure from other live tensors. We need to re-validate with actual L1
  // pressure to catch potential collisions.
  bool validateCurrentOpAndSpillIfNeeded(Operation *op, Value result,
                                         ProcessingContext &ctx, int64_t pos) {
    auto outputLayout = dyn_cast<TTNNLayoutAttr>(
        cast<RankedTensorType>(result.getType()).getEncoding());

    // Validate ALL ops with current L1 pressure, including L1 sharded ops.
    while (true) {
      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(op);

      OpConfig config;
      config.outputLayout = outputLayout;
      populateOpSpecificAttrs(op, config);

      auto validationResult = op_constraint_validation::validateOperation(
          op, inputLayouts, config, *ctx.currentL1Usage);

      if (validationResult.isSuccess()) {
        return true;
      }

      // Validation failed - try to spill something.
      if (ctx.liveTensors->empty()) {
        TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                     "[L1TensorManagement] Op {} validation failed with L1 "
                     "pressure {} but nothing to spill",
                     op->getName(), *ctx.currentL1Usage);
        return false;
      }

      Value spillCandidate = findSpillCandidate(ctx.liveTensors, pos);
      if (!spillCandidate) {
        TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                     "[L1TensorManagement] Op {} validation failed, no "
                     "spill candidate found",
                     op->getName());
        return false;
      }

      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] Op {} validation failed with L1 "
                   "pressure {}, spilling to reduce pressure (candidate from "
                   "liveTensors, already tracked)",
                   op->getName(), *ctx.currentL1Usage);
      spillTensor(spillCandidate, ctx, pos);
    }
  }

  // Step 4: Proactively validate if tensor can stay in L1 until last use.
  // Returns true if all ops in [pos, lastUsePos] can execute with this tensor.
  bool canKeepInL1UntilLastUse(Operation *op, Value tensor, uint64_t tensorSize,
                               int64_t pos, int64_t lastUsePos,
                               ProcessingContext &ctx) {
    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] canKeepInL1UntilLastUse: Op {} "
                 "tensorSize={} bytes, pos={}, lastUsePos={}, "
                 "currentL1Usage={} bytes, liveTensors={}",
                 op->getName(), tensorSize, pos, lastUsePos,
                 *ctx.currentL1Usage, ctx.liveTensors->size());

    // Log all live tensors for debugging
    for (const auto &[value, info] : *ctx.liveTensors) {
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement]   Live tensor: op={} size={} bytes "
                   "lastUsePos={} isFork={}",
                   info.definingOp->getName(), info.sizeBytes,
                   info.lastUsePosition, info.isForkOutput);
    }

    // Validate all ops in the lifetime range can execute with this reservation.
    // Pass the tensor so we don't double-count its L1 usage for consumer ops.
    if (!validateScheduleRangeWithReservation(ctx, pos, lastUsePos, tensorSize,
                                              tensor)) {
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] Op {} output ({} bytes) cannot stay "
                   "in L1 until pos {} - range validation failed",
                   op->getName(), tensorSize, lastUsePos);
      return false;
    }
    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] Op {} CAN stay in L1 until pos {}",
                 op->getName(), lastUsePos);
    return true;
  }

  // Step 4: Add tensor to live set.
  void addToLiveSet(Operation *op, Value result, uint64_t tensorSize,
                    ProcessingContext &ctx, int64_t pos) {
    Operation *lastUseOp =
        findLastUseOp(result, ctx.blockInfo, op, ctx.schedulePos);
    int64_t lastUsePos = pos;
    if (lastUseOp && ctx.schedulePos.count(lastUseOp)) {
      lastUsePos = ctx.schedulePos.at(lastUseOp);
    }

    auto layout = dyn_cast<TTNNLayoutAttr>(
        cast<RankedTensorType>(result.getType()).getEncoding());

    L1TensorInfo info;
    info.value = result;
    info.definingOp = op;
    info.layout = layout;
    info.sizeBytes = tensorSize;
    info.lastUseOp = lastUseOp;
    info.lastUsePosition = lastUsePos;
    info.isForkOutput = isForkPoint(op);
    info.numRemainingUses = countUsers(result);

    (*ctx.liveTensors)[result] = info;
    *ctx.currentL1Usage += tensorSize;
  }

  // Process a single operation.
  void processOp(Operation *op, ProcessingContext &ctx, int64_t pos) {
    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] ========== Processing pos={} op={} "
                 "currentL1Usage={} liveTensors={} ==========",
                 pos, op->getName(), *ctx.currentL1Usage,
                 ctx.liveTensors->size());

    // Step 1: Free expired tensors.
    freeExpiredTensors(ctx, pos);

    // Step 2: Handle conv2d deallocateActivation.
    handleConvDeallocateActivation(op, ctx);

    // Step 3: Check if this op produces a tensor result.
    if (op->getNumResults() == 0) {
      return;
    }

    // Step 3a: Get L1 size for existing L1 tensors (sharded or interleaved).
    uint64_t tensorSize = getL1SizeBytes(op);

    TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                 "[L1TensorManagement] Op {} initial L1 size: {} bytes",
                 op->getName(), tensorSize);

    // Step 3b: If not already in L1, try to promote DRAM -> L1 interleaved.
    if (tensorSize == 0) {
      tensorSize = tryPromoteToL1Interleaved(op, *ctx.currentL1Usage);
      if (tensorSize == 0) {
        TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                     "[L1TensorManagement] Op {} not in L1 and cannot promote "
                     "to L1 interleaved - skipping",
                     op->getName());
        // Not in L1 and can't be promoted - nothing to track.
        return;
      }
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] Op {} promoted to L1 interleaved: "
                   "{} bytes",
                   op->getName(), tensorSize);
    }

    Value result = op->getResult(0);

    // Step 4: Validate current op can execute with current L1 pressure.
    // Spill existing tensors if needed to make room for this op to execute.
    bool currentOpValid =
        validateCurrentOpAndSpillIfNeeded(op, result, ctx, pos);

    if (!currentOpValid) {
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] Op {} cannot execute with L1 "
                   "sharded output under current pressure - needs DRAM "
                   "fallback",
                   op->getName());
      return;
    }

    // Step 5: Calculate last use position.
    Operation *lastUseOp =
        findLastUseOp(result, ctx.blockInfo, op, ctx.schedulePos);
    int64_t lastUsePos = pos;
    if (lastUseOp && ctx.schedulePos.count(lastUseOp)) {
      lastUsePos = ctx.schedulePos.at(lastUseOp);
    }

    // Step 6: Proactively validate if tensor can stay in L1 until last use.
    // This checks all ops in [pos, lastUsePos] can execute with this tensor.
    if (!canKeepInL1UntilLastUse(op, result, tensorSize, pos, lastUsePos, ctx)) {
      // Cannot keep in L1 - spill immediately to DRAM.
      // The output is already L1 sharded from Pass 1, but we need to
      // insert a ToMemoryConfigOp right after to move it to DRAM.
      TTMLIR_TRACE(ttmlir::LogComponent::L1Optimizer,
                   "[L1TensorManagement] ***SPILL DECISION*** Op {} output "
                   "({} bytes) will be spilled to DRAM immediately because "
                   "range validation failed. pos={} lastUsePos={} "
                   "currentL1Usage={} l1Budget={}",
                   op->getName(), tensorSize, pos, lastUsePos,
                   *ctx.currentL1Usage, ctx.l1Budget);

      // Create temporary info for immediate spill.
      L1TensorInfo tempInfo;
      tempInfo.value = result;
      tempInfo.definingOp = op;
      tempInfo.layout = dyn_cast<TTNNLayoutAttr>(
          cast<RankedTensorType>(result.getType()).getEncoding());
      tempInfo.sizeBytes = tensorSize;
      tempInfo.lastUseOp = lastUseOp;
      tempInfo.lastUsePosition = lastUsePos;
      tempInfo.isForkOutput = isForkPoint(op);
      tempInfo.numRemainingUses = countUsers(result);

      // Add temporarily to trigger spill logic.
      // Also add to currentL1Usage so spillTensor's subtraction brings it back
      // to the original value (tensor was never tracked in currentL1Usage).
      (*ctx.liveTensors)[result] = tempInfo;
      *ctx.currentL1Usage += tensorSize;
      spillTensor(result, ctx, pos);
      return;
    }

    // Step 7: All validations passed - add to live set.
    addToLiveSet(op, result, tensorSize, ctx, pos);
  }

  //===--------------------------------------------------------------------===//
  // Schedule building
  //===--------------------------------------------------------------------===//

  void buildSchedule(func::FuncOp funcOp,
                     llvm::SmallVector<Operation *> &schedule,
                     llvm::DenseMap<Operation *, int64_t> &schedulePos) {
    funcOp.walk([&](Operation *op) {
      if (isa<TTNNDialect>(op->getDialect())) {
        schedulePos[op] = static_cast<int64_t>(schedule.size());
        schedule.push_back(op);
      }
    });
  }

  //===--------------------------------------------------------------------===//
  // Main function processing
  //===--------------------------------------------------------------------===//

  void processFunction(func::FuncOp funcOp, uint64_t l1Budget) {
    // Build liveness analysis.
    Liveness liveness(funcOp);
    const LivenessBlockInfo *blockInfo =
        liveness.getLiveness(&funcOp.getBody().front());
    if (!blockInfo) {
      return;
    }

    // Build schedule.
    llvm::SmallVector<Operation *> schedule;
    llvm::DenseMap<Operation *, int64_t> schedulePos;
    buildSchedule(funcOp, schedule, schedulePos);

    // Initialize state.
    IRRewriter rewriter(&getContext());
    llvm::DenseMap<Value, L1TensorInfo> liveTensors;
    uint64_t currentL1Usage = 0;

    // Create processing context.
    ProcessingContext ctx(rewriter, blockInfo, schedule, schedulePos,
                          liveTensors, currentL1Usage, l1Budget);

    // Process each operation in schedule order.
    for (int64_t pos = 0; pos < static_cast<int64_t>(schedule.size()); ++pos) {
      processOp(schedule[pos], ctx, pos);
    }

    // Post-process: fix deallocate_activation for Conv2d ops whose input still
    // has multiple users. This runs after spilling so we account for actual IR
    // state - if fork edges were spilled, Conv2d may now be the only user.
    fixConvDeallocateActivationForForks(funcOp);
  }
};

} // namespace mlir::tt::ttnn
