// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Utils.h"
#include <atomic>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNTRACEHOISTTRANSFORM
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

using TraceSmallString = llvm::SmallString<64>;

class TTNNTraceHoistTransform
    : public impl::TTNNTraceHoistTransformBase<TTNNTraceHoistTransform> {
public:
  using impl::TTNNTraceHoistTransformBase<
      TTNNTraceHoistTransform>::TTNNTraceHoistTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = this->getOperation();
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (failed(processFuncOp(funcOp))) {
        signalPassFailure();
      }
    });
  }

private:
  bool shouldHoistOp(Operation *op) {
    bool shouldHoist = true;
    shouldHoist &= !::mlir::isa<func::ReturnOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttcore::LoadCachedOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttnn::CaptureOrExecuteTraceOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttnn::GetDeviceOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttnn::MeshShardOp>(op);
    shouldHoist &=
        !(op->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>());
    return shouldHoist;
  }

  std::uint64_t getUniqueTraceFuncIndex() {
    static std::atomic<std::uint64_t> traceFunctionIndex = 0;
    return traceFunctionIndex.fetch_add(1, std::memory_order_relaxed);
  }

  TraceSmallString getTraceFuncName(func::FuncOp funcOp,
                                    uint64_t traceFuncIndex) {
    TraceSmallString traceFuncName("trace");
    traceFuncName.append("_" + std::to_string(traceFuncIndex) + "_");
    traceFuncName.append(funcOp.getName().str());
    return traceFuncName;
  }

  TraceSmallString getCaptureTraceFuncName(func::FuncOp funcOp,
                                           uint64_t traceFuncIndex) {
    TraceSmallString runAndCaptureTraceFuncName;
    runAndCaptureTraceFuncName.append(g_TTNNCaptureTracePrefix);
    runAndCaptureTraceFuncName.append(getTraceFuncName(funcOp, traceFuncIndex));
    return runAndCaptureTraceFuncName;
  }

  TraceSmallString getExecuteTraceFuncName(func::FuncOp funcOp,
                                           uint64_t traceFuncIndex) {
    TraceSmallString executeTraceFuncName;
    executeTraceFuncName.append(g_TTNNExecuteTracePrefix);
    executeTraceFuncName.append(getTraceFuncName(funcOp, traceFuncIndex));
    return executeTraceFuncName;
  }

  // Returns true if the argument should remain on device during trace capture.
  // These arguments (constants, parameters, and KV cache tensors) are persisted
  // on device and used directly without creating temporary slots or
  // transferring through system memory. This avoids unnecessary data movement
  // for:
  // - Constants/parameters: Already on device and immutable
  // - KV cache tensors: Device-native and updated in-place by cache operations
  bool shouldKeepArgOnDevice(func::FuncOp op, size_t argIndex) {
    return ttcore::isConstantOrParameterArgumentType(op, argIndex) ||
           ttcore::isKVCacheArgument(op, argIndex);
  }

  // Collect all inputs and outputs outside the operation set to hoist
  void collectFunctionBoundary(llvm::ArrayRef<Operation *> opsToHoist,
                               llvm::SmallVector<mlir::Value> &inputs,
                               llvm::SmallVector<mlir::Value> &outputs) {

    // Create set for quick lookup
    llvm::SmallPtrSet<Operation *, 16> opSet(opsToHoist.begin(),
                                             opsToHoist.end());
    llvm::SmallPtrSet<mlir::Value, 16> seenInputs;

    // Collect inputs: operands that come from outside the operation set
    for (Operation *op : opsToHoist) {
      for (auto operand : op->getOperands()) {
        if (!::mlir::isa<RankedTensorType>(operand.getType())) {
          continue;
        }
        Operation *definingOp = operand.getDefiningOp();
        if (!definingOp || !opSet.contains(definingOp)) {
          if (seenInputs.insert(operand).second) {
            inputs.push_back(operand);
          }
        }
      }
    }

    llvm::sort(inputs.begin(), inputs.end(), [](mlir::Value a, mlir::Value b) {
      // prioritize block arguments
      // this is ok now since we check that the funcOp has only 1 block
      // should be updated if we support multiple blocks in the future
      if (::mlir::isa<mlir::BlockArgument>(a) &&
          ::mlir::isa<mlir::BlockArgument>(b)) {
        return ::mlir::cast<mlir::BlockArgument>(a).getArgNumber() <
               ::mlir::cast<mlir::BlockArgument>(b).getArgNumber();
      }
      if (::mlir::isa<mlir::BlockArgument>(a)) {
        return true;
      }
      if (::mlir::isa<mlir::BlockArgument>(b)) {
        return false;
      }

      auto aResult = ::mlir::cast<mlir::OpResult>(a);
      auto bResult = ::mlir::cast<mlir::OpResult>(b);

      if (aResult.getOwner() == bResult.getOwner()) {
        return aResult.getResultNumber() < bResult.getResultNumber();
      }
      return aResult.getOwner()->isBeforeInBlock(bResult.getOwner());
    });

    // Collect outputs: results used outside the operation set
    for (Operation *op : opsToHoist) {
      for (auto result : op->getResults()) {
        for (auto &use : result.getUses()) {
          Operation *user = use.getOwner();
          if (!opSet.contains(user)) {
            outputs.push_back(result);
            break;
          }
        }
      }
    }
  }

  llvm::SmallVector<mlir::DictionaryAttr>
  getInputAttrs(MLIRContext *context, llvm::ArrayRef<mlir::Value> inputs) {
    llvm::SmallVector<mlir::DictionaryAttr> inputAttrs;
    for (mlir::Value input : inputs) {
      mlir::DictionaryAttr attrs = mlir::DictionaryAttr::get(context);
      if (mlir::isa<mlir::BlockArgument>(input)) {
        // Inherit the arg attributes from the function
        auto arg = mlir::cast<mlir::BlockArgument>(input);
        if (auto funcOp =
                mlir::dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
          attrs = funcOp.getArgAttrDict(arg.getArgNumber());
        }
      } else if (mlir::isa<mlir::OpResult>(input)) {
        auto result = mlir::cast<mlir::OpResult>(input);
        Operation *defOp = result.getDefiningOp();
        // If the input is a result of a load cached op
        // Then we can mark it as a constant since it's a consteval result
        if (mlir::isa<mlir::tt::ttcore::LoadCachedOp>(defOp)) {
          llvm::SmallVector<mlir::NamedAttribute> namedAttrs;
          namedAttrs.emplace_back(
              mlir::StringAttr::get(context, ttcore::ArgumentTypeAttr::name),
              ttcore::ArgumentTypeAttr::get(context,
                                            ttcore::ArgumentType::Constant));
          attrs = mlir::DictionaryAttr::get(context, namedAttrs);
        }
      }
      inputAttrs.push_back(attrs);
    }
    return inputAttrs;
  }

  // Creates the trace function
  ::mlir::LogicalResult
  createTraceFunction(func::FuncOp funcOp,
                      llvm::ArrayRef<Operation *> opsToHoist,
                      uint64_t traceFuncIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Type> inputTypes;
    llvm::SmallVector<mlir::Value> outputs;
    llvm::SmallVector<mlir::Type> outputTypes;

    collectFunctionBoundary(opsToHoist, inputs, outputs);

    for (mlir::Value input : inputs) {
      inputTypes.push_back(input.getType());
    }
    for (mlir::Value output : outputs) {
      outputTypes.push_back(output.getType());
    }

    llvm::SmallVector<mlir::DictionaryAttr> inputAttrs =
        getInputAttrs(context, inputs);

    TraceSmallString traceFuncName = getTraceFuncName(funcOp, traceFuncIndex);

    auto traceFuncType = builder.getFunctionType(inputTypes, outputTypes);

    // Create the function
    builder.setInsertionPoint(funcOp);
    auto traceFuncOp = builder.create<func::FuncOp>(
        funcOp.getLoc(), traceFuncName, traceFuncType);
    ttmlir::utils::setFunctionType(traceFuncOp,
                                   ttmlir::utils::FunctionType::TraceMain);

    traceFuncOp.setAllArgAttrs(inputAttrs);
    traceFuncOp.setPrivate();

    // Build the body of the new function
    auto *traceFuncEntryBlock = traceFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(traceFuncEntryBlock);

    // maps original input values to trace function input
    // arguments/intermediates
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (size_t i = 0; i < inputs.size(); i++) {
      valueMap.insert({inputs[i], traceFuncOp.getArgument(i)});
    }

    for (Operation *op : opsToHoist) {
      // clone the operation into the trace function
      Operation *clonedOp = builder.clone(*op);

      // Update the op's operands with trace function input arguments
      for (size_t i = 0; i < clonedOp->getNumOperands(); i++) {
        auto originalOperand = op->getOperand(i);
        auto it = valueMap.find(originalOperand);
        if (it != valueMap.end()) {
          clonedOp->setOperand(i, it->second);
          continue;
        }
        // Special case where an op has a device operand.
        // In this case, we need to insert a GetDeviceOp within the trace
        // function. The verifier will ensure that this device matches the
        // device of the trace op.
        if (::mlir::isa<DeviceType>(originalOperand.getType())) {
          auto device = utils::getOrInsertDevice(rewriter, clonedOp);
          clonedOp->setOperand(i, device);
          continue;
        }
        return funcOp.emitError("Could not map operand in hoisted function");
      }

      // Update the op's results with trace function op output result
      for (size_t i = 0; i < op->getNumResults(); i++) {
        valueMap[op->getResult(i)] = clonedOp->getResult(i);
      }
    }

    // Finally, we need to add a return operation to the trace function
    llvm::SmallVector<mlir::Value> returnValues;
    for (mlir::Value output : outputs) {
      auto it = valueMap.find(output);
      if (it != valueMap.end()) {
        returnValues.push_back(it->second);
      } else {
        return funcOp.emitError(
            "Could not map output value in hoisted function");
      }
    }
    builder.create<func::ReturnOp>(funcOp.getLoc(), returnValues);

    return mlir::success();
  }

  // Creates the run and capture function that wraps the trace function and
  // manages trace capture.
  mlir::LogicalResult
  createRunAndCaptureTraceFunction(func::FuncOp funcOp,
                                   uint64_t traceFuncIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    TraceSmallString traceFuncName = getTraceFuncName(funcOp, traceFuncIndex);
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    func::FuncOp traceFunc = moduleOp.lookupSymbol<func::FuncOp>(traceFuncName);
    if (!traceFunc) {
      return funcOp.emitError("Could not find trace function with name: " +
                              traceFuncName);
    }

    // Build input types for the capture function and corresponding device slot
    // types. The capture function accepts:
    // - Regular inputs from host memory that need to be transferred to device
    // - Constants/parameters that are already on device (persisted)
    // - KV cache tensors that are device-native and updated in-place
    // For each argument, we determine the appropriate input type and slot
    // allocation strategy.
    llvm::SmallVector<mlir::Type> inputTypes;
    llvm::SmallVector<mlir::Type> traceInputSlotTypes;
    for (size_t i = 0; i < traceFunc.getNumArguments(); i++) {
      mlir::Value traceFuncArg = traceFunc.getArgument(i);

      RankedTensorType originalRankedTensorType =
          mlir::cast<RankedTensorType>(traceFuncArg.getType());

      // Device-resident arguments (constants, parameters, KV cache) bypass host
      // transfer. These are already on device and will be used directly as
      // trace input slots.
      if (shouldKeepArgOnDevice(traceFunc, i)) {
        assert(utils::getBufferTypeFromTensor(originalRankedTensorType) ==
                   ttnn::BufferType::DRAM &&
               "Device-resident arguments must already be in device memory.");
        inputTypes.push_back(traceFuncArg.getType());
        traceInputSlotTypes.push_back(traceFuncArg.getType());
        continue;
      }

      // Regular input arguments require host-to-device transfer during trace
      // capture. We create two types for each regular input:
      // 1. Host-side type (system memory) for the function signature
      // 2. Device-side slot type (DRAM) for persistent trace storage
      RankedTensorType inputArgType = utils::RankedTensorTypeFactory::create(
          originalRankedTensorType, BufferType::SystemMemory);

      inputTypes.push_back(inputArgType);

      // Create the new output result type with the updated buffer type for the
      // trace input slot for this argument on host.
      RankedTensorType dramTraceInputSlotType =
          utils::RankedTensorTypeFactory::create(originalRankedTensorType,
                                                 BufferType::DRAM);

      traceInputSlotTypes.push_back(dramTraceInputSlotType);
    }

    // The capture function returns multiple values for trace management:
    // 1. traceId - Identifier for the captured trace
    // 2. actual outputs - Results from the first execution (non-traced)
    // 3. trace input slots - Persistent device memory for input data
    // 4. trace output slots - Persistent device memory for output data
    llvm::SmallVector<mlir::Type> outputTypes;

    // Trace ID for the captured trace, used for correlation between capture and
    // execution.
    outputTypes.push_back(utils::getTraceIdType(context));

    // Actual outputs from the first execution of the trace function
    // (non-traced).
    for (mlir::Type outputType : traceFunc.getFunctionType().getResults()) {
      outputTypes.push_back(outputType);
    }

    // Trace input slots for all inputs (including constants/parameters and KV
    // cache) that are persisted on device.
    for (mlir::Type traceInputSlotType : traceInputSlotTypes) {
      outputTypes.push_back(traceInputSlotType);
    }

    // Trace output slots for all outputs that will be captured on device.
    for (mlir::Type outputType : traceFunc.getFunctionType().getResults()) {
      outputTypes.push_back(outputType);
    }

    // Create and insert function.
    auto runAndCaptureTraceFuncType =
        builder.getFunctionType(inputTypes, outputTypes);

    TraceSmallString runAndCaptureTraceFuncName =
        getCaptureTraceFuncName(funcOp, traceFuncIndex);

    builder.setInsertionPoint(funcOp);
    auto runAndCaptureTraceFunc = builder.create<func::FuncOp>(
        funcOp.getLoc(), runAndCaptureTraceFuncName,
        runAndCaptureTraceFuncType);
    ttmlir::utils::setFunctionType(
        runAndCaptureTraceFunc,
        ttmlir::utils::FunctionType::TraceRunAndCapture);
    if (traceFunc.getAllArgAttrs()) {
      runAndCaptureTraceFunc.setAllArgAttrs(traceFunc.getAllArgAttrs());
    }
    runAndCaptureTraceFunc.setPrivate();

    // Build the body of the function.
    auto *runAndCaptureTraceFuncEntryBlock =
        runAndCaptureTraceFunc.addEntryBlock();
    builder.setInsertionPointToStart(runAndCaptureTraceFuncEntryBlock);

    auto deviceOp =
        utils::getOrInsertDevice(rewriter, runAndCaptureTraceFuncEntryBlock);
    auto device = ttcore::lookupDevice(deviceOp);

    // Create or reuse trace input slots on device.
    // - Device-resident args (constants/parameters/KV cache): use directly
    // - Regular inputs: allocate new empty tensors on device for data transfer
    llvm::SmallVector<mlir::Value> traceInputSlots;
    for (size_t i = 0; i < runAndCaptureTraceFunc.getNumArguments(); i++) {
      if (shouldKeepArgOnDevice(traceFunc, i)) {
        traceInputSlots.push_back(runAndCaptureTraceFunc.getArgument(i));
        continue;
      }

      // Regular inputs need device memory allocation for host-to-device
      // transfer. Create empty tensors on device that will serve as persistent
      // slots for trace input data during capture and replay.
      mlir::Type traceInputSlotType = traceInputSlotTypes[i];

      RankedTensorType deviceTensorType =
          mlir::cast<RankedTensorType>(traceInputSlotType);

      ttnn::TTNNLayoutAttr ttnnLayoutAttr =
          utils::getLayoutAttrFromTensor(deviceTensorType);
      ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
          context, ttnnLayoutAttr.getMemLayout(),
          ttnn::BufferTypeAttr::get(context, ttnnLayoutAttr.getBufferType()),
          utils::createShardSpecIfNeeded(ttnnLayoutAttr,
                                         device.getWorkerGrid()));

      // Allocate an empty tensor on the device to serve as the trace input slot
      // for this argument.
      auto emptyOp = builder.create<ttnn::EmptyOp>(
          runAndCaptureTraceFunc.getLoc(), deviceTensorType, deviceOp,
          ttnn::ShapeAttr::get(context, deviceTensorType.getShape()),
          ttcore::DataTypeAttr::get(context, ttnnLayoutAttr.getDataType()),
          ttnn::LayoutAttr::get(context, ttnnLayoutAttr.getLayout()),
          memoryConfigAttr);

      traceInputSlots.push_back(emptyOp.getResult());
    }

    // Transfer host inputs to their corresponding device slots.
    // Device-resident arguments are skipped as they're already in place.
    for (size_t i = 0; i < traceInputSlots.size(); i++) {
      if (shouldKeepArgOnDevice(traceFunc, i)) {
        continue;
      }

      // Copy the input argument from host to the allocated device slot for this
      // input.
      mlir::Value input = runAndCaptureTraceFunc.getArgument(i);

      builder.create<ttnn::WriteTensorOp>(runAndCaptureTraceFunc.getLoc(),
                                          input, traceInputSlots[i],
                                          /*blocking=*/false, /*cq_id=*/0);
    }

    // Execute the trace function once without capture to compile programs and
    // populate program cache. The results are discarded since this execution is
    // just for warming up.
    auto traceFuncCall = builder.create<func::CallOp>(
        runAndCaptureTraceFunc.getLoc(), traceFunc, traceInputSlots);

    // Start capturing the trace.
    auto beginTraceCaptureOp = builder.create<ttnn::BeginTraceCaptureOp>(
        runAndCaptureTraceFunc.getLoc(), utils::getTraceIdType(context),
        deviceOp,
        /*cq_id=*/0);

    // Execute the trace on device and capture it.
    auto captureTraceCall = builder.create<func::CallOp>(
        runAndCaptureTraceFunc.getLoc(), traceFunc, traceInputSlots);

    // Complete the trace capture.
    builder.create<ttnn::EndTraceCaptureOp>(runAndCaptureTraceFunc.getLoc(),
                                            deviceOp, beginTraceCaptureOp,
                                            /*cq_id=*/0);

    // Assemble return values: trace ID, actual outputs, and persistent slots.
    llvm::SmallVector<mlir::Value> returnValues;

    // Return the trace ID for correlation with execution.
    returnValues.push_back(beginTraceCaptureOp.getTraceId());
    // Return the actual outputs from the first execution (non-traced).
    for (mlir::Value output : traceFuncCall.getResults()) {
      returnValues.push_back(output);
    }
    // Return the trace input slots for all inputs (including
    // constants/parameters and KV cache) that are persisted on device.
    for (mlir::Value inputSlot : traceInputSlots) {
      returnValues.push_back(inputSlot);
    }
    // Return the trace output slots for all outputs that will be captured on
    // device.
    for (mlir::Value outputSlot : captureTraceCall.getResults()) {
      returnValues.push_back(outputSlot);
    }

    builder.create<func::ReturnOp>(runAndCaptureTraceFunc.getLoc(),
                                   returnValues);

    return mlir::success();
  }

  mlir::LogicalResult createExecuteTraceFunction(func::FuncOp funcOp,
                                                 uint64_t traceFuncIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    TraceSmallString traceFuncName = getTraceFuncName(funcOp, traceFuncIndex);
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    func::FuncOp traceFunc = moduleOp.lookupSymbol<func::FuncOp>(traceFuncName);
    if (!traceFunc) {
      return funcOp.emitError("Could not find trace function with name: " +
                              traceFuncName);
    }

    llvm::SmallVector<mlir::Type> inputTypes;
    inputTypes.push_back(utils::getTraceIdType(context));

    llvm::SmallVector<mlir::Type> outputTypes;

    // Create and insert function
    auto executeTraceFuncType =
        builder.getFunctionType(inputTypes, outputTypes);

    TraceSmallString executeTraceFuncName =
        getExecuteTraceFuncName(funcOp, traceFuncIndex);

    builder.setInsertionPoint(funcOp);
    auto executeTraceFunc = builder.create<func::FuncOp>(
        funcOp.getLoc(), executeTraceFuncName, executeTraceFuncType);
    ttmlir::utils::setFunctionType(executeTraceFunc,
                                   ttmlir::utils::FunctionType::TraceExecute);
    executeTraceFunc.setPrivate();

    // Build the body of the function
    auto *executeTraceFuncEntryBlock = executeTraceFunc.addEntryBlock();
    builder.setInsertionPointToStart(executeTraceFuncEntryBlock);

    auto deviceOp =
        utils::getOrInsertDevice(rewriter, executeTraceFuncEntryBlock);
    mlir::Value traceId = executeTraceFunc.getArgument(0);
    builder.create<ttnn::ExecuteTraceOp>(funcOp.getLoc(), deviceOp, traceId,
                                         /*cq_id=*/0, /*blocking=*/false);

    builder.create<func::ReturnOp>(funcOp.getLoc());

    return mlir::success();
  }

  // Optimizes function signatures by merging ToLayoutOp operations into
  // function arguments.
  //
  // Background: During trace hoisting, when we move operations into a separate
  // trace function, the function arguments initially preserve their original
  // types. However, these arguments are often immediately converted to
  // different layouts (e.g., from system memory to device memory layouts) via
  // ToLayoutOp operations that were inserted during earlier passes.
  //
  // This optimization identifies such patterns where:
  // 1. A function argument is immediately converted via ToLayoutOp
  // 2. A function argument goes through MeshShardOp then ToLayoutOp
  //
  // By updating the function signature to accept the target layout directly,
  // we:
  // - Eliminate redundant layout conversion operations inside the trace
  // function
  // - Move the layout conversion responsibility to the caller
  // - Reduce the overhead during trace execution (traces should be as lean as
  // possible)
  mlir::LogicalResult mergeToLayoutOpsWithFuncArgs(func::FuncOp funcOp) {
    // Only process forward device functions.
    auto funcType = ttmlir::utils::getFunctionType(funcOp);
    if (!funcType || *funcType != ttmlir::utils::FunctionType::ForwardDevice) {
      return mlir::success();
    }

    mlir::OpBuilder builder(&this->getContext());
    mlir::IRRewriter rewriter(builder);

    bool hasChanges = false;
    auto &entryBlock = funcOp.getBlocks().front();
    llvm::SmallVector<mlir::Type> newInputTypes;
    llvm::SmallVector<mlir::Operation *> opsToErase;

    // Scan each function argument to find layout conversion patterns that can
    // be optimized. We look for arguments that are immediately used by
    // ToLayoutOp (directly or via MeshShardOp).
    for (size_t argIdx = 0; argIdx < funcOp.getNumArguments(); argIdx++) {
      BlockArgument arg = funcOp.getArgument(argIdx);
      RankedTensorType currentTensorType =
          mlir::cast<RankedTensorType>(arg.getType());

      ttnn::ToLayoutOp layoutOp = nullptr;
      ttnn::MeshShardOp meshShardOp = nullptr;

      // Check if argument has only one use.
      if (arg.hasOneUse()) {
        auto *user = *arg.getUsers().begin();

        // Check if it's a direct ToLayoutOp.
        if (auto directLayoutOp = mlir::dyn_cast<ttnn::ToLayoutOp>(user)) {
          layoutOp = directLayoutOp;
        }
        // Check if it's a MeshShardOp that leads to ToLayoutOp.
        else if (auto meshShard = mlir::dyn_cast<ttnn::MeshShardOp>(user)) {
          meshShardOp = meshShard;
          // Check if mesh_shard has a single use and it's a ToLayoutOp.
          if (meshShard.getResult().hasOneUse()) {
            auto *meshUser = *meshShard.getResult().getUsers().begin();
            if (auto toLayout = mlir::dyn_cast<ttnn::ToLayoutOp>(meshUser)) {
              layoutOp = toLayout;
            }
          }
        }
      }

      // If there's no ToLayoutOp pattern, keep the original type.
      if (!layoutOp) {
        newInputTypes.push_back(currentTensorType);
        continue;
      }

      // Get the target type from the ToLayoutOp.
      RankedTensorType targetTensorType = layoutOp.getResult().getType();

      if (meshShardOp) {
        // Case 1: Argument -> MeshShardOp -> ToLayoutOp
        // We need to update the function argument type and the mesh_shard
        // operation. The new input type should match the layout after
        // ToLayoutOp.

        // Create new type for function argument with the target layout
        // but keeping the original shape (before mesh_shard).
        TTNNLayoutAttr targetLayoutAttr =
            utils::getLayoutAttrFromTensor(targetTensorType);
        TTNNLayoutAttr currentLayoutAttr =
            utils::getLayoutAttrFromTensor(currentTensorType);
        assert(targetLayoutAttr.getDataType() ==
                   currentLayoutAttr.getDataType() &&
               "The data type should be the same since added ToLayoutOp only "
               "changed buffer type.");
        auto newArgType = utils::RankedTensorTypeFactory::create(
            currentTensorType, targetLayoutAttr.getBufferType());
        newArgType = utils::RankedTensorTypeFactory::create(
            newArgType, targetLayoutAttr.getLayout());

        newInputTypes.push_back(newArgType);

        // Update mesh_shard's result type to match the target layout.
        // This must be done before replacing uses to maintain type consistency.
        meshShardOp.getResult().setType(targetTensorType);

        // Now we can safely replace ToLayoutOp uses with mesh_shard output.
        layoutOp.getResult().replaceAllUsesWith(meshShardOp.getResult());
        opsToErase.push_back(layoutOp);
      } else {
        // Case 2: Argument -> ToLayoutOp
        // We can directly update the function argument type to the target type.
        TTNNLayoutAttr targetLayoutAttr =
            utils::getLayoutAttrFromTensor(targetTensorType);
        TTNNLayoutAttr currentLayoutAttr =
            utils::getLayoutAttrFromTensor(currentTensorType);
        assert(targetLayoutAttr.getDataType() ==
                   currentLayoutAttr.getDataType() &&
               "The data type should be the same since added ToLayoutOp only "
               "changed buffer type.");
        newInputTypes.push_back(targetTensorType);

        // Replace all uses of ToLayoutOp with the function argument
        layoutOp.getResult().replaceAllUsesWith(arg);
        opsToErase.push_back(layoutOp);
      }

      hasChanges = true;
    }

    if (hasChanges) {
      // Update function signature.
      auto funcType = funcOp.getFunctionType();
      auto newFuncType =
          builder.getFunctionType(newInputTypes, funcType.getResults());
      funcOp.setFunctionType(newFuncType);

      // Update block argument types.
      for (size_t i = 0; i < newInputTypes.size(); i++) {
        entryBlock.getArgument(i).setType(newInputTypes[i]);
      }

      // Erase the ToLayoutOps.
      for (auto *op : opsToErase) {
        rewriter.eraseOp(op);
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult
  insertCaptureOrExecuteTraceOp(func::FuncOp funcOp,
                                llvm::ArrayRef<Operation *> opsToHoist,
                                uint64_t traceFuncIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    TraceSmallString captureTraceFuncName =
        getCaptureTraceFuncName(funcOp, traceFuncIndex);
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    func::FuncOp captureTraceFunc =
        moduleOp.lookupSymbol<func::FuncOp>(captureTraceFuncName);
    if (!captureTraceFunc) {
      return funcOp.emitError(
          "Could not find capture trace function with name: " +
          captureTraceFuncName);
    }

    TraceSmallString executeTraceFuncName =
        getExecuteTraceFuncName(funcOp, traceFuncIndex);
    func::FuncOp executeTraceFunc =
        moduleOp.lookupSymbol<func::FuncOp>(executeTraceFuncName);
    if (!executeTraceFunc) {
      return funcOp.emitError(
          "Could not find execute trace function with name: " +
          executeTraceFuncName);
    }

    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Value> outputs;
    llvm::SmallVector<mlir::Type> outputTypes;

    collectFunctionBoundary(opsToHoist, inputs, outputs);

    for (mlir::Value output : outputs) {
      outputTypes.push_back(output.getType());
    }

    auto captureTraceSymbolAttr =
        mlir::SymbolRefAttr::get(context, captureTraceFuncName);
    auto executeTraceSymbolAttr =
        mlir::SymbolRefAttr::get(context, executeTraceFuncName);

    Operation *firstOp = opsToHoist.front();

    builder.setInsertionPoint(firstOp);

    auto device = utils::getOrInsertDevice(rewriter, firstOp);

    // Convert inputs to match capture function signature
    llvm::SmallVector<mlir::Value> captureOrExecuteTraceOpInputs;

    for (size_t i = 0; i < inputs.size(); i++) {
      mlir::Value input = inputs[i];
      if (!mlir::isa<RankedTensorType>(input.getType())) {
        captureOrExecuteTraceOpInputs.push_back(input);
        continue;
      }

      // Check if this is a constant/parameter that should remain on device
      bool isConstant = false;
      if (mlir::isa<mlir::BlockArgument>(input)) {
        auto arg = mlir::cast<mlir::BlockArgument>(input);
        if (auto funcOp =
                mlir::dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp())) {
          isConstant = shouldKeepArgOnDevice(funcOp, arg.getArgNumber());
        }
      } else if (auto result = mlir::dyn_cast<mlir::OpResult>(input)) {
        // Check if it's from a load_cached op (constant evaluation result)
        isConstant =
            mlir::isa<mlir::tt::ttcore::LoadCachedOp>(result.getDefiningOp());
      }

      RankedTensorType tensorType =
          mlir::cast<RankedTensorType>(input.getType());
      auto layout = mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

      // Constants must be on device, they can be captured directly without
      // needing to move them to system memory.
      if (isConstant) {
        assert(layout.getBufferType() == ttnn::BufferType::DRAM &&
               "Constant/parameter inputs must be on device.");
        captureOrExecuteTraceOpInputs.push_back(input);
      }
      // For inputs, convert them to system memory/row major if needed
      else if (layout.getBufferType() != ttnn::BufferType::SystemMemory) {
        // Convert to system memory using ToLayoutOp
        RankedTensorType systemMemoryTileType =
            utils::RankedTensorTypeFactory::create(
                tensorType, ttnn::BufferType::SystemMemory);

        // Create memory config for system memory
        auto memoryConfigAttr = ttnn::MemoryConfigAttr::get(
            context, nullptr,
            ttnn::BufferTypeAttr::get(context, ttnn::BufferType::SystemMemory),
            /*shard_spec=*/nullptr);

        auto toLayoutOp = builder.create<ttnn::ToLayoutOp>(
            funcOp.getLoc(), systemMemoryTileType, input,
            /*layout=*/LayoutAttr::get(context, layout.getLayout()),
            /*dtype=*/ttcore::DataTypeAttr::get(context, layout.getDataType()),
            /*memory_config=*/memoryConfigAttr);
        captureOrExecuteTraceOpInputs.push_back(toLayoutOp.getResult());
      } else {
        // Already on system memory
        captureOrExecuteTraceOpInputs.push_back(input);
      }
    }

    auto traceOp = builder.create<ttnn::CaptureOrExecuteTraceOp>(
        funcOp.getLoc(), outputTypes, device, captureTraceSymbolAttr,
        executeTraceSymbolAttr, captureOrExecuteTraceOpInputs);

    // Replace uses of original outputs with the output of the trace op function
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].replaceAllUsesWith(traceOp->getResult(i));
    }

    // Remove the original ops in reverse order (to avoid dependency issues)
    for (auto it = opsToHoist.rbegin(); it != opsToHoist.rend(); it++) {
      rewriter.eraseOp(*it);
    }

    return mlir::success();
  }

  mlir::LogicalResult
  performHoistTransform(func::FuncOp funcOp,
                        llvm::ArrayRef<Operation *> opsToHoist) {
    uint64_t traceFuncIndex = getUniqueTraceFuncIndex();
    // Create trace function and trace op if there are ops to hoist
    ::mlir::LogicalResult result =
        createTraceFunction(funcOp, opsToHoist, traceFuncIndex);
    if (failed(result)) {
      return result;
    }

    result = createRunAndCaptureTraceFunction(funcOp, traceFuncIndex);
    if (failed(result)) {
      return result;
    }

    result = createExecuteTraceFunction(funcOp, traceFuncIndex);
    if (failed(result)) {
      return result;
    }

    result = insertCaptureOrExecuteTraceOp(funcOp, opsToHoist, traceFuncIndex);
    if (failed(result)) {
      return result;
    }

    return mlir::success();
  }

  mlir::LogicalResult processFuncOp(func::FuncOp funcOp) {
    // Skip non-forward functions.
    if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
      return mlir::success();
    }

    if (funcOp.getBlocks().size() != 1) {
      return funcOp.emitError("FuncOp should have exactly one block");
    }

    llvm::SmallVector<Operation *> opsToHoist;

    mlir::Block &block = funcOp.getBlocks().front();

    // Collect all hoistable ops, but skip the first non-hoistable ops and the
    // last non-hoistable ops. Non-hoistable ops at the boundaries should remain
    // outside the trace
    bool startedCollecting = false;
    llvm::SmallVector<Operation *> allOps;
    for (mlir::Operation &op : block.getOperations()) {
      if (!::mlir::isa<func::ReturnOp>(op)) {
        allOps.push_back(&op);
      }
    }

    // Find the first hoistable op
    size_t firstHoistable = 0;
    for (size_t i = 0; i < allOps.size(); i++) {
      if (shouldHoistOp(allOps[i])) {
        firstHoistable = i;
        startedCollecting = true;
        break;
      }
    }

    // If we found hoistable ops, collect them until we hit non-hoistable ops at
    // the end
    if (startedCollecting) {
      // Find the last hoistable op (before any trailing non-hoistable ops)
      size_t lastHoistable = firstHoistable;
      for (size_t i = allOps.size() - 1; i > firstHoistable; i--) {
        if (shouldHoistOp(allOps[i])) {
          lastHoistable = i;
          break;
        }
      }

      // Collect all hoistable ops between first and last
      for (size_t i = firstHoistable; i <= lastHoistable; i++) {
        if (shouldHoistOp(allOps[i])) {
          opsToHoist.push_back(allOps[i]);
        } else {
          // We found a non-hoistable op in the middle - this is an error
          return allOps[i]->emitError(
              "Non-hoistable op found in the middle of hoistable ops");
        }
      }
    }

    if (opsToHoist.empty()) {
      return mlir::success();
    }

    // Perform the hoist transform
    mlir::LogicalResult result = performHoistTransform(funcOp, opsToHoist);
    if (failed(result)) {
      return result;
    }

    if (failed(mergeToLayoutOpsWithFuncArgs(funcOp))) {
      return result;
    }

    return mlir::success();
  }
};
} // namespace mlir::tt::ttnn
