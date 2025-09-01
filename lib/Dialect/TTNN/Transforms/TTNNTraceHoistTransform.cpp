// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
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

  bool isConstantOrParameter(func::FuncOp op, size_t argIndex) {
    auto argAttrDict = op.getArgAttrDict(argIndex);
    if (argAttrDict && argAttrDict.contains(ttcore::ArgumentTypeAttr::name)) {
      Attribute attr = argAttrDict.get(ttcore::ArgumentTypeAttr::name);
      auto argTypeAttr = mlir::cast<ttcore::ArgumentTypeAttr>(attr);
      ttcore::ArgumentType argType = argTypeAttr.getValue();
      if (argType == ttcore::ArgumentType::Constant ||
          argType == ttcore::ArgumentType::Parameter) {
        return true;
      }
    }
    return false;
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
    traceFuncOp->setAttr(g_TTNNTraceAttrName, builder.getUnitAttr());
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

    // Input types will match the trace function
    llvm::SmallVector<mlir::Type> inputTypes;
    for (mlir::Value input : traceFunc.getArguments()) {
      inputTypes.push_back(input.getType());
    }

    // output types are: traceId, actual outputs, trace input slots, trace
    // output slots
    llvm::SmallVector<mlir::Type> outputTypes;

    outputTypes.push_back(utils::getTraceIdType(context));

    for (mlir::Type outputType : traceFunc.getFunctionType().getResults()) {
      outputTypes.push_back(outputType);
    }

    for (mlir::Value input : traceFunc.getArguments()) {
      outputTypes.push_back(input.getType());
    }

    for (mlir::Type outputType : traceFunc.getFunctionType().getResults()) {
      outputTypes.push_back(outputType);
    }

    // Create and insert function
    auto runAndCaptureTraceFuncType =
        builder.getFunctionType(inputTypes, outputTypes);

    TraceSmallString runAndCaptureTraceFuncName =
        getCaptureTraceFuncName(funcOp, traceFuncIndex);

    builder.setInsertionPoint(funcOp);
    auto runAndCaptureTraceFunc = builder.create<func::FuncOp>(
        funcOp.getLoc(), runAndCaptureTraceFuncName,
        runAndCaptureTraceFuncType);
    runAndCaptureTraceFunc->setAttr(g_TTNNTraceAttrName, builder.getUnitAttr());
    if (traceFunc.getAllArgAttrs()) {
      runAndCaptureTraceFunc.setAllArgAttrs(traceFunc.getAllArgAttrs());
    }
    runAndCaptureTraceFunc.setPrivate();

    // Build the body of the function
    auto *runAndCaptureTraceFuncEntryBlock =
        runAndCaptureTraceFunc.addEntryBlock();
    builder.setInsertionPointToStart(runAndCaptureTraceFuncEntryBlock);

    auto deviceOp =
        utils::getOrInsertDevice(rewriter, runAndCaptureTraceFuncEntryBlock);
    auto device = ttcore::lookupDevice(deviceOp);

    // allocate input slots
    llvm::SmallVector<mlir::Value> inputSlots;
    for (size_t i = 0; i < runAndCaptureTraceFunc.getNumArguments(); i++) {
      mlir::Type inputType = inputTypes[i];
      if (!mlir::isa<RankedTensorType>(inputType)) {
        return runAndCaptureTraceFunc.emitError(
            "Input type must be a ranked tensor type");
      }

      // Don't create empty slots for constants/parameters
      if (isConstantOrParameter(runAndCaptureTraceFunc, i)) {
        inputSlots.push_back(runAndCaptureTraceFunc.getArgument(i));
        continue;
      }

      RankedTensorType inputTensorType =
          mlir::cast<RankedTensorType>(inputType);
      ttnn::TTNNLayoutAttr ttnnLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(inputTensorType.getEncoding());
      ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
          context, ttnnLayoutAttr.getMemLayout(),
          ttnn::BufferTypeAttr::get(context, ttnnLayoutAttr.getBufferType()),
          utils::createShardSpecIfNeeded(ttnnLayoutAttr,
                                         device.getWorkerGrid()));

      auto emptyOp = builder.create<ttnn::EmptyOp>(
          runAndCaptureTraceFunc.getLoc(), inputType, deviceOp,
          ttnn::ShapeAttr::get(context, inputTensorType.getShape()),
          ttcore::DataTypeAttr::get(context, ttnnLayoutAttr.getDataType()),
          ttnn::LayoutAttr::get(context, ttnnLayoutAttr.getLayout()),
          memoryConfigAttr);

      inputSlots.push_back(emptyOp.getResult());
    }

    // move inputs to host and copy into input slots
    for (size_t i = 0; i < inputSlots.size(); i++) {
      // Skip inputs that are constants/parameters
      if (isConstantOrParameter(runAndCaptureTraceFunc, i)) {
        continue;
      }
      mlir::Value input = runAndCaptureTraceFunc.getArgument(i);
      RankedTensorType currentInputType =
          mlir::cast<RankedTensorType>(input.getType());
      RankedTensorType newResultType = utils::RankedTensorTypeFactory::create(
          currentInputType, ttnn::BufferType::SystemMemory);

      auto fromDeviceOp = builder.create<ttnn::FromDeviceOp>(
          runAndCaptureTraceFunc.getLoc(), newResultType, input);

      builder.create<ttnn::WriteTensorOp>(runAndCaptureTraceFunc.getLoc(),
                                          fromDeviceOp, inputSlots[i],
                                          /*blocking=*/false, /*cq_id=*/0);
    }

    // call the trace function on the input slots
    auto traceFuncCall = builder.create<func::CallOp>(
        runAndCaptureTraceFunc.getLoc(), traceFunc, inputSlots);

    // now, we can capture the trace
    auto beginTraceCaptureOp = builder.create<ttnn::BeginTraceCaptureOp>(
        runAndCaptureTraceFunc.getLoc(), utils::getTraceIdType(context),
        deviceOp,
        /*cq_id=*/0);

    auto captureTraceCall = builder.create<func::CallOp>(
        runAndCaptureTraceFunc.getLoc(), traceFunc, inputSlots);

    builder.create<ttnn::EndTraceCaptureOp>(runAndCaptureTraceFunc.getLoc(),
                                            deviceOp, beginTraceCaptureOp,
                                            /*cq_id=*/0);

    // create the return op
    llvm::SmallVector<mlir::Value> returnValues;

    returnValues.push_back(beginTraceCaptureOp.getTraceId());
    for (mlir::Value output : traceFuncCall.getResults()) {
      returnValues.push_back(output);
    }
    for (mlir::Value inputSlot : inputSlots) {
      returnValues.push_back(inputSlot);
    }
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
    executeTraceFunc->setAttr(g_TTNNTraceAttrName, builder.getUnitAttr());
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

    auto traceOp = builder.create<ttnn::CaptureOrExecuteTraceOp>(
        funcOp.getLoc(), outputTypes, device, captureTraceSymbolAttr,
        executeTraceSymbolAttr, inputs);

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
    // skip const-eval functions
    if (ttmlir::utils::isConstEvalFunc(funcOp)) {
      return mlir::success();
    }

    // skip trace functions
    if (utils::isTTNNTraceFunc(funcOp)) {
      return mlir::success();
    }

    if (funcOp.getBlocks().size() != 1) {
      return funcOp.emitError("FuncOp should have exactly one block");
    }

    llvm::SmallVector<Operation *> opsToHoist;

    bool seenHoistableOp = false;
    mlir::Block &block = funcOp.getBlocks().front();
    for (mlir::Operation &op : block.getOperations()) {
      if (shouldHoistOp(&op)) {
        // Hoist all ops starting from this op into a new func
        seenHoistableOp = true;
        opsToHoist.push_back(&op);
        continue;
      }
      // If a non-hoistable op is found after a hoistable op, it must be a
      // return op
      if (seenHoistableOp && !::mlir::isa<func::ReturnOp>(op)) {
        return op.emitError(
            "Non-hoistable op found after seeing a hoistable op");
      }
    }

    if (opsToHoist.empty()) {
      return mlir::success();
    }

    return performHoistTransform(funcOp, opsToHoist);
  }
};
} // namespace mlir::tt::ttnn
