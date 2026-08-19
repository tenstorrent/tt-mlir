// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
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
#include "llvm/ADT/Sequence.h"
#include <algorithm>
#include <atomic>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNTRACEHOISTTRANSFORM
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

using TraceSmallString = llvm::SmallString<64>;

namespace {
// GlobalSemaphore values are pass-through handles into the trace wrapper:
// they are part of the trace function signature so device kernels can use
// them, but they have no host-side representation, no persistent device
// slot, and no host-to-device transfer. Their position in the wrapper
// signature is fixed at the end (after all tensor inputs) so the runtime
// program signature matches.
inline bool isSemaphoreType(mlir::Type type) {
  return ::mlir::isa<ttnn::GlobalSemaphoreType>(type);
}
inline bool isSemaphoreValue(mlir::Value value) {
  return isSemaphoreType(value.getType());
}
} // namespace

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
  // Ops that read tensor data back to host cannot live inside a trace: a
  // trace is captured once and replayed, so every replay would silently reuse
  // the capturing step's value. ttnn.adamw reads lr and the bias-correction
  // operands back, since ttml takes them as floats.
  static bool performsHostReadback(Operation *op) {
    return ::mlir::isa<mlir::tt::ttnn::AdamWOp>(op);
  }

  bool shouldHoistOp(Operation *op) {
    bool shouldHoist = true;
    shouldHoist &= !::mlir::isa<func::ReturnOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttcore::LoadCachedOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttnn::CaptureOrExecuteTraceOp>(op);
    shouldHoist &= !::mlir::isa<mlir::tt::ttnn::GetDeviceOp>(op);
    shouldHoist &=
        !(op->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>());
    shouldHoist &= !performsHostReadback(op);
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

  TraceSmallString getAllocateSlotsTraceFuncName(func::FuncOp funcOp,
                                                 uint64_t traceFuncIndex) {
    TraceSmallString allocateSlotsTraceFuncName;
    allocateSlotsTraceFuncName.append(g_TTNNAllocateSlotsTracePrefix);
    allocateSlotsTraceFuncName.append(getTraceFuncName(funcOp, traceFuncIndex));
    return allocateSlotsTraceFuncName;
  }

  TraceSmallString getCaptureTraceFuncName(func::FuncOp funcOp,
                                           uint64_t traceFuncIndex) {
    TraceSmallString captureTraceFuncName;
    captureTraceFuncName.append(g_TTNNCaptureTracePrefix);
    captureTraceFuncName.append(getTraceFuncName(funcOp, traceFuncIndex));
    return captureTraceFuncName;
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

  // Check if a value should remain on device during trace capture.
  // Handles three cases:
  // 1. Direct BlockArgument → checks shouldKeepArgOnDevice
  // 2. LoadCachedOp result → always device-resident (consteval)
  // 3. ttnn.empty / ttnn.alloc result → prelude-allocated device scratch
  //    buffer (e.g. stats scratch for distributed_rms_norm) that must be
  //    passed through, not host-staged.
  bool isDeviceResidentValue(mlir::Value value) {
    if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
      auto funcOp =
          mlir::dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp());
      if (!funcOp) {
        return false;
      }
      return shouldKeepArgOnDevice(funcOp, blockArg.getArgNumber());
    }

    auto *defOp = value.getDefiningOp();
    if (!defOp) {
      return false;
    }

    if (mlir::isa<mlir::tt::ttcore::LoadCachedOp>(defOp)) {
      return true;
    }

    if (mlir::isa<ttnn::EmptyOp, ttnn::AllocOp>(defOp)) {
      return true;
    }

    return false;
  }

  // Collect all inputs and outputs outside the operation set to hoist.
  // Tensor inputs come first, followed by semaphore inputs.
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
        if (!::mlir::isa<RankedTensorType>(operand.getType()) &&
            !isSemaphoreValue(operand)) {
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
      // Tensors before semaphores; capture/execute op operand groups and the
      // runtime program signature follow this layout.
      bool aIsSemaphore = isSemaphoreValue(a);
      bool bIsSemaphore = isSemaphoreValue(b);
      if (aIsSemaphore != bIsSemaphore) {
        return !aIsSemaphore;
      }
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
        // LoadCachedOp results are consteval; ttnn.empty / ttnn.alloc are
        // prelude device-scratch buffers. Mark both as Constant so the trace
        // wrapper passes them through device-resident.
        if (mlir::isa<mlir::tt::ttcore::LoadCachedOp>(defOp) ||
            mlir::isa<ttnn::EmptyOp, ttnn::AllocOp>(defOp)) {
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

  // Describes the trace-main function: the hoisted computation's boundary
  // values, the device-resident/host-staged partition of its tensor inputs,
  // and the persistent slot types derived from them. Computed once when the
  // trace-main function is created; the slot allocation, capture, and execute
  // functions - and the trace op itself - all derive their signatures from
  // it, which is what keeps them mutually consistent. All indices refer to
  // trace-main's [tensor inputs][output slots][semaphores] argument order,
  // with device-resident tensors grouped before host-staged ones.
  struct TraceMainFuncInfo {
    // The trace-main function itself.
    func::FuncOp traceFunc;
    // Values of the original function feeding the trace, device-resident
    // tensors first, then host-staged tensors, then semaphores.
    llvm::SmallVector<mlir::Value> boundaryInputs;
    // Values of the original function that the trace produces.
    llvm::SmallVector<mlir::Value> boundaryOutputs;
    // Number of leading tensor input arguments.
    size_t numTensorInputs = 0;
    // Split point of the tensor input indices [0, numTensorInputs):
    // [0, numDeviceResident) is device-resident, the rest is host-staged. A
    // device-resident input (constant/parameter/KV cache, or a
    // prelude-allocated scratch buffer) needs no host staging and acts as its
    // own persistent input slot; everything else is staged from host into a
    // freshly allocated slot. This partition is the compile-time twin of the
    // runtime's storage-type test on the trace op's inputs:
    // insertCaptureOrExecuteTraceOp forwards device-resident inputs on device
    // and converts the rest to system memory, and the runtime uses the
    // resulting storage type to decide which inputs to pass to which program.
    size_t numDeviceResident = 0;
    // Persistent slot type per tensor input argument: a device-resident
    // argument is its own slot and keeps its type; a host-staged argument
    // gets a DRAM slot to be written into.
    llvm::SmallVector<mlir::Type> inputSlotTypes;
    // Original arg attrs per tensor input argument, as set on trace-main's
    // arguments. Never null: missing attrs are normalized to empty dicts.
    // Slot arguments derived from these follow one rule: a device-resident
    // slot is the original argument and keeps its attrs; a host-staged slot
    // is a fresh buffer and gets none.
    llvm::SmallVector<mlir::DictionaryAttr> inputArgAttrs;
    // Types of the output slot arguments (also the trace op's result types).
    llvm::SmallVector<mlir::Type> outputSlotTypes;
    // Types of the trailing semaphore arguments.
    llvm::SmallVector<mlir::Type> semaphoreTypes;

    // Contiguous views over the two halves of the tensor-input partition.
    auto deviceResidentIndices() const {
      return llvm::seq<size_t>(0, numDeviceResident);
    }
    auto hostStagedIndices() const {
      return llvm::seq(numDeviceResident, numTensorInputs);
    }
    llvm::ArrayRef<mlir::Type> deviceResidentSlotTypes() const {
      return llvm::ArrayRef<mlir::Type>(inputSlotTypes)
          .take_front(numDeviceResident);
    }
    llvm::ArrayRef<mlir::Type> hostStagedSlotTypes() const {
      return llvm::ArrayRef<mlir::Type>(inputSlotTypes)
          .drop_front(numDeviceResident);
    }
    llvm::ArrayRef<mlir::DictionaryAttr> deviceResidentSlotArgAttrs() const {
      return llvm::ArrayRef<mlir::DictionaryAttr>(inputArgAttrs)
          .take_front(numDeviceResident);
    }
  };

  // Creates the trace-main function and returns the TraceMainFuncInfo
  // describing it, from which the slot allocation and capture functions are
  // derived.
  //
  // Results are written into persistent output slots passed in as trailing
  // arguments rather than returned. Argument order is
  // [tensor inputs][output slots][semaphores].
  ::mlir::FailureOr<TraceMainFuncInfo>
  createTraceFunction(func::FuncOp funcOp,
                      llvm::ArrayRef<Operation *> opsToHoist,
                      uint64_t traceFuncIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<mlir::Value> outputs;
    llvm::SmallVector<mlir::Type> outputTypes;

    collectFunctionBoundary(opsToHoist, inputs, outputs);

    // collectFunctionBoundary sorts tensors before semaphores; split there so
    // the output slots can be inserted between the two groups.
    size_t numTensorInputs = 0;
    while (numTensorInputs < inputs.size() &&
           !isSemaphoreValue(inputs[numTensorInputs])) {
      numTensorInputs++;
    }

    // Group device-resident tensor inputs ahead of host-staged ones, so each
    // partition is a contiguous range and the downstream builders can slice
    // rather than gather.
    size_t numDeviceResident =
        std::stable_partition(
            inputs.begin(), inputs.begin() + numTensorInputs,
            [&](mlir::Value input) { return isDeviceResidentValue(input); }) -
        inputs.begin();

    for (mlir::Value output : outputs) {
      outputTypes.push_back(output.getType());
    }

    llvm::SmallVector<mlir::DictionaryAttr> inputAttrs =
        getInputAttrs(context, inputs);
    // Types and attributes for all arguments of the new trace main function,
    // in signature order: [input tensors][output slots][semaphores].
    llvm::SmallVector<mlir::Type> argTypes;
    llvm::SmallVector<mlir::DictionaryAttr> newArgAttrs;

    for (size_t i = 0; i < numTensorInputs; i++) {
      argTypes.push_back(inputs[i].getType());
      newArgAttrs.push_back(inputAttrs[i]);
    }
    for (mlir::Type outputType : outputTypes) {
      argTypes.push_back(outputType);
      newArgAttrs.push_back(mlir::DictionaryAttr::get(context));
    }
    for (size_t i = numTensorInputs; i < inputs.size(); i++) {
      argTypes.push_back(inputs[i].getType());
      newArgAttrs.push_back(inputAttrs[i]);
    }

    TraceSmallString traceFuncName = getTraceFuncName(funcOp, traceFuncIndex);

    auto traceFuncType = builder.getFunctionType(argTypes, /*results=*/{});

    // Create the function
    builder.setInsertionPoint(funcOp);
    auto traceFuncOp = builder.create<func::FuncOp>(
        funcOp.getLoc(), traceFuncName, traceFuncType);
    ttmlir::utils::setFunctionType(traceFuncOp,
                                   ttmlir::utils::FunctionType::TraceMain);

    traceFuncOp.setAllArgAttrs(newArgAttrs);
    traceFuncOp.setPrivate();

    // Build the body of the new function
    auto *traceFuncEntryBlock = traceFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(traceFuncEntryBlock);

    // maps original input values to trace function input
    // arguments/intermediates
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (size_t i = 0; i < numTensorInputs; i++) {
      valueMap.insert({inputs[i], traceFuncOp.getArgument(i)});
    }
    for (size_t i = numTensorInputs; i < inputs.size(); i++) {
      valueMap.insert(
          {inputs[i], traceFuncOp.getArgument(outputTypes.size() + i)});
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

    // Store each result into its persistent output slot. These stores are part
    // of the traced computation, so replaying the trace lands results in the
    // slots, and the capture function's uncaptured call compiles them.
    for (auto [i, output] : llvm::enumerate(outputs)) {
      auto it = valueMap.find(output);
      if (it == valueMap.end()) {
        return funcOp.emitError(
            "Could not map output value in hoisted function");
      }
      builder.create<ttnn::CopyOp>(
          funcOp.getLoc(), it->second,
          traceFuncOp.getArgument(numTensorInputs + i));
    }
    builder.create<func::ReturnOp>(funcOp.getLoc());

    TraceMainFuncInfo info;
    info.traceFunc = traceFuncOp;
    info.numTensorInputs = numTensorInputs;
    info.numDeviceResident = numDeviceResident;
    for (size_t i = 0; i < numTensorInputs; i++) {
      auto tensorType = mlir::cast<RankedTensorType>(inputs[i].getType());
      info.inputArgAttrs.push_back(
          inputAttrs[i] ? inputAttrs[i] : mlir::DictionaryAttr::get(context));
      if (i < numDeviceResident) {
        info.inputSlotTypes.push_back(tensorType);
      } else {
        info.inputSlotTypes.push_back(utils::RankedTensorTypeFactory::create(
            tensorType, BufferType::DRAM));
      }
    }
    for (size_t i = numTensorInputs; i < inputs.size(); i++) {
      info.semaphoreTypes.push_back(inputs[i].getType());
    }
    info.outputSlotTypes = std::move(outputTypes);
    info.boundaryInputs = std::move(inputs);
    info.boundaryOutputs = std::move(outputs);
    return info;
  }

  // Creates the slot allocation function.
  //
  // This function allocates the persistent device buffers that every capture of
  // this trace reads from and writes to, and it is the ONLY place they are
  // allocated. It runs exactly once per trace, on the first capture. Keeping
  // allocation out of the capture function is what lets a stale trace be
  // recaptured without allocating anything that outlives its own capture
  // window, so a recapture cannot perturb the device allocator state that other
  // cached traces have baked into their command streams.
  //
  // Signature: (device-resident tensor args) -> (input slots..., output
  // slots...) Device-resident arguments are passed through as their own slots;
  // host-staged inputs and all outputs get freshly allocated DRAM slots.
  mlir::LogicalResult
  createAllocateSlotsTraceFunction(func::FuncOp funcOp, uint64_t traceFuncIndex,
                                   const TraceMainFuncInfo &info) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    // Results: every input slot, then every output slot.
    llvm::SmallVector<mlir::Type> outputTypes(info.inputSlotTypes);
    llvm::append_range(outputTypes, info.outputSlotTypes);

    TraceSmallString allocateSlotsFuncName =
        getAllocateSlotsTraceFuncName(funcOp, traceFuncIndex);

    builder.setInsertionPoint(funcOp);
    // Inputs: only the device-resident arguments, which pass through as slots.
    auto allocateSlotsFunc = builder.create<func::FuncOp>(
        funcOp.getLoc(), allocateSlotsFuncName,
        builder.getFunctionType(info.deviceResidentSlotTypes(), outputTypes));
    ttmlir::utils::setFunctionType(
        allocateSlotsFunc, ttmlir::utils::FunctionType::TraceAllocateSlots);
    allocateSlotsFunc.setAllArgAttrs(info.deviceResidentSlotArgAttrs());
    allocateSlotsFunc.setPrivate();

    auto *entryBlock = allocateSlotsFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto deviceOp = utils::getOrInsertDevice(rewriter, entryBlock);

    llvm::SmallVector<mlir::Value> slots(
        allocateSlotsFunc.getArguments().begin(),
        allocateSlotsFunc.getArguments().end());
    for (mlir::Type slotType : info.hostStagedSlotTypes()) {
      slots.push_back(createSlot(builder, allocateSlotsFunc.getLoc(), deviceOp,
                                 mlir::cast<RankedTensorType>(slotType)));
    }
    for (mlir::Type outputSlotType : info.outputSlotTypes) {
      slots.push_back(createSlot(builder, allocateSlotsFunc.getLoc(), deviceOp,
                                 mlir::cast<RankedTensorType>(outputSlotType)));
    }

    builder.create<func::ReturnOp>(allocateSlotsFunc.getLoc(), slots);

    return mlir::success();
  }

  // Allocates one persistent device buffer of the given type.
  mlir::Value createSlot(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value deviceOp, RankedTensorType slotType) {
    mlir::MLIRContext *context = &this->getContext();
    return builder
        .create<ttnn::EmptyOp>(
            loc, slotType, deviceOp,
            ttnn::ShapeAttr::get(context, slotType.getShape()))
        .getResult();
  }

  // Creates the capture function.
  //
  // Captures the trace against slots that the allocation function already
  // created and that are passed in as arguments. This is the single capture
  // path: the runtime invokes it for the initial capture and again, with the
  // very same slots, to recapture a stale trace. Because it allocates no
  // persistent buffer of its own, a recapture leaves the allocator untouched.
  //
  // Signature:
  //   (host-staged inputs..., input slots..., output slots..., semaphores...)
  //     -> trace id
  mlir::LogicalResult
  createRunAndCaptureTraceFunction(func::FuncOp funcOp, uint64_t traceFuncIndex,
                                   const TraceMainFuncInfo &info) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

    func::FuncOp traceFunc = info.traceFunc;
    llvm::SmallVector<mlir::Type> argTypes;
    llvm::SmallVector<mlir::DictionaryAttr> argAttrs;

    // Host-staged inputs, in system memory, to be written into their slots.
    for (size_t argIndex : info.hostStagedIndices()) {
      auto originalType =
          mlir::cast<RankedTensorType>(info.boundaryInputs[argIndex].getType());
      argTypes.push_back(utils::RankedTensorTypeFactory::create(
          originalType, BufferType::SystemMemory));
      argAttrs.push_back(info.inputArgAttrs[argIndex]);
    }
    // Input slots, then output slots, then semaphores. A device-resident slot
    // is the original argument and keeps its attrs; a host-staged slot is a
    // fresh buffer and gets none.
    for (size_t argIndex : info.deviceResidentIndices()) {
      argTypes.push_back(info.inputSlotTypes[argIndex]);
      argAttrs.push_back(info.inputArgAttrs[argIndex]);
    }
    for (size_t argIndex : info.hostStagedIndices()) {
      argTypes.push_back(info.inputSlotTypes[argIndex]);
      argAttrs.push_back(mlir::DictionaryAttr::get(context));
    }
    for (mlir::Type outputSlotType : info.outputSlotTypes) {
      argTypes.push_back(outputSlotType);
      argAttrs.push_back(mlir::DictionaryAttr::get(context));
    }
    for (mlir::Type semaphoreType : info.semaphoreTypes) {
      argTypes.push_back(semaphoreType);
      argAttrs.push_back(mlir::DictionaryAttr::get(context));
    }

    TraceSmallString captureFuncName =
        getCaptureTraceFuncName(funcOp, traceFuncIndex);

    builder.setInsertionPoint(funcOp);
    auto captureFunc = builder.create<func::FuncOp>(
        funcOp.getLoc(), captureFuncName,
        builder.getFunctionType(argTypes, {utils::getTraceIdType(context)}));
    ttmlir::utils::setFunctionType(
        captureFunc, ttmlir::utils::FunctionType::TraceRunAndCapture);
    captureFunc.setAllArgAttrs(argAttrs);
    captureFunc.setPrivate();

    auto *entryBlock = captureFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto deviceOp = utils::getOrInsertDevice(rewriter, entryBlock);

    const size_t inputSlotBase = info.hostStagedIndices().size();
    const size_t outputSlotBase = inputSlotBase + info.numTensorInputs;
    const size_t semaphoreBase = outputSlotBase + info.outputSlotTypes.size();

    // Each trace function tensor argument's input slot, in argument order.
    llvm::SmallVector<mlir::Value> inputSlots;
    for (size_t i = 0; i < info.numTensorInputs; i++) {
      inputSlots.push_back(captureFunc.getArgument(inputSlotBase + i));
    }

    llvm::SmallVector<mlir::Value> traceOutputSlots;
    for (size_t i = 0; i < info.outputSlotTypes.size(); i++) {
      traceOutputSlots.push_back(captureFunc.getArgument(outputSlotBase + i));
    }

    // Transfer the host-staged inputs into their device slots. This has to
    // happen before the capture window opens: tt-metal rejects host writes
    // while a capture is active.
    for (auto [h, argIndex] : llvm::enumerate(info.hostStagedIndices())) {
      builder.create<ttnn::WriteTensorOp>(
          captureFunc.getLoc(), captureFunc.getArgument(h),
          inputSlots[argIndex], /*blocking=*/false, /*cq_id=*/0);
    }

    // Rebuild the trace function's call arguments in its own argument order:
    // [tensor inputs from their slots][output slots][semaphores].
    llvm::SmallVector<mlir::Value> traceCallArgs(inputSlots);
    llvm::append_range(traceCallArgs, traceOutputSlots);
    for (size_t i = 0; i < info.semaphoreTypes.size(); i++) {
      traceCallArgs.push_back(captureFunc.getArgument(semaphoreBase + i));
    }

    // Execute the trace function once without capture to compile programs and
    // populate program cache.
    builder.create<func::CallOp>(captureFunc.getLoc(), traceFunc,
                                 traceCallArgs);

    auto beginTraceCaptureOp = builder.create<ttnn::BeginTraceCaptureOp>(
        captureFunc.getLoc(), utils::getTraceIdType(context), deviceOp,
        /*cq_id=*/0);

    // Execute the trace function again and capture it.
    builder.create<func::CallOp>(captureFunc.getLoc(), traceFunc,
                                 traceCallArgs);

    builder.create<ttnn::EndTraceCaptureOp>(captureFunc.getLoc(), deviceOp,
                                            beginTraceCaptureOp,
                                            /*cq_id=*/0);

    // Replay once so the output slots hold valid results for this invocation.
    builder.create<ttnn::ExecuteTraceOp>(captureFunc.getLoc(), deviceOp,
                                         beginTraceCaptureOp,
                                         /*cq_id=*/0, /*blocking=*/false);

    builder.create<func::ReturnOp>(
        captureFunc.getLoc(),
        mlir::ValueRange{beginTraceCaptureOp.getTraceId()});

    return mlir::success();
  }

  mlir::LogicalResult createExecuteTraceFunction(func::FuncOp funcOp,
                                                 uint64_t traceFuncIndex) {
    mlir::MLIRContext *context = &this->getContext();
    mlir::OpBuilder builder(context);
    mlir::IRRewriter rewriter(builder);

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
  // This optimization identifies such patterns where a function argument is
  // immediately converted via ToLayoutOp.
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
    // be optimized. We look for arguments that are immediately used by a
    // ToLayoutOp.
    for (size_t argIdx = 0; argIdx < funcOp.getNumArguments(); argIdx++) {
      BlockArgument arg = funcOp.getArgument(argIdx);
      RankedTensorType currentTensorType =
          mlir::cast<RankedTensorType>(arg.getType());

      ttnn::ToTensorSpecOp tensorSpecOp = nullptr;

      // Check if argument has only one use and it's a ToTensorSpecOp.
      if (arg.hasOneUse()) {
        auto *user = *arg.getUsers().begin();
        if (auto directTensorSpecOp =
                mlir::dyn_cast<ttnn::ToTensorSpecOp>(user)) {
          tensorSpecOp = directTensorSpecOp;
        }
      }

      // If there's no ToTensorSpecOp pattern, keep the original type.
      if (!tensorSpecOp) {
        newInputTypes.push_back(currentTensorType);
        continue;
      }

      // Get the target type from the ToTensorSpecOp and update the function
      // argument type directly.
      RankedTensorType targetTensorType = tensorSpecOp.getResult().getType();
      TTNNLayoutAttr targetLayoutAttr =
          utils::getLayoutAttrFromTensor(targetTensorType);
      TTNNLayoutAttr currentLayoutAttr =
          utils::getLayoutAttrFromTensor(currentTensorType);
      if (targetLayoutAttr.getDataType() != currentLayoutAttr.getDataType()) {
        return funcOp.emitError(
                   "ToTensorSpecOp changed data type for argument ")
               << argIdx << ", expected only buffer type change";
      }
      newInputTypes.push_back(targetTensorType);

      // Replace all uses of ToTensorSpecOp with the function argument.
      tensorSpecOp.getResult().replaceAllUsesWith(arg);
      opsToErase.push_back(tensorSpecOp);

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

  mlir::LogicalResult insertCaptureOrExecuteTraceOp(
      func::FuncOp funcOp, llvm::ArrayRef<Operation *> opsToHoist,
      uint64_t traceFuncIndex, const TraceMainFuncInfo &info) {
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

    TraceSmallString allocateSlotsTraceFuncName =
        getAllocateSlotsTraceFuncName(funcOp, traceFuncIndex);
    func::FuncOp allocateSlotsTraceFunc =
        moduleOp.lookupSymbol<func::FuncOp>(allocateSlotsTraceFuncName);
    if (!allocateSlotsTraceFunc) {
      return funcOp.emitError(
          "Could not find slot allocation trace function with name: " +
          allocateSlotsTraceFuncName);
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

    auto captureTraceSymbolAttr =
        mlir::SymbolRefAttr::get(context, captureTraceFuncName);
    auto allocateSlotsTraceSymbolAttr =
        mlir::SymbolRefAttr::get(context, allocateSlotsTraceFuncName);
    auto executeTraceSymbolAttr =
        mlir::SymbolRefAttr::get(context, executeTraceFuncName);

    Operation *firstOp = opsToHoist.front();

    builder.setInsertionPoint(firstOp);

    auto device = utils::getOrInsertDevice(rewriter, firstOp);

    // Split inputs into the two operand groups expected by the op; the
    // boundary inputs hold the tensors first and the semaphores after them.
    llvm::SmallVector<mlir::Value> tensorInputs(info.numTensorInputs);
    llvm::SmallVector<mlir::Value> semaphoreInputs(info.boundaryInputs.begin() +
                                                       info.numTensorInputs,
                                                   info.boundaryInputs.end());

    // Device-resident values (constants, parameters, KV cache) can be
    // captured directly without moving to system memory.
    for (size_t argIndex : info.deviceResidentIndices()) {
      mlir::Value input = info.boundaryInputs[argIndex];
      auto tensorType = mlir::cast<RankedTensorType>(input.getType());
      auto layout = mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
      if (layout.getBufferType() == ttnn::BufferType::SystemMemory) {
        return funcOp.emitError(
            "Device-resident input must be on device, but found on "
            "system memory");
      }
      tensorInputs[argIndex] = input;
    }
    // Host-staged inputs are converted to system memory if needed.
    for (size_t argIndex : info.hostStagedIndices()) {
      mlir::Value input = info.boundaryInputs[argIndex];
      auto tensorType = mlir::cast<RankedTensorType>(input.getType());
      auto layout = mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
      if (layout.getBufferType() != ttnn::BufferType::SystemMemory) {
        // Convert to system memory using ToTensorSpecOp
        RankedTensorType systemMemoryTileType =
            utils::RankedTensorTypeFactory::create(
                tensorType, ttnn::BufferType::SystemMemory);

        auto toTensorSpecOp = builder.create<ttnn::ToTensorSpecOp>(
            funcOp.getLoc(), systemMemoryTileType, input);
        tensorInputs[argIndex] = toTensorSpecOp.getResult();
      } else {
        // Already on system memory
        tensorInputs[argIndex] = input;
      }
    }

    auto traceOp = builder.create<ttnn::CaptureOrExecuteTraceOp>(
        funcOp.getLoc(), info.outputSlotTypes, device,
        allocateSlotsTraceSymbolAttr, captureTraceSymbolAttr,
        executeTraceSymbolAttr, tensorInputs, semaphoreInputs);

    // Replace uses of original outputs with the output of the trace op function
    for (size_t i = 0; i < info.boundaryOutputs.size(); i++) {
      mlir::Value output = info.boundaryOutputs[i];
      output.replaceAllUsesWith(traceOp->getResult(i));
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
    // Create the trace function. Everything the downstream builders need to
    // know about it is computed here, once.
    mlir::FailureOr<TraceMainFuncInfo> info =
        createTraceFunction(funcOp, opsToHoist, traceFuncIndex);
    if (failed(info)) {
      return mlir::failure();
    }

    mlir::LogicalResult result =
        createAllocateSlotsTraceFunction(funcOp, traceFuncIndex, *info);
    if (failed(result)) {
      return result;
    }

    result = createRunAndCaptureTraceFunction(funcOp, traceFuncIndex, *info);
    if (failed(result)) {
      return result;
    }

    result = createExecuteTraceFunction(funcOp, traceFuncIndex);
    if (failed(result)) {
      return result;
    }

    result = insertCaptureOrExecuteTraceOp(funcOp, opsToHoist, traceFuncIndex,
                                           *info);
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
      // A host-readback op cannot be traced, and unlike the creation ops
      // handled below it cannot be sunk above the trace either (it writes the
      // parameters in place), so the trace region ends before the first one.
      // A training step is compute followed by the optimizer, so the
      // expensive part still gets traced.
      size_t traceEnd = allOps.size();
      for (size_t i = firstHoistable; i < allOps.size(); i++) {
        if (performsHostReadback(allOps[i])) {
          traceEnd = i;
          break;
        }
      }

      // Find the last hoistable op (before any trailing non-hoistable ops)
      size_t lastHoistable = firstHoistable;
      for (size_t i = traceEnd - 1; i > firstHoistable; i--) {
        if (shouldHoistOp(allOps[i])) {
          lastHoistable = i;
          break;
        }
      }

      // Collect all hoistable ops between first and last. Creation ops (e.g.
      // ttnn.constant / ttnn.full materialized mid-graph by op decompositions
      // such as SDPA) are not hoistable, but they only depend on the device
      // (no data dependency on surrounding compute). Rather than failing, sink
      // them above the trace region so they are treated as regular trace inputs
      // (recreated per execution), matching how leading creation ops are
      // handled when const-eval is disabled.
      llvm::SmallVector<Operation *> creationOpsToSink;
      for (size_t i = firstHoistable; i <= lastHoistable; i++) {
        if (shouldHoistOp(allOps[i])) {
          opsToHoist.push_back(allOps[i]);
        } else if (allOps[i]
                       ->hasTrait<
                           mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>()) {
          creationOpsToSink.push_back(allOps[i]);
        } else {
          // We found a non-hoistable op in the middle - this is an error
          return allOps[i]->emitError(
              "Non-hoistable op found in the middle of hoistable ops");
        }
      }

      // Move the mid-graph creation ops above the first hoistable op so the
      // trace op (inserted at the first hoistable op) dominates its operands.
      for (Operation *creationOp : creationOpsToSink) {
        creationOp->moveBefore(allOps[firstHoistable]);
      }
    }

    if (opsToHoist.empty()) {
      return mlir::success();
    }

    // Perform the hoist transform
    if (failed(performHoistTransform(funcOp, opsToHoist))) {
      return mlir::failure();
    }

    if (failed(mergeToLayoutOpsWithFuncArgs(funcOp))) {
      return mlir::failure();
    }

    return mlir::success();
  }
};
} // namespace mlir::tt::ttnn
