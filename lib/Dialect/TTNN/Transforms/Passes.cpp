// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCREATEINPUTGENERATORS
#define GEN_PASS_DEF_TTNNCREATEMAINFORTEST
#define GEN_PASS_DEF_TTNNLOADINPUTTENSORS
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNTUPLIFYTENSORS
#define GEN_PASS_DEF_TTNNEMPYWORKAROUNDS
#define GEN_PASS_DEF_TTNNPREPAREMODULEFOREXPORT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDeallocate : public impl::TTNNDeallocateBase<TTNNDeallocate> {

public:
  using impl::TTNNDeallocateBase<TTNNDeallocate>::TTNNDeallocateBase;

  Operation *getLastValueUsageOp(const LivenessBlockInfo *livenessInfo,
                                 Value value) {
    Operation *startOp = livenessInfo->getStartOperation(value);
    Operation *endOp = livenessInfo->getEndOperation(value, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(value);
        });

    // In case of DPS op keep going until we find the last usage of the tensor.
    //
    while (
        opOperandIter != endOp->getOpOperands().end() &&
        isa<DestinationStyleOpInterface>(endOp) &&
        cast<DestinationStyleOpInterface>(endOp).isDpsInit(&(*opOperandIter))) {
      OpResult result =
          cast<DestinationStyleOpInterface>(endOp).getTiedOpResult(
              &(*opOperandIter));
      endOp = livenessInfo->getEndOperation(result, endOp);
      opOperandIter =
          llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
            return opOperand.is(result);
          });
    }

    return endOp;
  }

  // Check if a value has a conv2d/conv_transpose2d user with
  // deallocate_activation=true that is not the last user, which would cause a
  // use-after-free.
  LogicalResult checkConv2dUseAfterFree(Value value, Operation *lastOp) {
    for (Operation *user : value.getUsers()) {
      if (user == lastOp) {
        continue;
      }

      auto result =
          llvm::TypeSwitch<Operation *, LogicalResult>(user)
              .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&](auto convOp) {
                if (convOp.getInput() == value &&
                    convOp.getConv2dConfigAttr() &&
                    convOp.getConv2dConfigAttr().getDeallocateActivation() &&
                    convOp.getConv2dConfigAttr()
                        .getDeallocateActivation()
                        .getValue()) {
                  convOp->emitError(
                      "use-after-free detected: op deallocates its input but "
                      "is not the last user");
                  return failure();
                }
                return success();
              })
              .Default([](Operation *) { return success(); });

      if (failed(result)) {
        return failure();
      }
    }
    return success();
  }

  // Check and insert deallocation for a value after finding its last usage.
  // Returns success() if checks pass and deallocation was inserted/skipped,
  // returns failure() if validation checks fail (e.g., use-after-free
  // detected).
  LogicalResult checkAndInsertDeallocation(IRRewriter &rewriter, Value value,
                                           Operation *lastOp) {
    if (isa<func::ReturnOp>(lastOp)) {
      return success();
    }

    RankedTensorType valueTy = mlir::cast<RankedTensorType>(value.getType());
    assert(valueTy.getEncoding());
    BufferType bufferType =
        llvm::TypeSwitch<Attribute, BufferType>(valueTy.getEncoding())
            .Case<TTNNLayoutAttr, TTNNNDLayoutAttr>(
                [](auto layoutAttr) { return layoutAttr.getBufferType(); })
            .Default([](Attribute) {
              llvm_unreachable("Unsupported layout attribute type");
              // This returns a default value to avoid a compile error.
              return BufferType::DRAM;
            });

    if (bufferType == BufferType::L1) {
      // deallocate_activation is an option for Conv2d ops to deallocate
      // their input activations only if it is in L1 memory.

      // Sanity check: if there are any conv2d ops that consume this
      // value and deallocate it (via deallocate_activation=true), ensure
      // they are the last user to prevent use-after-free.
      if (failed(checkConv2dUseAfterFree(value, lastOp))) {
        return failure();
      }

      // Don't deallocate the activation after conv2d/conv_transpose2d op if
      // 'deallocate_activation' in Conv2dConfig is set to true.
      bool skipDeallocation =
          llvm::TypeSwitch<Operation *, bool>(lastOp)
              .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&](auto convOp) {
                return convOp.getInput() == value &&
                       convOp.getConv2dConfigAttr() &&
                       convOp.getConv2dConfigAttr().getDeallocateActivation() &&
                       convOp.getConv2dConfigAttr()
                           .getDeallocateActivation()
                           .getValue();
              })
              .Default([](Operation *) { return false; });

      if (skipDeallocation) {
        return success();
      }
    }

    rewriter.setInsertionPointAfter(lastOp);
    rewriter.create<DeallocateOp>(lastOp->getLoc(), value);
    return success();
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "found func that didn't have one block!");
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      // Collect all values to deallocate with their last usage operations.
      SmallVector<std::pair<Value, Operation *>> valuesToDeallocate;

      // Const eval subgraphs and trace functions may not dealloc their params
      // since they don't own them.
      if (!ttmlir::utils::isConstEvalFunc(func) &&
          !ttmlir::utils::isTraceFunc(func)) {
        // Collect func op input parameters
        for (BlockArgument arg : func.getArguments()) {
          if (!isa<RankedTensorType>(arg.getType())) {
            continue;
          }
          Operation *lastOp = getLastValueUsageOp(livenessInfo, arg);
          valuesToDeallocate.push_back({arg, lastOp});
        }
      }

      // Collect results from non-DPS ops which do not store function result
      // and are used to allocate tensors. DPS ops are handled via
      // ttnn::EmptyOp.
      func->walk([&](Operation *op) {
        if (isa<DestinationStyleOpInterface>(op)) {
          return;
        }

        // Skip ops which do not have results.
        if (op->getNumResults() == 0) {
          return;
        }

        // Iterate over all results of the op.
        for (OpResult result : op->getResults()) {
          // Check if result is ranked tensor type.
          if (!isa<RankedTensorType>(result.getType())) {
            continue;
          }

          Operation *lastOp = getLastValueUsageOp(livenessInfo, result);
          valuesToDeallocate.push_back({result, lastOp});
        }
      });

      // Check and insert deallocations for all collected values.
      for (auto [value, lastOp] : valuesToDeallocate) {
        if (failed(checkAndInsertDeallocation(rewriter, value, lastOp))) {
          signalPassFailure();
          return;
        }
      }
    });
  }
};

template <typename Derived>
class TTNNInputFunctionCreatorBase {
protected:
  void runOnOperationImpl(ModuleOp moduleOp, IRRewriter &rewriter,
                          const std::string &functionPrefix) {
    // Ensure that the module has a single region and a single block within that
    // region.
    //
    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).hasOneBlock());

    // Get the only existing block.
    //
    Block *block = moduleOp.getBody(0);

    // Find all the func.func ops in the module that are "forward" functions.
    //
    SmallVector<func::FuncOp, 1> forwardFuncOps;
    block->walk([&](func::FuncOp funcOp) {
      // Rename the function if it's named `main` to `_main`. This is done to
      // avoid name conflicts.
      //
      if (funcOp.getName() == "main") {
        rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setSymName("_main"); });
      }

      if (funcOp.isPrivate()) {
        return mlir::WalkResult::skip();
      }

      forwardFuncOps.push_back(funcOp);
      return mlir::WalkResult::advance();
    });

    // Iterate over all func ops and add input tensor functions if needed.
    //
    llvm::SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 1>
        forwardAndInputFuncOps;
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      rewriter.setInsertionPointToEnd(block);
      mlir::func::FuncOp inputFuncOp;
      // Only create input function if the forward function has inputs
      if (!forwardFuncOp.getFunctionType().getInputs().empty()) {
        inputFuncOp = createInputFunctionImpl(rewriter, forwardFuncOp.getLoc(),
                                              forwardFuncOp, functionPrefix);
      }
      forwardAndInputFuncOps.emplace_back(forwardFuncOp, inputFuncOp);
    }

    // Create a main function to call input functions and forward funcs.
    //
    createMainFunction(moduleOp, rewriter, forwardAndInputFuncOps);
  }

  func::FuncOp createInputFunctionImpl(IRRewriter &rewriter, Location loc,
                                       func::FuncOp forwardFuncOp,
                                       const std::string &functionPrefix) {
    MLIRContext *ctx = rewriter.getContext();

    // Create a new function that will handle the input tensors.
    //
    std::string inputFuncName = functionPrefix + forwardFuncOp.getName().str();

    // Create the function type.
    //
    llvm::SmallVector<mlir::Type> returnTypes =
        llvm::to_vector(forwardFuncOp.getFunctionType().getInputs());
    if (returnTypes.empty()) {
      returnTypes = {mlir::TupleType::get(ctx, {})};
    }
    FunctionType functionType = mlir::FunctionType::get(ctx, {}, returnTypes);

    // Create the function.
    //
    func::FuncOp inputFuncOp =
        rewriter.create<mlir::func::FuncOp>(loc, inputFuncName, functionType);

    // Mark this function as an input generator function.
    //
    ttmlir::utils::setFunctionType(inputFuncOp,
                                   ttmlir::utils::FunctionType::InputGenerator);

    // Add a Block to func op and set insertion point to the beginning of the
    // Block.
    //
    rewriter.modifyOpInPlace(inputFuncOp, [&]() {
      rewriter.setInsertionPointToStart(inputFuncOp.addEntryBlock());
    });

    // Create/load input tensors.
    //
    assert(
        returnTypes.size() == 1 && mlir::isa<TupleType>(returnTypes.front()) &&
        "Expected input function to return a single tuple of input tensors!");

    SmallVector<Value> tensors;
    size_t argIndex = 0;
    for (const Type &type :
         mlir::cast<mlir::TupleType>(returnTypes[0]).getTypes()) {
      // Ensure that the type is a RankedTensorType.
      //
      RankedTensorType rankedTensorType =
          mlir::dyn_cast<RankedTensorType>(type);
      assert(rankedTensorType &&
             "Expected input tensor to be of type RankedTensorType!");

      tensors.push_back(static_cast<Derived *>(this)->createTensor(
          rewriter, loc, rankedTensorType, argIndex));
      argIndex++;
    }

    // Create a tuple from the tensors.
    //
    ttcore::TupleOp tuple =
        rewriter.create<ttcore::TupleOp>(loc, returnTypes, tensors);

    // Create ReturnOp.
    //
    rewriter.create<func::ReturnOp>(forwardFuncOp.getLoc(),
                                    tuple->getResults());

    return inputFuncOp;
  }

private:
  void createMainFunction(
      ModuleOp moduleOp, IRRewriter &rewriter,
      llvm::SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 1>
          forwardAndInputFuncOps) {
    std::string mainFuncName = "main";

    // Create a function type.
    //
    FunctionType functionType = mlir::FunctionType::get(
        rewriter.getContext(), {}, rewriter.getI32Type());

    // Set insertion point to end of the block.
    //
    Block *block = moduleOp.getBody(0);
    rewriter.setInsertionPointToEnd(block);

    // Create the main function.
    //
    func::FuncOp mainFuncOp = rewriter.create<mlir::func::FuncOp>(
        moduleOp.getLoc(), mainFuncName, functionType);

    // Mark this function as a main function.
    //
    ttmlir::utils::setFunctionType(mainFuncOp,
                                   ttmlir::utils::FunctionType::Main);

    // Set insertion point to the start of the main function.
    //
    rewriter.modifyOpInPlace(mainFuncOp, [&]() {
      rewriter.setInsertionPointToStart(mainFuncOp.addEntryBlock());
    });

    for (auto [forwardFuncOp, inputFuncOp] : forwardAndInputFuncOps) {

      llvm::SmallVector<Value> operands;
      // Generate/load the input tensors for a forwardFuncOp if needed.
      // inputFuncOp will be null if the forward function has no inputs.
      //
      if (inputFuncOp) {
        func::CallOp tensors = rewriter.create<mlir::func::CallOp>(
            forwardFuncOp.getLoc(), inputFuncOp,
            /*operands=*/ValueRange());
        operands = tensors->getResults();
      }

      // Call a forward function. If there are input tensors, pass them as
      // operands.
      //
      rewriter.create<mlir::func::CallOp>(forwardFuncOp.getLoc(), forwardFuncOp,
                                          operands);
    }

    // Return 0
    //
    // func::ReturnOp requires a Value to be returned, which means that an SSA
    // needs to be returned, hence create a constant 0 via arith::ConstantOp.
    //
    Value constantZero = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(0));
    rewriter.create<func::ReturnOp>(mainFuncOp->getLoc(), constantZero);
  }
};

class TTNNCreateInputGenerators
    : public impl::TTNNCreateInputGeneratorsBase<TTNNCreateInputGenerators>,
      public TTNNInputFunctionCreatorBase<TTNNCreateInputGenerators> {

public:
  using impl::TTNNCreateInputGeneratorsBase<
      TTNNCreateInputGenerators>::TTNNCreateInputGeneratorsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());
    runOnOperationImpl(moduleOp, rewriter, "create_inputs_for_");
  }

  mlir::Value createTensor(IRRewriter &rewriter, Location loc, Type type,
                           size_t argIndex) {
    return generateTensor(rewriter, loc, type);
  }

private:
  // Currently only supports generating tensors of ones.
  // TODO(azecevic): Support generating other types of tensors that has a
  // `TTCore_CreationOpTrait`.
  // https://github.com/tenstorrent/tt-mlir/issues/3261
  //
  static mlir::Value generateTensor(IRRewriter &rewriter, Location loc,
                                    Type type) {
    MLIRContext *ctx = rewriter.getContext();
    RankedTensorType tensorType = llvm::cast<mlir::RankedTensorType>(type);

    // Get the layout attribute.
    //
    ttnn::TTNNLayoutAttr layoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

    // Get the shape of the tensor, tensor layout, and data type.
    //
    ShapeAttr shapeAttr = ttnn::ShapeAttr::get(ctx, tensorType.getShape());
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
    ttcore::DataTypeAttr dTypeAttr =
        ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());

    mlir::Value device;
    if (layoutAttr.isDeviceBufferType()) {
      device = ttnn::utils::getOrInsertDevice(rewriter,
                                              rewriter.getInsertionBlock());
    }
    // Create a new tensor of ones.
    //
    ttnn::OnesOp onesOp = rewriter.create<ttnn::OnesOp>(
        loc, tensorType, device, shapeAttr, dTypeAttr, tensorLayoutAttr,
        /*memory_config=*/nullptr);

    return onesOp;
  }
};

class TTNNLoadInputTensors
    : public impl::TTNNLoadInputTensorsBase<TTNNLoadInputTensors>,
      public TTNNInputFunctionCreatorBase<TTNNLoadInputTensors> {

public:
  using impl::TTNNLoadInputTensorsBase<
      TTNNLoadInputTensors>::TTNNLoadInputTensorsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());
    runOnOperationImpl(moduleOp, rewriter, "load_inputs_for_");
  }

  mlir::Value createTensor(IRRewriter &rewriter, Location loc, Type type,
                           size_t argIndex) {
    return loadTensor(rewriter, loc, type, argIndex, this->tensorLoadDirectory,
                      this->tensorLoadFilePrefix);
  }

private:
  static mlir::Value loadTensor(IRRewriter &rewriter, Location loc, Type type,
                                size_t argIndex,
                                std::string tensorLoadDirectory,
                                std::string tensorLoadFilePrefix) {
    RankedTensorType tensorType = llvm::cast<mlir::RankedTensorType>(type);

    // Get the layout attribute.
    //
    ttnn::TTNNLayoutAttr layoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

    // Create filename, defaults are:
    // tensorsLoadDirectory = "" (current directory)
    // tensorLoadFilePrefix = "arg"
    // For example: arg0.tensorbin, arg1.tensorbin, etc.
    //
    std::string filename;
    if (tensorLoadDirectory.empty()) {
      filename = tensorLoadFilePrefix + std::to_string(argIndex) + ".tensorbin";
    } else {
      filename = tensorLoadDirectory + "/" + tensorLoadFilePrefix +
                 std::to_string(argIndex) + ".tensorbin";
    }
    StringAttr filePathAttr = rewriter.getStringAttr(filename);

    mlir::Value device;
    if (layoutAttr.isDeviceBufferType()) {
      device = ttnn::utils::getOrInsertDevice(rewriter,
                                              rewriter.getInsertionBlock());
    }
    // Create LoadTensorOp to load tensor from disk.
    //
    ttnn::LoadTensorOp loadTensorOp = rewriter.create<ttnn::LoadTensorOp>(
        loc, tensorType, filePathAttr, device);

    return loadTensorOp;
  }
};

class TTNNCreateMainForTest
    : public impl::TTNNCreateMainForTestBase<TTNNCreateMainForTest> {

public:
  using impl::TTNNCreateMainForTestBase<
      TTNNCreateMainForTest>::TTNNCreateMainForTestBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).hasOneBlock());

    Block *block = moduleOp.getBody(0);

    // Find the forward function (_main after tuplification).
    //
    func::FuncOp forwardFuncOp = nullptr;
    block->walk([&](func::FuncOp funcOp) {
      if (ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        forwardFuncOp = funcOp;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    if (!forwardFuncOp) {
      return;
    }

    // The forward function should take a single tuple input after
    // tuplification.
    //
    FunctionType forwardFuncType = forwardFuncOp.getFunctionType();
    if (forwardFuncType.getInputs().empty()) {
      return;
    }

    createMainForTestFunction(rewriter, moduleOp, forwardFuncOp);
  }

private:
  void createMainForTestFunction(IRRewriter &rewriter, ModuleOp moduleOp,
                                 func::FuncOp forwardFuncOp) {
    MLIRContext *ctx = rewriter.getContext();
    Location loc = forwardFuncOp.getLoc();

    FunctionType forwardFuncType = forwardFuncOp.getFunctionType();

    // Create function type: same inputs as forward func + device arg.
    //
    DeviceType deviceType = DeviceType::get(ctx);
    SmallVector<Type> inputTypes(forwardFuncType.getInputs().begin(),
                                 forwardFuncType.getInputs().end());
    inputTypes.push_back(deviceType);

    FunctionType mainForTestFuncType = FunctionType::get(
        ctx, inputTypes, forwardFuncType.getResults());

    // Create the function at end of module.
    //
    Block *block = moduleOp.getBody(0);
    rewriter.setInsertionPointToEnd(block);

    func::FuncOp mainForTestOp = rewriter.create<func::FuncOp>(
        loc, "main_for_test", mainForTestFuncType);

    // Set emitpy.name attributes for parameters.
    //
    mainForTestOp.setArgAttr(0, "emitpy.name",
                             rewriter.getStringAttr("input"));
    mainForTestOp.setArgAttr(inputTypes.size() - 1, "emitpy.name",
                             rewriter.getStringAttr("device"));

    // Add entry block.
    //
    rewriter.modifyOpInPlace(mainForTestOp, [&]() {
      rewriter.setInsertionPointToStart(mainForTestOp.addEntryBlock());
    });

    // Get the input tuple and device arguments.
    //
    Value inputTuple = mainForTestOp.getArgument(0);
    Value deviceArg = mainForTestOp.getArgument(inputTypes.size() - 1);

    // The input should be a single tuple of tensors. Extract each tensor,
    // move to device if needed, and pack into a new tuple.
    //
    assert(forwardFuncType.getInputs().size() == 1 &&
           "Expected forward function to have a single tuple input!");

    TupleType inputTupleType =
        mlir::cast<TupleType>(forwardFuncType.getInputs()[0]);

    SmallVector<Value> preparedTensors;
    for (size_t i = 0; i < inputTupleType.getTypes().size(); i++) {
      Type tensorType = inputTupleType.getType(i);
      RankedTensorType rankedTensorType =
          mlir::cast<RankedTensorType>(tensorType);

      // Extract tensor from tuple.
      //
      ttcore::GetTupleElementOp getElem =
          rewriter.create<ttcore::GetTupleElementOp>(loc, inputTuple, i);

      TTNNLayoutAttr layoutAttr =
          mlir::cast<TTNNLayoutAttr>(rankedTensorType.getEncoding());

      if (layoutAttr.isDeviceBufferType()) {
        // Create MemoryConfigAttr from the layout's memory configuration.
        //
        MemoryConfigAttr memConfigAttr = MemoryConfigAttr::get(
            ctx, layoutAttr.getMemLayout(),
            BufferTypeAttr::get(ctx, layoutAttr.getBufferType()),
            /*shardSpec=*/std::nullopt);

        // Move tensor to device with the expected memory config.
        //
        Value deviceTensor = rewriter.create<ttnn::ToDeviceOp>(
            loc, rankedTensorType, getElem, deviceArg, memConfigAttr);
        preparedTensors.push_back(deviceTensor);
      } else {
        preparedTensors.push_back(getElem);
      }
    }

    // Create a new tuple from prepared tensors.
    //
    SmallVector<Type> tupleResultTypes = {inputTupleType};
    ttcore::TupleOp newTuple =
        rewriter.create<ttcore::TupleOp>(loc, tupleResultTypes, preparedTensors);

    // Call the forward function.
    //
    func::CallOp callOp = rewriter.create<func::CallOp>(
        loc, forwardFuncOp, newTuple->getResults());

    // Return the results.
    //
    rewriter.create<func::ReturnOp>(loc, callOp->getResults());
  }
};

class TTNNTuplifyTensors
    : public impl::TTNNTuplifyTensorsBase<TTNNTuplifyTensors> {

public:
  using impl::TTNNTuplifyTensorsBase<
      TTNNTuplifyTensors>::TTNNTuplifyTensorsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region.
    //
    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).getBlocks().size() == 1);

    Block *block = moduleOp.getBody(0);

    // Rename `main` to `_main` as `main` is a reserved name in C/C++.
    //
    // TODO (svuckovic): Move this to its own pass.
    //
    block->walk([&](func::FuncOp funcOp) {
      // Rename the function if it's named `main` to `_main`. This is done
      // as compiler will complain that `main` must return `int` and that
      // it's first parameter must also be an `int`.
      if (funcOp.getName() == "main") {
        rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setSymName("_main"); });
      }
    });

    // Find all the func.func ops in the module that are target functions for
    // both input tuplification, and result tuplification.
    //
    SmallVector<func::FuncOp, 1> targetFuncOpsInput;
    SmallVector<func::FuncOp, 1> targetFuncOpsResult;
    block->walk([&](func::FuncOp funcOp) {
      // Skip private functions that are not const-eval functions.
      //
      if (funcOp.isPrivate() && !ttmlir::utils::isConstEvalFunc(funcOp)) {
        return mlir::WalkResult::skip();
      }

      mlir::FunctionType functionType = funcOp.getFunctionType();

      // Check that input is not empty and that all args are of type
      // RankedTensorType.
      //
      // If `tuplifyInputIfEmpty` option is set, tuplify the input even if the
      // function has no inputs.
      //
      if ((tuplifyInputIfEmpty || !functionType.getInputs().empty()) &&
          llvm::all_of(functionType.getInputs(),
                       [](Type t) { return mlir::isa<RankedTensorType>(t); })) {
        targetFuncOpsInput.push_back(funcOp);
      }

      // Check that results are not empty and that all args are of type
      // RankedTensorType.
      //
      if (!functionType.getResults().empty() &&
          llvm::all_of(functionType.getResults(),
                       [](Type t) { return mlir::isa<RankedTensorType>(t); })) {
        targetFuncOpsResult.push_back(funcOp);
      }
      return mlir::WalkResult::advance();
    });

    // Iterate over all the input target func ops and modify their signatures.
    //
    for (mlir::func::FuncOp targetFuncOpInput : targetFuncOpsInput) {
      // Replace the signature of the target function so that all the tensor
      // arguments are packed into a single tuple.
      //
      mlir::FunctionType originalFuncType = targetFuncOpInput.getFunctionType();

      // Create TupleType object containing all input tensors.
      //
      mlir::TupleType tuplifiedInputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getInputs());

      // Create modified function type (signature) that takes the input tuple
      // as an operand.
      //
      FunctionType modifiedFuncType = originalFuncType.clone(
          tuplifiedInputTensors, originalFuncType.getResults());

      rewriter.modifyOpInPlace(targetFuncOpInput,
                               [&targetFuncOpInput, &modifiedFuncType]() {
                                 targetFuncOpInput.setType(modifiedFuncType);
                               });

      // First block of the function (often referred to as "entry block") needs
      // its arguments updated as well - the args need to match the containing
      // func's arguments; this is implemented here by first inserting the tuple
      // as the first argument of the block, inserting GetTupleElementOp ops to
      // start of the block in order to unpack tuple elements, and then
      // replacing all uses of the original block arguments with the
      // GetTupleElementOp results - after this it's finally safe to remove
      // original block arguments as they have no live uses anymore.
      //
      Block &entryBlock = targetFuncOpInput.getBlocks().front();
      constexpr size_t paramOffset = 1;
      entryBlock.insertArgument(/*index=*/0u, tuplifiedInputTensors,
                                targetFuncOpInput.getLoc());

      // Add GetTupleElementOp ops to unpack the tuple elements.
      //
      rewriter.setInsertionPointToStart(&entryBlock);
      for (size_t idx = 0; idx < originalFuncType.getNumInputs(); idx++) {
        ttcore::GetTupleElementOp getTupleElementOp =
            rewriter.create<ttcore::GetTupleElementOp>(
                targetFuncOpInput.getLoc(), targetFuncOpInput.getArgument(0),
                idx);

        // Replace all uses of the original tensor arguments with the
        // GetTupleElementOp results.
        rewriter.replaceAllUsesWith(entryBlock.getArgument(paramOffset + idx),
                                    getTupleElementOp);
      }

      // Erase original arguments.
      //
      entryBlock.eraseArguments(paramOffset,
                                originalFuncType.getInputs().size());
    }

    // Iterate over all the result target func ops and modify their signatures.
    //
    for (mlir::func::FuncOp targetFuncOpResult : targetFuncOpsResult) {
      // Replace the signature of the target function so that all the return
      // value tensors are packed into a tuple.
      //
      mlir::FunctionType originalFuncType =
          targetFuncOpResult.getFunctionType();

      // Create TupleType object containing all result tensors.
      //
      mlir::TupleType tuplifiedOutputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getResults());

      // Create modified function type (signature) that takes the result tuple
      // as the return value.
      //
      FunctionType modifiedFuncType = originalFuncType.clone(
          originalFuncType.getInputs(), tuplifiedOutputTensors);

      rewriter.modifyOpInPlace(targetFuncOpResult,
                               [&targetFuncOpResult, &modifiedFuncType]() {
                                 targetFuncOpResult.setType(modifiedFuncType);
                               });

      // Find return statement and replace with result tuple.
      //
      targetFuncOpResult.walk<WalkOrder::PostOrder, ReverseIterator>(
          [&](mlir::func::ReturnOp returnOp) {
            rewriter.setInsertionPoint(returnOp);
            ttcore::TupleOp tupleOp = rewriter.create<ttcore::TupleOp>(
                returnOp.getLoc(), returnOp.getOperands());
            rewriter.modifyOpInPlace(returnOp, [&]() {
              returnOp.getOperandsMutable().assign(tupleOp);
            });
          });
    }
  }
};

class TTNNPrepareModuleForExport
    : public impl::TTNNPrepareModuleForExportBase<TTNNPrepareModuleForExport> {

public:
  using impl::TTNNPrepareModuleForExportBase<
      TTNNPrepareModuleForExport>::TTNNPrepareModuleForExportBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region.
    //
    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).getBlocks().size() == 1);

    Block *block = moduleOp.getBody(0);

    // Find the first forward function.
    //
    func::FuncOp targetFuncOp = nullptr;
    block->walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return mlir::WalkResult::skip();
      }
      targetFuncOp = funcOp;
      return mlir::WalkResult::interrupt();
    });

    if (!targetFuncOp) {
      return;
    }

    // Rename the function to "forward".
    //
    rewriter.modifyOpInPlace(targetFuncOp,
                             [&]() { targetFuncOp.setSymName("forward"); });

    // Add device argument to the function signature.
    //
    DeviceType deviceType = DeviceType::get(&getContext());
    Block &entryBlock = targetFuncOp.getBlocks().front();
    BlockArgument deviceArg =
        entryBlock.addArgument(deviceType, targetFuncOp.getLoc());

    // Update function type to include device argument.
    //
    mlir::FunctionType originalFuncType = targetFuncOp.getFunctionType();
    SmallVector<Type> newInputTypes(originalFuncType.getInputs().begin(),
                                    originalFuncType.getInputs().end());
    newInputTypes.push_back(deviceType);
    FunctionType newFuncType = FunctionType::get(&getContext(), newInputTypes,
                                                 originalFuncType.getResults());

    rewriter.modifyOpInPlace(targetFuncOp,
                             [&]() { targetFuncOp.setType(newFuncType); });

    // Set the emitpy.name attribute for the input tuple and device arguments.
    // The input tuple should be named "input" and the device should be named
    // "device".
    //
    if (!newInputTypes.empty()) {
      targetFuncOp.setArgAttr(0, "emitpy.name",
                              rewriter.getStringAttr("input"));
    }
    targetFuncOp.setArgAttr(newInputTypes.size() - 1, "emitpy.name",
                            rewriter.getStringAttr("device"));

    // Find all GetDeviceOp operations and replace their uses with the device
    // argument.
    //
    SmallVector<ttnn::GetDeviceOp> getDeviceOps;
    targetFuncOp.walk(
        [&](ttnn::GetDeviceOp op) { getDeviceOps.push_back(op); });

    for (ttnn::GetDeviceOp getDeviceOp : getDeviceOps) {
      rewriter.replaceOp(getDeviceOp, deviceArg);
    }
  }
};

} // namespace mlir::tt::ttnn
