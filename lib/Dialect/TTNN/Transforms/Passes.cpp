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
    TTNNLayoutAttr layoutAttr =
        mlir::cast<TTNNLayoutAttr>(valueTy.getEncoding());

    if (layoutAttr.getBufferType() == BufferType::L1) {
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
          !utils::isTTNNTraceFunc(func)) {
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
public:
  struct FunctionTriple {
    mlir::func::FuncOp forwardFunc;
    mlir::func::FuncOp inputGenFunc;
    mlir::func::FuncOp paramGenFunc;
  };

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

      if (funcOp.isPrivate() || ttmlir::utils::isConstEvalFunc(funcOp)) {
        return mlir::WalkResult::skip();
      }

      forwardFuncOps.push_back(funcOp);
      return mlir::WalkResult::advance();
    });

    // Iterate over all func ops and add input/parameter generator functions if
    // needed.
    //
    llvm::SmallVector<FunctionTriple, 1> functionTriples;
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      // Check if the forward function has any arguments. If it doesn't, we
      // do not create generator functions for it.
      llvm::SmallVector<mlir::func::FuncOp> generatorFuncs = {nullptr, nullptr};

      ::mlir::Region::BlockArgListType args = forwardFuncOp.getArguments();
      assert(args.size() >= 0 && args.size() <= 2 &&
             "Expected forward function to have zero, one or two arguments!");
      for (size_t i = 0; i < args.size(); i++) {
        rewriter.setInsertionPointToEnd(block);
        assert(isa<TupleType>(args[i].getType()) &&
               "Expected function argument to be of TupleType!");
        TupleType argType = mlir::cast<TupleType>(args[i].getType());

        auto typeAttr =
            forwardFuncOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
                args[i].getArgNumber(), ttcore::ArgumentTypeAttr::name);

        assert(typeAttr && "Expected ArgumentType attribute on all tuple "
                           "arguments in forward function.");

        ttcore::ArgumentType argTypeValue = typeAttr.getValue();
        std::string generatorFuncName;
        if (argTypeValue == ttcore::ArgumentType::Input) {
          generatorFuncName =
              "create_inputs_for_" + forwardFuncOp.getName().str();
        } else if (argTypeValue == ttcore::ArgumentType::Parameter) {
          generatorFuncName =
              "create_params_for_" + forwardFuncOp.getName().str();
        } else {
          llvm::errs() << "Unexpected ArgumentType value on function argument "
                          "in forward function: "
                       << static_cast<uint32_t>(argTypeValue) << "\n";
          llvm::report_fatal_error(
              "Aborting due to unexpected ArgumentType value.");
        }

        auto originalArgPositionsAttr =
            forwardFuncOp.getArgAttrOfType<ttcore::OriginalArgPositionsAttr>(
                args[i].getArgNumber(), ttcore::OriginalArgPositionsAttr::name);

        assert(originalArgPositionsAttr &&
               "Expected OriginalArgPositions attribute on all tuple arguments "
               "in forward function.");

        generatorFuncs[i] = createGeneratorFunctionImpl(
            rewriter, forwardFuncOp.getLoc(), argType,
            originalArgPositionsAttr.getPositions(), generatorFuncName);
      }

      functionTriples.push_back(
          {forwardFuncOp, generatorFuncs[0], generatorFuncs[1]});
    }

    // Create a main function to call generator functions and forward funcs.
    //
    createMainFunction(moduleOp, rewriter, functionTriples);
  }

  func::FuncOp
  createGeneratorFunctionImpl(IRRewriter &rewriter, Location loc,
                              Type generatorFuncType,
                              ArrayRef<unsigned> originalArgPositions,
                              const std::string &generatorFuncName) {
    MLIRContext *ctx = rewriter.getContext();

    FunctionType functionType =
        mlir::FunctionType::get(ctx, {}, {generatorFuncType});

    // Create the function.
    //
    func::FuncOp genFuncOp = rewriter.create<mlir::func::FuncOp>(
        loc, generatorFuncName, functionType);

    // Add a Block to func op and set insertion point to the beginning of the
    // Block.
    //
    rewriter.modifyOpInPlace(genFuncOp, [&]() {
      rewriter.setInsertionPointToStart(genFuncOp.addEntryBlock());
    });

    // Create/load tensors for the selected tuple (input or parameter).
    //
    SmallVector<Value> tensors;
    size_t argIndex = 0;
    for (const Type &type :
         mlir::cast<mlir::TupleType>(generatorFuncType).getTypes()) {
      // Ensure that the type is a RankedTensorType.
      //
      RankedTensorType rankedTensorType =
          mlir::dyn_cast<RankedTensorType>(type);
      assert(rankedTensorType &&
             "Expected tensor to be of type RankedTensorType!");

      tensors.push_back(static_cast<Derived *>(this)->createTensor(
          rewriter, loc, rankedTensorType, originalArgPositions[argIndex]));
      argIndex++;
    }

    // Create tuple from the tensors.
    //
    ttcore::TupleOp tuple =
        rewriter.create<ttcore::TupleOp>(loc, generatorFuncType, tensors);

    // Create ReturnOp with the tuple.
    //
    rewriter.create<func::ReturnOp>(loc, tuple.getResult());

    return genFuncOp;
  }

private:
  void
  createMainFunction(ModuleOp moduleOp, IRRewriter &rewriter,
                     llvm::SmallVector<FunctionTriple, 1> functionTriples) {
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

    // Set insertion point to the start of the main function.
    //
    rewriter.modifyOpInPlace(mainFuncOp, [&]() {
      rewriter.setInsertionPointToStart(mainFuncOp.addEntryBlock());
    });

    for (auto &triple : functionTriples) {
      llvm::SmallVector<Value> operands;

      // Generate/load the input tensors and parameters for a forwardFuncOp if
      // needed.
      //
      if (triple.inputGenFunc) {
        // Call input generator
        func::CallOp inputCall = rewriter.create<mlir::func::CallOp>(
            triple.forwardFunc.getLoc(), triple.inputGenFunc,
            /*operands=*/ValueRange());

        operands.push_back(inputCall.getResult(0));
      }

      if (triple.paramGenFunc) {
        // Call parameter generator
        func::CallOp paramCall = rewriter.create<mlir::func::CallOp>(
            triple.forwardFunc.getLoc(), triple.paramGenFunc,
            /*operands=*/ValueRange());

        operands.push_back(paramCall.getResult(0));
      }

      // Call a forward function. If there are input/parameter tensors, pass
      // them as operands.
      //
      rewriter.create<mlir::func::CallOp>(triple.forwardFunc.getLoc(),
                                          triple.forwardFunc, operands);
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

class TTNNTuplifyTensors
    : public impl::TTNNTuplifyTensorsBase<TTNNTuplifyTensors> {

private:
  // Helper structure to hold information about a group of arguments to be
  // tuplified.
  struct TupleGroup {
    llvm::SmallVector<Type> types;
    llvm::SmallVector<size_t> indices;
    llvm::SmallVector<unsigned> originalArgPositions;
    llvm::SmallVector<DictionaryAttr> attrs;

    size_t count() const { return types.size(); }
  };

  // Helper function to tuplify function inputs. Takes a function and a list of
  // tuple groups, where each group represents arguments to be packed into a
  // single tuple.
  void tuplifyFunctionInputs(mlir::func::FuncOp funcOp,
                             llvm::ArrayRef<TupleGroup> tupleGroups,
                             llvm::ArrayRef<DictionaryAttr> tupleArgAttrs,
                             IRRewriter &rewriter) {
    mlir::FunctionType originalFuncType = funcOp.getFunctionType();

    // Create TupleType objects for each group.
    llvm::SmallVector<Type> newInputTypes;
    for (const auto &group : tupleGroups) {
      newInputTypes.push_back(mlir::TupleType::get(&getContext(), group.types));
    }

    // Create modified function type with tuple inputs.
    FunctionType modifiedFuncType = FunctionType::get(
        &getContext(), newInputTypes, originalFuncType.getResults());

    // Update function signature.
    rewriter.modifyOpInPlace(funcOp, [&]() {
      funcOp.setType(modifiedFuncType);
      if (!tupleArgAttrs.empty()) {
        funcOp.setAllArgAttrs(tupleArgAttrs);
      }
    });

    // Update entry block arguments.
    Block &entryBlock = funcOp.getBlocks().front();
    size_t tupleOffset = tupleGroups.size();

    // Insert tuple arguments at the beginning.
    for (size_t i = 0; i < tupleGroups.size(); i++) {
      entryBlock.insertArgument(i, cast<mlir::TupleType>(newInputTypes[i]),
                                funcOp.getLoc());
    }

    // Add GetTupleElementOp ops to unpack each tuple and preserve attributes.
    rewriter.setInsertionPointToStart(&entryBlock);
    for (size_t groupIdx = 0; groupIdx < tupleGroups.size(); groupIdx++) {
      const auto &group = tupleGroups[groupIdx];
      for (size_t elemIdx = 0; elemIdx < group.indices.size(); elemIdx++) {
        size_t originalArgIdx = group.indices[elemIdx];
        ttcore::GetTupleElementOp getTupleElementOp =
            rewriter.create<ttcore::GetTupleElementOp>(
                funcOp.getLoc(), funcOp.getArgument(groupIdx), elemIdx);

        // Copy attributes from the saved argument attributes.
        if (elemIdx < group.attrs.size() && group.attrs[elemIdx]) {
          for (auto attr : group.attrs[elemIdx]) {
            getTupleElementOp->setAttr(attr.getName(), attr.getValue());
          }
        }

        // Replace all uses of the original tensor arguments.
        rewriter.replaceAllUsesWith(
            entryBlock.getArgument(tupleOffset + originalArgIdx),
            getTupleElementOp);
      }
    }

    // Erase original arguments.
    entryBlock.eraseArguments(tupleOffset, originalFuncType.getInputs().size());
  }

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
    // Separate into two categories:
    // - const_eval functions: tuplify all args into a single tuple
    // - regular functions: tuplify into two tuples (inputs and parameters)
    //
    SmallVector<func::FuncOp, 1> targetFuncOpsInput; // Regular functions
    SmallVector<func::FuncOp, 1>
        targetConstEvalFuncOpsInput; // Const_eval functions
    SmallVector<func::FuncOp, 1> targetFuncOpsResult;
    block->walk([&](func::FuncOp funcOp) {
      // Skip function declarations (CPU-hoisted functions).
      //
      if (funcOp.isDeclaration()) {
        return mlir::WalkResult::skip();
      }

      mlir::FunctionType functionType = funcOp.getFunctionType();
      bool isConstEval = ttmlir::utils::isConstEvalFunc(funcOp);

      // Check that input is not empty and that all args are of type
      // RankedTensorType.
      //
      // If `tuplifyInputIfEmpty` option is set, tuplify the input even if the
      // function has no inputs.
      //
      if ((tuplifyInputIfEmpty || !functionType.getInputs().empty()) &&
          llvm::all_of(functionType.getInputs(),
                       [](Type t) { return mlir::isa<RankedTensorType>(t); })) {
        if (isConstEval) {
          targetConstEvalFuncOpsInput.push_back(funcOp);
        } else {
          targetFuncOpsInput.push_back(funcOp);
        }
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

    // Iterate over all non const-eval func ops and and tuplify their inputs
    // into a two tuples: one for inputs and one for parameters.
    //
    for (mlir::func::FuncOp targetFuncOpInput : targetFuncOpsInput) {
      // Replace the signature of the target function so that the tensor
      // arguments are packed into two tuples: one for inputs and one for
      // parameters.
      //
      mlir::FunctionType originalFuncType = targetFuncOpInput.getFunctionType();

      // Separate arguments into inputs and parameters based on their
      // ArgumentType attribute. Also save argument attributes before modifying
      // the function signature.
      //
      TupleGroup inputGroup;
      TupleGroup paramGroup;

      for (size_t idx = 0; idx < originalFuncType.getNumInputs(); idx++) {
        auto typeAttr =
            targetFuncOpInput.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
                idx, ttcore::ArgumentTypeAttr::name);

        assert(typeAttr && "Expected ArgumentType attribute on argument");

        auto originalArgPosAttr =
            targetFuncOpInput.getArgAttrOfType<ttcore::OriginalArgPositionAttr>(
                idx, ttcore::OriginalArgPositionAttr::name);

        assert(originalArgPosAttr &&
               "Expected original_arg_position attribute on argument");

        auto argTypeValue = typeAttr.getValue();
        if (argTypeValue == ttcore::ArgumentType::Input ||
            argTypeValue == ttcore::ArgumentType::Default) {
          inputGroup.types.push_back(originalFuncType.getInput(idx));
          inputGroup.indices.push_back(idx);
          inputGroup.attrs.push_back(targetFuncOpInput.getArgAttrDict(idx));
          inputGroup.originalArgPositions.push_back(
              originalArgPosAttr.getPosition());
        } else if (argTypeValue == ttcore::ArgumentType::Parameter ||
                   argTypeValue == ttcore::ArgumentType::Constant) {
          paramGroup.types.push_back(originalFuncType.getInput(idx));
          paramGroup.indices.push_back(idx);
          paramGroup.attrs.push_back(targetFuncOpInput.getArgAttrDict(idx));
          paramGroup.originalArgPositions.push_back(
              originalArgPosAttr.getPosition());
        }
      }

      // Create argument attributes for the two tuple arguments.
      // Mark the first tuple as "input" type and second as "parameter" type.
      //
      llvm::SmallVector<TupleGroup> tupleGroups;
      llvm::SmallVector<DictionaryAttr> tupleArgAttrs;
      if (inputGroup.count() > 0 || tuplifyInputIfEmpty) {
        tupleGroups.push_back(inputGroup);
        llvm::SmallVector<mlir::NamedAttribute> inputTupleAttrs;
        inputTupleAttrs.push_back(rewriter.getNamedAttr(
            ttcore::ArgumentTypeAttr::name,
            ttcore::ArgumentTypeAttr::get(&getContext(),
                                          ttcore::ArgumentType::Input)));
        inputTupleAttrs.push_back(rewriter.getNamedAttr(
            ttcore::OriginalArgPositionsAttr::name,
            ttcore::OriginalArgPositionsAttr::get(
                &getContext(), inputGroup.originalArgPositions)));
        tupleArgAttrs.push_back(rewriter.getDictionaryAttr(inputTupleAttrs));
      }
      if (paramGroup.count() > 0 || tuplifyInputIfEmpty) {
        tupleGroups.push_back(paramGroup);
        llvm::SmallVector<mlir::NamedAttribute> paramTupleAttrs;
        paramTupleAttrs.push_back(rewriter.getNamedAttr(
            ttcore::ArgumentTypeAttr::name,
            ttcore::ArgumentTypeAttr::get(&getContext(),
                                          ttcore::ArgumentType::Parameter)));
        paramTupleAttrs.push_back(rewriter.getNamedAttr(
            ttcore::OriginalArgPositionsAttr::name,
            ttcore::OriginalArgPositionsAttr::get(
                &getContext(), paramGroup.originalArgPositions)));
        tupleArgAttrs.push_back(rewriter.getDictionaryAttr(paramTupleAttrs));
      }

      // Use helper function to perform the tuplification.
      tuplifyFunctionInputs(targetFuncOpInput, tupleGroups, tupleArgAttrs,
                            rewriter);
    }

    // Iterate over const_eval functions and tuplify their inputs into a single
    // tuple (no input/parameter split).
    //
    for (mlir::func::FuncOp targetConstEvalFunc : targetConstEvalFuncOpsInput) {
      mlir::FunctionType originalFuncType =
          targetConstEvalFunc.getFunctionType();

      // Create a single tuple group containing all arguments.
      //
      TupleGroup singleGroup;
      for (size_t idx = 0; idx < originalFuncType.getNumInputs(); idx++) {
        singleGroup.types.push_back(originalFuncType.getInput(idx));
        singleGroup.indices.push_back(idx);
        singleGroup.attrs.push_back(targetConstEvalFunc.getArgAttrDict(idx));
      }

      // Use helper function to perform the tuplification.
      // No special tuple argument attributes needed for const_eval functions.
      tuplifyFunctionInputs(targetConstEvalFunc, {singleGroup}, {}, rewriter);
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
              returnOp.getOperandsMutable().assign(tupleOp.getResult());
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

    // Find the first public (non-private, non-const-eval) function.
    //
    func::FuncOp targetFuncOp = nullptr;
    block->walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration() || funcOp.isPrivate() ||
          ttmlir::utils::isConstEvalFunc(funcOp)) {
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
