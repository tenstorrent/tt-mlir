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
#define GEN_PASS_DEF_TTNNSPLITACTIVATIONSANDWEIGHTS
#define GEN_PASS_DEF_TTNNEMPYWORKAROUNDS
#define GEN_PASS_DEF_TTNNPREPAREMODULEFOREXPORT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDeallocate : public impl::TTNNDeallocateBase<TTNNDeallocate> {

public:
  using impl::TTNNDeallocateBase<TTNNDeallocate>::TTNNDeallocateBase;

  Operation *getLastValueUsageOp(const LivenessBlockInfo *livenessInfo,
                                 Value value) {
    Value currentValue = value;
    Operation *startOp = livenessInfo->getStartOperation(currentValue);
    Operation *endOp = livenessInfo->getEndOperation(currentValue, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(currentValue);
        });

    // In case of DPS op keep going until we find the last usage of the tensor.
    //
    //
    // Follow aliasing chains until we reach the true final user.
    // Today we model:
    //  1) DPS init operands (tensor flows into tied result)
    //  2) Identity mesh_shard (input/result alias for shape tracking)
    while (true) {
      if (opOperandIter != endOp->getOpOperands().end() &&
          isa<DestinationStyleOpInterface>(endOp) &&
          cast<DestinationStyleOpInterface>(endOp).isDpsInit(
              &(*opOperandIter))) {
        OpResult result =
            cast<DestinationStyleOpInterface>(endOp).getTiedOpResult(
                &(*opOperandIter));
        currentValue = result;
        endOp = livenessInfo->getEndOperation(currentValue, endOp);
        opOperandIter =
            llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
              return opOperand.is(currentValue);
            });
        continue;
      }

      if (auto meshShardOp = dyn_cast<ttnn::MeshShardOp>(endOp);
          meshShardOp &&
          meshShardOp.getShardType() == ttcore::MeshShardType::Identity &&
          meshShardOp.getInput() == currentValue) {
        currentValue = meshShardOp.getResult();
        endOp = livenessInfo->getEndOperation(currentValue, endOp);
        opOperandIter =
            llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
              return opOperand.is(currentValue);
            });
        continue;
      }

      break;
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

        // Identity mesh_shard is an alias-only op (no new storage ownership),
        // so its result must not receive a separate deallocate.
        if (auto meshShardOp = dyn_cast<ttnn::MeshShardOp>(op);
            meshShardOp &&
            meshShardOp.getShardType() == ttcore::MeshShardType::Identity) {
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

/// Recovers the original flat argument indices for activations and weights
/// from the ttcore.original_argument_types attribute. This is needed to
static void recoverOriginalArgIndices(func::FuncOp funcOp,
                                      SmallVector<size_t> &activationIndices,
                                      SmallVector<size_t> &weightIndices) {
  auto origTypesAttr =
      funcOp->getAttrOfType<ArrayAttr>("ttcore.original_argument_types");
  assert(origTypesAttr && "Expected ttcore.original_argument_types on function "
                          "that has split activation and weight inputs!");
  for (unsigned i = 0; i < origTypesAttr.size(); ++i) {
    auto typeAttr = mlir::cast<ttcore::ArgumentTypeAttr>(origTypesAttr[i]);
    if (typeAttr.getValue() == ttcore::ArgumentType::Input) {
      activationIndices.push_back(i);
    } else {
      weightIndices.push_back(i);
    }
  }
}

template <typename Derived>
class TTNNInputFunctionCreatorBase {
  /// Bundles the input generator functions created for a single forward
  /// function.
  struct ForwardFuncInputGeneratorOps {
    func::FuncOp forwardFuncOp;
    SmallVector<func::FuncOp, 2> inputGeneratorOps;
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

      if (!ttmlir::utils::isForwardDeviceFunc(funcOp) || funcOp.isPrivate()) {
        return mlir::WalkResult::skip();
      }

      if (funcOp.getFunctionType().getNumInputs() > 0) {
        forwardFuncOps.push_back(funcOp);
      }

      return mlir::WalkResult::advance();
    });

    // Iterate over all forward functions and add input tensor functions if
    // needed.
    //
    SmallVector<ForwardFuncInputGeneratorOps, 1> forwardAndInputFuncOps;
    for (func::FuncOp forwardFuncOp : forwardFuncOps) {
      ForwardFuncInputGeneratorOps entry;
      entry.forwardFuncOp = forwardFuncOp;

      if (ttmlir::utils::isSplitInput(forwardFuncOp)) {
        SmallVector<size_t> activationOrigIndices, weightOrigIndices;
        recoverOriginalArgIndices(forwardFuncOp, activationOrigIndices,
                                  weightOrigIndices);

        for (auto type : forwardFuncOp.getFunctionType().getInputs()) {
          if (auto tupleType = dyn_cast<mlir::TupleType>(type)) {
            entry.inputGeneratorOps.push_back(createTupleInputFunctionImpl(
                rewriter, block, forwardFuncOp.getLoc(), forwardFuncOp,
                functionPrefix + "activations_for_", tupleType,
                activationOrigIndices));
          }
          if (isa<ttcore::DictType>(type)) {
            entry.inputGeneratorOps.push_back(createDictInputFunctionImpl(
                rewriter, block, forwardFuncOp.getLoc(), forwardFuncOp,
                functionPrefix + "weights_for_", weightOrigIndices));
          }
        }
      } else {
        auto inputs = forwardFuncOp.getFunctionType().getInputs();
        assert(inputs.size() == 1 && isa<mlir::TupleType>(inputs[0]) &&
               "Expected upstream pass to have tuplified inputs!");
        entry.inputGeneratorOps.push_back(createTupleInputFunctionImpl(
            rewriter, block, forwardFuncOp.getLoc(), forwardFuncOp,
            functionPrefix + "inputs_for_",
            mlir::cast<mlir::TupleType>(inputs[0])));
      }

      forwardAndInputFuncOps.push_back(entry);
    }

    // Create a main function to call input functions and forward funcs.
    //
    createMainFunction(moduleOp, rewriter, forwardAndInputFuncOps);
  }

  // Creates a function input generator for a tuple of tensors.
  func::FuncOp createTupleInputFunctionImpl(
      IRRewriter &rewriter, Block *block, Location loc,
      func::FuncOp forwardFuncOp, const std::string &functionPrefix,
      mlir::TupleType tupleType, ArrayRef<size_t> originalArgIndices = {}) {
    MLIRContext *ctx = rewriter.getContext();
    std::string funcName = functionPrefix + forwardFuncOp.getName().str();
    FunctionType functionType = mlir::FunctionType::get(ctx, {}, tupleType);

    // Create the function.
    //
    rewriter.setInsertionPointToEnd(block);
    func::FuncOp funcOp =
        rewriter.create<mlir::func::FuncOp>(loc, funcName, functionType);

    // Mark this function as an input generator function.
    //
    ttmlir::utils::setFunctionType(funcOp,
                                   ttmlir::utils::FunctionType::InputGenerator);

    // Add a Block to func op and set insertion point to the beginning of the
    // Block.
    //
    rewriter.modifyOpInPlace(funcOp, [&]() {
      rewriter.setInsertionPointToStart(funcOp.addEntryBlock());
    });

    // Create/load tensors.
    //
    SmallVector<Value> tensors;
    for (auto [i, type] : llvm::enumerate(tupleType.getTypes())) {
      auto rankedTensorType = mlir::dyn_cast<RankedTensorType>(type);
      assert(rankedTensorType &&
             "Expected tensor to be of type RankedTensorType!");

      size_t argIndex = originalArgIndices.empty() ? i : originalArgIndices[i];
      tensors.push_back(static_cast<Derived *>(this)->createTensor(
          rewriter, loc, rankedTensorType, argIndex));
    }

    // Create a tuple from the tensors.
    //
    ttcore::TupleOp tuple =
        rewriter.create<ttcore::TupleOp>(loc, tupleType, tensors);

    // Create ReturnOp.
    //
    rewriter.create<func::ReturnOp>(forwardFuncOp.getLoc(),
                                    tuple->getResults());

    return funcOp;
  }

  // Creates a function input generator for a dictionary of tensors. This is
  // used to pack weights into a dictionary.
  func::FuncOp
  createDictInputFunctionImpl(IRRewriter &rewriter, Block *block, Location loc,
                              func::FuncOp forwardFuncOp,
                              const std::string &functionPrefix,
                              ArrayRef<size_t> originalArgIndices = {}) {
    MLIRContext *ctx = rewriter.getContext();
    std::string forwardFuncName = forwardFuncOp.getName().str();
    std::string funcName = functionPrefix + forwardFuncName;
    auto dictType = ttcore::DictType::get(ctx);
    auto functionType = mlir::FunctionType::get(ctx, {}, {dictType});

    // Create the global dictionary (module-level).
    //
    rewriter.setInsertionPointToEnd(block);
    std::string dictName = forwardFuncName + "_weights";
    rewriter.create<ttcore::GlobalOp>(loc, dictName, dictType,
                                      /*index=*/IntegerAttr());

    // Create the function.
    //
    func::FuncOp funcOp =
        rewriter.create<mlir::func::FuncOp>(loc, funcName, functionType);

    // Mark this function as an input generator function.
    //
    ttmlir::utils::setFunctionType(funcOp,
                                   ttmlir::utils::FunctionType::InputGenerator);

    // Add a Block to func op and set insertion point to the beginning of the
    // Block.
    //
    rewriter.modifyOpInPlace(funcOp, [&]() {
      rewriter.setInsertionPointToStart(funcOp.addEntryBlock());
    });

    // Retrieve the global dictionary.
    //
    Value dict = rewriter.create<ttcore::GetGlobalOp>(loc, dictType, dictName)
                     ->getResult(0);

    // Collect tensor names and types from GetKeyValueOp ops in the
    // forward function body. All GetKeyValueOps in the forward function body
    // are relevant since at this point all of them are created by
    // TTNNSplitActivationsAndWeights pass.
    //
    SmallVector<std::pair<Attribute, Type>> namesAndTypes;
    forwardFuncOp.walk([&](ttcore::GetKeyValueOp op) {
      namesAndTypes.push_back({op.getKeyAttr(), op.getResult(0).getType()});
    });

    // Create/load tensors and store them in the dictionary.
    //
    for (auto [i, nameAndType] : llvm::enumerate(namesAndTypes)) {
      auto &[key, type] = nameAndType;
      auto rankedTensorType = mlir::dyn_cast<RankedTensorType>(type);
      assert(rankedTensorType &&
             "Expected tensor to be of type RankedTensorType!");

      size_t argIndex = originalArgIndices.empty() ? i : originalArgIndices[i];
      Value tensor = static_cast<Derived *>(this)->createTensor(
          rewriter, loc, rankedTensorType, argIndex);

      rewriter.create<ttcore::SetKeyValueOp>(loc, dict, key, tensor);
    }

    // Create ReturnOp.
    //
    rewriter.create<func::ReturnOp>(funcOp.getLoc(), dict);

    return funcOp;
  }

private:
  void createMainFunction(
      ModuleOp moduleOp, IRRewriter &rewriter,
      SmallVector<ForwardFuncInputGeneratorOps, 1> forwardAndInputFuncOps) {
    MLIRContext *ctx = rewriter.getContext();
    std::string mainFuncName = "main";

    // Create a function type.
    //
    FunctionType functionType =
        mlir::FunctionType::get(ctx, {}, rewriter.getI32Type());

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

    for (auto &[forwardFuncOp, inputGeneratorOps] : forwardAndInputFuncOps) {
      llvm::SmallVector<Value> operands;
      for (auto generatorOp : inputGeneratorOps) {
        func::CallOp call = rewriter.create<mlir::func::CallOp>(
            forwardFuncOp.getLoc(), generatorOp,
            /*operands=*/ValueRange());
        operands.push_back(call.getResult(0));
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
    runOnOperationImpl(moduleOp, rewriter, "create_");
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
    runOnOperationImpl(moduleOp, rewriter, "load_");
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

    FunctionType forwardFuncType = forwardFuncOp.getFunctionType();
    if (forwardFuncType.getInputs().empty()) {
      return;
    }

    // Inject device argument into the forward function, replacing GetDeviceOp
    // uses. This follows the same pattern as TTNNPrepareModuleForExport so
    // that the EmitPy LoadCachedOpConversionPattern can find the device arg
    // in the enclosing function and propagate it to const-eval wrapper calls.
    //
    // Const-eval functions get device injected separately by
    // targetModuleConversion in the EmitPy conversion pass (it must happen
    // there because load_cached ops only accept tensors, not device types,
    // so the module verifier would reject a device operand on load_cached).
    //
    injectDeviceArg(rewriter, forwardFuncOp);

    // After modifying _main's signature, existing callers (e.g. the main()
    // entry point created by input generators) need to pass device too.
    //
    fixExistingCallsToForward(rewriter, moduleOp, forwardFuncOp);

    createMainForTestFunction(rewriter, moduleOp, forwardFuncOp);
  }

private:
  // Add a device argument to a function and replace all GetDeviceOp uses
  // with it. Mirrors TTNNPrepareModuleForExport's device handling.
  //
  void injectDeviceArg(IRRewriter &rewriter, func::FuncOp funcOp) {
    MLIRContext *ctx = rewriter.getContext();
    DeviceType deviceType = DeviceType::get(ctx);

    // Add device argument to the function signature.
    //
    auto originalFuncType = funcOp.getFunctionType();
    SmallVector<Type> newInputTypes(originalFuncType.getInputs().begin(),
                                    originalFuncType.getInputs().end());
    newInputTypes.push_back(deviceType);
    FunctionType newFuncType =
        FunctionType::get(ctx, newInputTypes, originalFuncType.getResults());

    funcOp.setFunctionType(newFuncType);

    // Pad the existing arg_attrs array to match the new argument count.
    // setArgAttr does not auto-resize, so accessing the new index would
    // trigger an out-of-bounds assertion.
    //
    ArrayAttr existingArgAttrs = funcOp.getAllArgAttrs();
    if (existingArgAttrs) {
      SmallVector<Attribute> paddedAttrs(existingArgAttrs.begin(),
                                         existingArgAttrs.end());
      paddedAttrs.resize(newInputTypes.size(), DictionaryAttr::get(ctx));
      funcOp.setAllArgAttrs(paddedAttrs);
    }

    Block &entryBlock = funcOp.getBody().front();
    BlockArgument deviceArg =
        entryBlock.addArgument(deviceType, funcOp.getLoc());
    funcOp.setArgAttr(newInputTypes.size() - 1, "emitpy.name",
                      rewriter.getStringAttr("device"));

    // Replace all GetDeviceOp operations with the new device argument.
    //
    SmallVector<ttnn::GetDeviceOp> getDeviceOps;
    funcOp.walk([&](ttnn::GetDeviceOp op) { getDeviceOps.push_back(op); });
    for (ttnn::GetDeviceOp op : getDeviceOps) {
      rewriter.replaceOp(op, deviceArg);
    }
  }

  // Fix existing CallOps to the forward function after its signature changed.
  // The main() entry point (created by input generators) calls _main(tuple),
  // but after device injection _main expects (tuple, device).
  //
  void fixExistingCallsToForward(IRRewriter &rewriter, ModuleOp moduleOp,
                                 func::FuncOp forwardFuncOp) {
    StringRef forwardName = forwardFuncOp.getName();
    SmallVector<func::CallOp> callsToFix;

    moduleOp.walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == forwardName &&
          callOp.getNumOperands() < forwardFuncOp.getNumArguments()) {
        callsToFix.push_back(callOp);
      }
    });

    for (func::CallOp callOp : callsToFix) {
      rewriter.setInsertionPoint(callOp);
      ttnn::GetDeviceOp deviceOp =
          ttnn::utils::getOrInsertDevice(rewriter, callOp->getBlock());

      SmallVector<Value> newOperands(callOp.getOperands());
      newOperands.push_back(deviceOp);
      rewriter.replaceOpWithNewOp<func::CallOp>(callOp, forwardFuncOp,
                                                newOperands);
    }
  }

  // Inject device argument into all const-eval functions in the module.
  // Mirrors targetModuleConversion in the EmitPy conversion pass.
  //
  void injectDeviceArgIntoConstEvalFuncs(IRRewriter &rewriter,
                                         ModuleOp moduleOp) {
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isConstEvalFunc(funcOp) || funcOp.isExternal()) {
        return;
      }
      injectDeviceArg(rewriter, funcOp);
    });
  }

  void createMainForTestFunction(IRRewriter &rewriter, ModuleOp moduleOp,
                                 func::FuncOp forwardFuncOp) {
    MLIRContext *ctx = rewriter.getContext();
    Location loc = forwardFuncOp.getLoc();

    // After injectDeviceArg, the forward function already has the device arg.
    // main_for_test has the same signature: (input_tuple, device) -> results.
    //
    FunctionType forwardFuncType = forwardFuncOp.getFunctionType();

    // Create the function at end of module.
    //
    Block *block = moduleOp.getBody(0);
    rewriter.setInsertionPointToEnd(block);

    func::FuncOp mainForTestOp =
        rewriter.create<func::FuncOp>(loc, "main_for_test", forwardFuncType);

    // Set emitpy.name attributes for parameters.
    //
    unsigned numArgs = forwardFuncType.getNumInputs();
    mainForTestOp.setArgAttr(0, "emitpy.name", rewriter.getStringAttr("input"));
    mainForTestOp.setArgAttr(numArgs - 1, "emitpy.name",
                             rewriter.getStringAttr("device"));

    // Add entry block.
    //
    rewriter.modifyOpInPlace(mainForTestOp, [&]() {
      rewriter.setInsertionPointToStart(mainForTestOp.addEntryBlock());
    });

    // Get the input tuple and device arguments.
    //
    Value inputTuple = mainForTestOp.getArgument(0);
    Value deviceArg = mainForTestOp.getArgument(numArgs - 1);

    // The forward function should have (tuple, device) as inputs after
    // device injection.
    //
    assert(forwardFuncType.getNumInputs() == 2 &&
           "Expected forward function to have (tuple, device) inputs!");

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
    ttcore::TupleOp newTuple = rewriter.create<ttcore::TupleOp>(
        loc, tupleResultTypes, preparedTensors);

    // Call the forward function, passing the prepared inputs and device.
    //
    SmallVector<Value> callArgs;
    callArgs.append(newTuple->getResults().begin(),
                    newTuple->getResults().end());
    callArgs.push_back(deviceArg);
    func::CallOp callOp =
        rewriter.create<func::CallOp>(loc, forwardFuncOp, callArgs);

    // Return the results.
    //
    rewriter.create<func::ReturnOp>(loc, callOp->getResults());
  }
};

// Splits the inputs of the forward functions into activations and
// weights. Only performed on forward functions which inputs have
// ttcore.argument_type attributes set, so that each input argument can
// be properly classified as an activation or a weight.
class TTNNSplitActivationsAndWeights
    : public impl::TTNNSplitActivationsAndWeightsBase<
          TTNNSplitActivationsAndWeights> {
public:
  using impl::TTNNSplitActivationsAndWeightsBase<
      TTNNSplitActivationsAndWeights>::TTNNSplitActivationsAndWeightsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext &ctx = getContext();
    IRRewriter rewriter(&ctx);

    // Ensure that the module has a single region and a single block within
    // that region.
    //
    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).hasOneBlock());

    Block *block = moduleOp.getBody(0);

    // Collect forward functions eligible for splitting.
    //
    SmallVector<func::FuncOp> forwardFuncOps;
    block->walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return mlir::WalkResult::skip();
      }
      if (funcOp.getFunctionType().getNumInputs() == 0) {
        return mlir::WalkResult::skip();
      }
      if (isMissingArgTypes(funcOp)) {
        return mlir::WalkResult::skip();
      }
      forwardFuncOps.push_back(funcOp);
      return mlir::WalkResult::advance();
    });

    for (func::FuncOp funcOp : forwardFuncOps) {
      mlir::FunctionType oldFunctionType = funcOp.getFunctionType();

      // Classify the function arguments into activations and weights.
      //
      llvm::SmallDenseSet<unsigned> activationArgIndices, weightArgIndices;
      SmallVector<Attribute> originalArgTypes;
      SmallVector<Attribute> originalArgNames;
      classifyFuncArguments(funcOp, activationArgIndices, weightArgIndices,
                            originalArgTypes, originalArgNames);

      bool hasActivations = !activationArgIndices.empty();
      bool hasWeights = !weightArgIndices.empty();

      assert((hasActivations || hasWeights) &&
             "Expected at least one activation or weight argument!");

      // Collect the activation argument types.
      //
      SmallVector<mlir::Type> activationTypes;
      for (unsigned i = 0; i < oldFunctionType.getNumInputs(); ++i) {
        if (activationArgIndices.count(i)) {
          activationTypes.push_back(oldFunctionType.getInput(i));
        }
      }

      // Create and set the new function type.
      //
      SmallVector<mlir::Type> newInputType;
      if (hasActivations) {
        newInputType.push_back(mlir::TupleType::get(&ctx, activationTypes));
      }
      if (hasWeights) {
        newInputType.push_back(ttcore::DictType::get(&ctx));
      }

      mlir::FunctionType newFunctionType =
          oldFunctionType.clone(newInputType, oldFunctionType.getResults());

      rewriter.modifyOpInPlace(funcOp, [&funcOp, &newFunctionType]() {
        funcOp.setFunctionType(newFunctionType);
        funcOp->removeAttr(funcOp.getArgAttrsAttrName());
      });

      // Save original argument types and names as function-level attributes
      // before the function signature is modified.
      //
      assert(!originalArgTypes.empty() && "Expected argument types to be set!");
      funcOp->setAttr("ttcore.original_argument_types",
                      ArrayAttr::get(&ctx, originalArgTypes));
      if (!originalArgNames.empty()) {
        funcOp->setAttr("ttcore.original_argument_names",
                        ArrayAttr::get(&ctx, originalArgNames));
      }

      // Insert the new block arguments.
      //
      Block &entryBlock = funcOp.getBody().front();
      BlockArgument activationsTuple, weightsDict;
      unsigned newArgsCount = 0;
      if (hasActivations) {
        activationsTuple = entryBlock.insertArgument(
            newArgsCount, newInputType[newArgsCount], funcOp.getLoc());
        ++newArgsCount;
      }
      if (hasWeights) {
        weightsDict = entryBlock.insertArgument(
            newArgsCount, newInputType[newArgsCount], funcOp.getLoc());
        ++newArgsCount;
      }

      // Create ops to unpack original arguments from the tuple (activations)
      // or dict (weights), and replace all uses of the original arguments with
      // the unpacked values.
      //
      rewriter.setInsertionPointToStart(&entryBlock);
      size_t activationsTupleIdx = 0;
      size_t weightsNameIdx = 0;
      for (unsigned idx = 0; idx < oldFunctionType.getNumInputs(); ++idx) {
        Type originalType = oldFunctionType.getInputs()[idx];
        Value getElement;
        if (activationArgIndices.count(idx)) {
          getElement = rewriter
                           .create<ttcore::GetTupleElementOp>(
                               funcOp.getLoc(), originalType, activationsTuple,
                               activationsTupleIdx++)
                           ->getResult(0);
        } else {
          assert(weightArgIndices.count(idx) && "Expected weight argument!");
          getElement = rewriter
                           .create<ttcore::GetKeyValueOp>(
                               funcOp.getLoc(), originalType, weightsDict,
                               originalArgNames[weightsNameIdx++])
                           ->getResult(0);
        }

        rewriter.replaceAllUsesWith(entryBlock.getArgument(newArgsCount + idx),
                                    getElement);
      }

      // Erase original arguments.
      //
      entryBlock.eraseArguments(newArgsCount, oldFunctionType.getNumInputs());

      // Mark the function input as split.
      //
      ttmlir::utils::setSplitInput(funcOp);
    }
  }

  bool isMissingArgTypes(func::FuncOp funcOp) {
    return llvm::any_of(
        llvm::seq<unsigned>(0, funcOp.getFunctionType().getNumInputs()),
        [&](unsigned i) {
          return funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
                     i, ttcore::ArgumentTypeAttr::name) == nullptr;
        });
  }

  void
  classifyFuncArguments(func::FuncOp funcOp,
                        llvm::SmallDenseSet<unsigned> &activationArgIndices,
                        llvm::SmallDenseSet<unsigned> &weightArgIndices,
                        llvm::SmallVector<Attribute> &originalArgTypes,
                        llvm::SmallVector<Attribute> &originalArgNames) {
    mlir::FunctionType functionType = funcOp.getFunctionType();
    MLIRContext &ctx = getContext();

    unsigned unnamedWeightsCount = 0;
    for (unsigned i = 0; i < functionType.getNumInputs(); ++i) {
      if (auto typeAttr = funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
              i, ttcore::ArgumentTypeAttr::name)) {
        originalArgTypes.push_back(typeAttr);
        if (typeAttr.getValue() == ttcore::ArgumentType::Input) {
          activationArgIndices.insert(i);
        } else if (typeAttr.getValue() == ttcore::ArgumentType::Parameter ||
                   typeAttr.getValue() == ttcore::ArgumentType::Constant) {
          weightArgIndices.insert(i);
          if (auto nameAttr =
                  funcOp.getArgAttrOfType<StringAttr>(i, "ttir.name")) {
            originalArgNames.push_back(
                StringAttr::get(&ctx, nameAttr.getValue()));
          } else {
            originalArgNames.push_back(StringAttr::get(
                &ctx, "weight_" + std::to_string(unnamedWeightsCount++)));
          }
        }
      } else {
        llvm_unreachable("Pre-check should have skipped unannotated functions");
      }
    }
  }
};

// Tuplifies inputs and/or results of all the functions whose inputs and/or
// results are still flat (i.e. not packed into a tuple or dictionary).
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
