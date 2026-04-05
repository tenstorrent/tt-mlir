// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
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
#define GEN_PASS_DEF_TTNNLOADINPUTTENSORS
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNTUPLIFYTENSORS
#define GEN_PASS_DEF_TTNNSPLITFORWARDFUNCARGSBYTYPE
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
    // Today we model DPS init operands (tensor flows into tied result).
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
/// from the ttcore.original_argument_types attribute. This is needed to load
/// the tensor of the proper index when loadInputTensorsFromDisk is enabled.
static void recoverOriginalArgIndices(func::FuncOp funcOp,
                                      SmallVector<size_t> &activationIndices,
                                      SmallVector<size_t> &weightIndices) {
  auto origTypesAttr =
      funcOp->getAttrOfType<ArrayAttr>("ttcore.original_argument_types");
  assert(origTypesAttr && "Expected ttcore.original_argument_types on function "
                          "that has split arguments by type!");
  for (unsigned i = 0; i < origTypesAttr.size(); ++i) {
    auto typeAttr = mlir::cast<ttcore::ArgumentTypeAttr>(origTypesAttr[i]);
    if (typeAttr.getValue() == ttcore::ArgumentType::Input) {
      activationIndices.push_back(i);
    } else if (typeAttr.getValue() == ttcore::ArgumentType::Parameter ||
               typeAttr.getValue() == ttcore::ArgumentType::Constant) {
      weightIndices.push_back(i);
    } else {
      llvm_unreachable("Unexpected ttcore::ArgumentType value");
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

      // Skip non-forward device functions and private forward device functions.
      // The latter is needed because TTNNRecoverStructure decomposes a forward
      // function's body into private forward device sub-functions (per model
      // substructure) that don't need input generators.
      //
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp) || funcOp.isPrivate()) {
        return mlir::WalkResult::advance();
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

      if (ttmlir::utils::hasSplitForwardFuncArgsByType(forwardFuncOp)) {
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
    // TTNNSplitForwardFuncArgsByType pass.
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

    mlir::Value device;
    if (layoutAttr.isDeviceBufferType()) {
      device = ttnn::utils::getOrInsertDevice(rewriter,
                                              rewriter.getInsertionBlock());
    }
    // Create a new tensor of ones.
    //
    ttnn::OnesOp onesOp =
        rewriter.create<ttnn::OnesOp>(loc, tensorType, device, shapeAttr);

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

// Splits the forward functions inputs by type into activations and
// weights. Only performed on forward functions which inputs have
// ttcore.argument_type attributes set, so that each input argument can
// be properly classified as an activation or a weight.
class TTNNSplitForwardFuncArgsByType
    : public impl::TTNNSplitForwardFuncArgsByTypeBase<
          TTNNSplitForwardFuncArgsByType> {
public:
  using impl::TTNNSplitForwardFuncArgsByTypeBase<
      TTNNSplitForwardFuncArgsByType>::TTNNSplitForwardFuncArgsByTypeBase;

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
      if (!containsArgTypes(funcOp)) {
        return mlir::WalkResult::skip();
      }
      forwardFuncOps.push_back(funcOp);
      return mlir::WalkResult::advance();
    });

    for (func::FuncOp funcOp : forwardFuncOps) {
      mlir::FunctionType oldFunctionType = funcOp.getFunctionType();

      // Classify the function arguments into activations and weights. Collect
      // the original arg types (preserved as a function-level attribute) and
      // the original arg names for activations and weights separately
      // (preserved as per-arg attributes on the new tuple/dict args).
      //
      llvm::SmallDenseSet<unsigned> activationArgIndices, weightArgIndices;
      classifyArgIndices(funcOp, activationArgIndices, weightArgIndices);
      SmallVector<Attribute> originalArgTypes = collectOriginalArgTypes(funcOp);
      SmallVector<Attribute> originalActivationNames =
          collectOriginalArgNamesForTargetIndices(
              funcOp, activationArgIndices, /*unnamedPrefix=*/"activation");
      SmallVector<Attribute> originalWeightNames =
          collectOriginalArgNamesForTargetIndices(funcOp, weightArgIndices,
                                                  /*unnamedPrefix=*/"weight");

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

      // Save original argument types as a function-level attribute.
      //
      assert(!originalArgTypes.empty() && "Expected argument types to be set!");
      funcOp->setAttr("ttcore.original_argument_types",
                      ArrayAttr::get(&ctx, originalArgTypes));

      // Insert the new block arguments and attach the original argument names
      // as per-arg attributes.
      //
      Block &entryBlock = funcOp.getBody().front();
      BlockArgument activationsTuple, weightsDict;
      unsigned newArgsCount = 0;
      if (hasActivations) {
        activationsTuple = entryBlock.insertArgument(
            newArgsCount, newInputType[newArgsCount], funcOp.getLoc());
        funcOp.setArgAttr(newArgsCount,
                          ttcore::g_originalActivationNamesAttrName,
                          ArrayAttr::get(&ctx, originalActivationNames));
        ++newArgsCount;
      }
      if (hasWeights) {
        weightsDict = entryBlock.insertArgument(
            newArgsCount, newInputType[newArgsCount], funcOp.getLoc());
        funcOp.setArgAttr(newArgsCount, ttcore::g_originalWeightNamesAttrName,
                          ArrayAttr::get(&ctx, originalWeightNames));
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
          // Hint the original activation name onto the unpack op so that the
          // EmitPy codegen can use it as the local variable name in the
          // generated Python instead of an anonymous `activations_N`.
          //
          auto getTupleElementOp = rewriter.create<ttcore::GetTupleElementOp>(
              funcOp.getLoc(), originalType, activationsTuple,
              activationsTupleIdx);
          getTupleElementOp->setAttr(
              "emitpy.name", originalActivationNames[activationsTupleIdx++]);
          getElement = getTupleElementOp->getResult(0);
        } else {
          assert(weightArgIndices.count(idx) && "Expected weight argument!");
          getElement = rewriter
                           .create<ttcore::GetKeyValueOp>(
                               funcOp.getLoc(), originalType, weightsDict,
                               originalWeightNames[weightsNameIdx++])
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
      ttmlir::utils::setSplitForwardFuncArgsByType(funcOp);
    }
  }

  bool containsArgTypes(func::FuncOp funcOp) {
    return llvm::all_of(
        llvm::seq<unsigned>(0, funcOp.getFunctionType().getNumInputs()),
        [&](unsigned i) {
          return funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
                     i, ttcore::ArgumentTypeAttr::name) != nullptr;
        });
  }

  // Classifies forward function argument indices into activation and weight
  // indices based on each arg's `ttcore.argument_type` attribute.
  //
  void classifyArgIndices(func::FuncOp funcOp,
                          llvm::SmallDenseSet<unsigned> &activationArgIndices,
                          llvm::SmallDenseSet<unsigned> &weightArgIndices) {
    for (unsigned i = 0; i < funcOp.getFunctionType().getNumInputs(); ++i) {
      auto typeAttr = funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
          i, ttcore::ArgumentTypeAttr::name);
      assert(typeAttr && "Pre-check should have skipped unannotated functions");
      if (typeAttr.getValue() == ttcore::ArgumentType::Input) {
        activationArgIndices.insert(i);
      } else if (typeAttr.getValue() == ttcore::ArgumentType::Parameter ||
                 typeAttr.getValue() == ttcore::ArgumentType::Constant) {
        weightArgIndices.insert(i);
      } else {
        llvm_unreachable("Unexpected ttcore::ArgumentType value");
      }
    }
  }

  // Collects per-arg `ttcore.argument_type` attributes in the original
  // argument order.
  //
  llvm::SmallVector<Attribute> collectOriginalArgTypes(func::FuncOp funcOp) {
    llvm::SmallVector<Attribute> originalArgTypes;
    for (unsigned i = 0; i < funcOp.getFunctionType().getNumInputs(); ++i) {
      auto typeAttr = funcOp.getArgAttrOfType<ttcore::ArgumentTypeAttr>(
          i, ttcore::ArgumentTypeAttr::name);
      assert(typeAttr && "Pre-check should have skipped unannotated functions");
      originalArgTypes.push_back(typeAttr);
    }
    return originalArgTypes;
  }

  // Collects names for the arguments at `targetArgIndices` in the original
  // argument order. Uses `ttir.name` arg attribute if present, otherwise
  // auto-generates names in the form of `<unnamedPrefix>_N`.
  //
  llvm::SmallVector<Attribute> collectOriginalArgNamesForTargetIndices(
      func::FuncOp funcOp,
      const llvm::SmallDenseSet<unsigned> &targetArgIndices,
      llvm::StringRef unnamedPrefix) {
    llvm::SmallVector<Attribute> originalArgNames;
    MLIRContext &ctx = getContext();
    unsigned unnamedCount = 0;
    for (unsigned i = 0; i < funcOp.getFunctionType().getNumInputs(); ++i) {
      if (!targetArgIndices.contains(i)) {
        continue;
      }
      if (auto nameAttr = funcOp.getArgAttrOfType<StringAttr>(i, "ttir.name")) {
        originalArgNames.push_back(StringAttr::get(&ctx, nameAttr.getValue()));
      } else {
        originalArgNames.push_back(StringAttr::get(
            &ctx, unnamedPrefix.str() + "_" + std::to_string(unnamedCount++)));
      }
    }
    return originalArgNames;
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

      // Tuplify when either the function has at least one input, or when
      // `tuplifyInputIfEmpty` is set and the function is a forward device
      // function (in which case an empty input tuple is forced). In both
      // cases, all inputs must be `RankedTensorType`.
      //
      const bool forceTuplifyForward =
          tuplifyInputIfEmpty && ttmlir::utils::isForwardDeviceFunc(funcOp);
      const bool hasInputs = !functionType.getInputs().empty();
      const bool allInputsAreTensors =
          llvm::all_of(functionType.getInputs(),
                       [](Type t) { return mlir::isa<RankedTensorType>(t); });
      if ((forceTuplifyForward || hasInputs) && allInputsAreTensors) {
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

    // The pipeline runs TTNNTuplifyTensors with `tuplifyInputIfEmpty=true`
    // before this pass, so the forward function is guaranteed to have at
    // least one argument (the input tuple) at this point.
    //
    if (targetFuncOp.getNumArguments() == 0) {
      targetFuncOp.emitOpError(
          "expected forward function to have at least one (tuple) input; run "
          "`-ttnn-tuplify-tensors=tuplify-input-if-empty=true` before this "
          "pass");
      signalPassFailure();
      return;
    }

    // Add a device argument to the function and replace all GetDeviceOp uses
    // with it.
    //
    if (failed(injectDeviceArg(rewriter, targetFuncOp))) {
      signalPassFailure();
      return;
    }

    // The forward function must have exactly two
    // arguments: the input tuple from tuplification and the device argument
    // just injected.
    //
    if (targetFuncOp.getNumArguments() != 2) {
      targetFuncOp.emitOpError(
          "expected forward function to have exactly (tuple, device) "
          "arguments after device injection; got ")
          << targetFuncOp.getNumArguments() << " arguments";
      signalPassFailure();
      return;
    }

    // Set `emitpy.name` attributes for the forward function's arguments.
    //
    targetFuncOp.setArgAttr(0, "emitpy.name", rewriter.getStringAttr("input"));
    targetFuncOp.setArgAttr(1, "emitpy.name", rewriter.getStringAttr("device"));

    // Prepare the forward function input tensors to be in the right form
    // (host/device, layout, dtype).
    //
    prepareForwardFuncInputTensors(rewriter, targetFuncOp);
  }

private:
  // Adds a trailing device argument to the function's signature and
  // rewrites every GetDeviceOp inside the function to use it. Fails if
  // the argument could not be inserted.
  //
  LogicalResult injectDeviceArg(IRRewriter &rewriter, func::FuncOp funcOp) {
    DeviceType deviceType = DeviceType::get(rewriter.getContext());
    unsigned deviceArgIndex = funcOp.getNumArguments();
    if (failed(funcOp.insertArgument(deviceArgIndex, deviceType,
                                     /*argAttrs=*/DictionaryAttr{},
                                     funcOp.getLoc()))) {
      return funcOp.emitOpError(
          "failed to inject device argument into the function's signature!");
    }

    // Replace all GetDeviceOp operations with the new device argument.
    //
    BlockArgument deviceArg = funcOp.getArgument(deviceArgIndex);
    SmallVector<ttnn::GetDeviceOp> getDeviceOps;
    funcOp.walk([&](ttnn::GetDeviceOp op) { getDeviceOps.push_back(op); });
    for (auto getDeviceOp : getDeviceOps) {
      rewriter.replaceOp(getDeviceOp, deviceArg);
    }
    return success();
  }

  // Re-type every forward function input tensor to the appropriate form
  // (host/device, layout, dtype).
  //
  void prepareForwardFuncInputTensors(IRRewriter &rewriter,
                                      func::FuncOp forwardFuncOp) {
    MLIRContext *ctx = rewriter.getContext();
    Value inputTuple = forwardFuncOp.getArgument(0);
    Value deviceArg = forwardFuncOp.getArgument(1);
    auto inputTupleType = mlir::cast<mlir::TupleType>(inputTuple.getType());

    // The tt-xla/codegen entry-point contract is that the forward function
    // always receives each tensor on host in row-major form. However,
    // TTNNLayout pass rewrites the FuncOp inputs to device layout (with some
    // minor exceptions, e.g. Conv2d weights), and relies on the function caller
    // to prepare input tensors in the right form. For that reason, this pass
    // uses the proper host counterparts of the device-bound input tensors and
    // applies ToLayoutOp and ToDeviceOp to them.
    llvm::DenseMap<int64_t, RankedTensorType> hostTypesByIndex;
    for (size_t idx = 0; idx < inputTupleType.size(); ++idx) {
      auto tensorType =
          mlir::cast<RankedTensorType>(inputTupleType.getType(idx));
      auto targetLayout = mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());

      // System-memory inputs are already declared as host row-major.
      //
      if (!targetLayout.isDeviceBufferType()) {
        continue;
      }

      // For each device-bound tensor type, create the host row-major tensor
      // type that preserves the target shape and element type.
      //
      auto hostLayoutAttr = TTNNLayoutAttr::Builder(tensorType)
                                .setBufferType(BufferType::SystemMemory)
                                .setLayout(Layout::RowMajor)
                                .build();
      hostTypesByIndex[idx] = RankedTensorType::get(
          tensorType.getShape(), tensorType.getElementType(), hostLayoutAttr);
    }

    if (hostTypesByIndex.empty()) {
      return;
    }

    // Patch GetTupleElementOp results to a proper type.
    //
    SmallVector<ttcore::GetTupleElementOp> getElemOps;
    forwardFuncOp.walk([&](ttcore::GetTupleElementOp getElemOp) {
      if (getElemOp.getOperand() == inputTuple) {
        getElemOps.push_back(getElemOp);
      }
    });

    for (auto getElemOp : getElemOps) {
      int64_t idx = getElemOp.getIndex();
      auto it = hostTypesByIndex.find(idx);
      if (it == hostTypesByIndex.end()) {
        continue;
      }

      RankedTensorType hostTensorType = it->second;
      auto deviceTensorType =
          mlir::cast<RankedTensorType>(inputTupleType.getType(idx));
      auto targetLayout =
          mlir::cast<TTNNLayoutAttr>(deviceTensorType.getEncoding());

      auto oldTupleElemResult = getElemOp.getResult();
      oldTupleElemResult.setType(hostTensorType);

      rewriter.setInsertionPointAfter(getElemOp);
      // Build the intermediate host-staged type that
      // carries the target layout and element type.
      //
      auto hostStagedLayoutAttr = TTNNLayoutAttr::Builder(deviceTensorType)
                                      .setBufferType(BufferType::SystemMemory)
                                      .setLayout(targetLayout.getLayout())
                                      .build();
      auto hostStagedTensorType = RankedTensorType::get(
          deviceTensorType.getShape(), deviceTensorType.getElementType(),
          hostStagedLayoutAttr);

      // On-host layout/dtype conversion. If this is an
      // identity conversion the folder collapses the op to its input.
      //
      auto toLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
          getElemOp.getLoc(), hostStagedTensorType, oldTupleElemResult);

      // Host-to-device transfer. The target memory config is conveyed via the
      // result tensor type's `TTNNLayoutAttr` encoding.
      //
      auto toDeviceOp = rewriter.create<ttnn::ToDeviceOp>(
          getElemOp.getLoc(), deviceTensorType, toLayoutOp.getResult(),
          deviceArg);

      // Replace the original tuple element with the device tensor produced by
      // ToDeviceOp. The only use of the original tuple element preserved is
      // the one inside the ToLayoutOp itself.
      //
      rewriter.replaceAllUsesExcept(oldTupleElemResult, toDeviceOp.getResult(),
                                    toLayoutOp);
    }

    // Patch the tuple type and the forward function's input type so the IR
    // is internally consistent with the retyped tuple element results.
    //
    SmallVector<Type> newTupleElementTypes(inputTupleType.getTypes().begin(),
                                           inputTupleType.getTypes().end());
    for (auto &[idx, hostType] : hostTypesByIndex) {
      newTupleElementTypes[idx] = hostType;
    }
    auto newTupleType = mlir::TupleType::get(ctx, newTupleElementTypes);
    inputTuple.setType(newTupleType);

    FunctionType forwardFuncType = forwardFuncOp.getFunctionType();
    SmallVector<Type> newInputTypes(forwardFuncType.getInputs().begin(),
                                    forwardFuncType.getInputs().end());
    newInputTypes[0] = newTupleType;
    forwardFuncOp.setType(
        FunctionType::get(ctx, newInputTypes, forwardFuncType.getResults()));
  }
};

} // namespace mlir::tt::ttnn
