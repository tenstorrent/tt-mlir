// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNCREATEINPUTGENERATORS
#define GEN_PASS_DEF_TTNNMODIFYSIGNATURESFORDYLIB
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

  void runOnOperation() final {
    llvm::outs() << "s TTNNDeallocate::rOO\n";
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "found func that didn't have one block!");
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      // Handle non DPS ops which do not store function result and are used to
      // allocate tensors. DPS ops are handled via ttnn::EmptyOp.
      //
      func->walk([&](Operation *op) {
        if (isa<DestinationStyleOpInterface>(op)) {
          return;
        }

        // Skip ops which do not have results.
        //
        if (op->getNumResults() == 0) {
          return;
        }

        // Iterate over all results of the op.
        //
        for (OpResult result : op->getResults()) {
          // Check if result is ranked tensor type.
          //
          if (!isa<RankedTensorType>(result.getType())) {
            continue;
          }

          RankedTensorType resultTy =
              mlir::cast<RankedTensorType>(result.getType());
          assert(resultTy.getEncoding());

          Operation *lastOp = getLastValueUsageOp(livenessInfo, result);

          if (isa<func::ReturnOp>(lastOp)) {
            continue;
          }

          rewriter.setInsertionPointAfter(lastOp);
          rewriter.create<DeallocateOp>(lastOp->getLoc(), result);
        }
      });
    });
    llvm::outs() << "e TTNNDeallocate::rOO\n";
  }
};

class TTNNCreateInputGenerators
    : public impl::TTNNCreateInputGeneratorsBase<TTNNCreateInputGenerators> {

public:
  using impl::TTNNCreateInputGeneratorsBase<
      TTNNCreateInputGenerators>::TTNNCreateInputGeneratorsBase;

  void runOnOperation() final {
    llvm::outs() << "s TTNNCreateInputGenerators::rOO\n";

    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region
    assert(module->getRegions().size() == 1);
    assert(module->getRegion(0).getBlocks().size() == 1);

    // Get the first block of the region at index 0
    //
    Block *firstBlock = module.getBody(0);

    // Find all the func.func ops in the module that are "forward" functions
    //
    SmallVector<func::FuncOp, 1> forwardFuncOps;
    for (mlir::Operation &op : firstBlock->getOperations()) {
      if (mlir::func::FuncOp funcOp = dyn_cast<func::FuncOp>(op)) {

        // Skip functions that are called elsewhere in the IR
        //
        // This will skip utility functions that are used by other functions,
        // only top-level "forward" functions should be considered
        //
        if (!funcOp->getUses().empty()) {
          continue;
        }

        forwardFuncOps.push_back(funcOp);
      }
    }

    // Iterate over all the func ops and add input tensor generator functions
    //
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      // Get all the input tensors for the current forward func
      //
      llvm::SmallVector<mlir::RankedTensorType, 2> inputTensors;
      for (auto input : forwardFuncOp.getFunctionType().getInputs()) {
        inputTensors.push_back(llvm::cast<mlir::RankedTensorType>(input));
      }

      // Create a new function that will generate the input tensors
      //
      std::string inputGenFuncName =
          "createInputsFor_" + forwardFuncOp.getName().str();

      // Create function type
      //
      mlir::TypeRange returnTypeRange =
          mlir::TypeRange(forwardFuncOp.getFunctionType().getInputs());
      FunctionType functionType =
          mlir::FunctionType::get(&getContext(), {}, returnTypeRange);

      // Set insertion point to end of first block
      //
      rewriter.setInsertionPointToEnd(firstBlock);

      // Create the function
      //
      func::FuncOp inputGenFuncOp = rewriter.create<mlir::func::FuncOp>(
          module->getLoc(), inputGenFuncName, functionType);

      // Add a Block to func op and set insertion point to the beginning of the
      // Block
      //
      ::mlir::Block *currFnBlock = inputGenFuncOp.addEntryBlock();
      rewriter.setInsertionPointToStart(currFnBlock);

      // Create the input tensors
      //
      SmallVector<Value, 2> generatedTensors;
      for (Type tensorType : returnTypeRange) {
        assert(llvm::isa<mlir::RankedTensorType>(tensorType));

        RankedTensorType tensor =
            llvm::cast<mlir::RankedTensorType>(tensorType);

        // Get the layout attribute
        //
        ttnn::TTNNLayoutAttr layoutAttr =
            mlir::cast<ttnn::TTNNLayoutAttr>(tensor.getEncoding());

        // Get the shape of the tensor, tensor layout, and data type
        //
        ShapeAttr shapeAttr =
            ttnn::ShapeAttr::get(&getContext(), tensor.getShape());
        ttnn::LayoutAttr tensorLayoutAttr =
            ttnn::LayoutAttr::get(&getContext(), layoutAttr.getLayout());
        DataTypeAttr dTypeAttr =
            DataTypeAttr::get(&getContext(), layoutAttr.getDataType());

        // Create a new tensor
        //
        mlir::Value tensorValue = rewriter.create<ttnn::OnesOp>(
            forwardFuncOp->getLoc(), tensorType, shapeAttr, dTypeAttr,
            tensorLayoutAttr, nullptr, nullptr);

        generatedTensors.push_back(tensorValue);
      }

      // Return the generated tensors
      //
      rewriter.create<func::ReturnOp>(forwardFuncOp->getLoc(),
                                      generatedTensors);
    }

    // Create a main function to call input generators and forward funcs
    //
    {
      // Create a new function that will generate the input tensors
      //
      std::string mainFuncName = "main";

      // Create function type
      //
      mlir::TypeRange returnTypeRange = mlir::TypeRange(rewriter.getI32Type());
      FunctionType functionType =
          mlir::FunctionType::get(&getContext(), {}, returnTypeRange);

      // Set insertion point to end of first block
      //
      rewriter.setInsertionPointToEnd(firstBlock);

      // Create the function
      //
      func::FuncOp mainFuncOp = rewriter.create<mlir::func::FuncOp>(
          module->getLoc(), mainFuncName, functionType);

      ::mlir::Block *currFnBlock = mainFuncOp.addEntryBlock();

      // Set insertion point to the beginning of the block
      //
      rewriter.setInsertionPointToStart(currFnBlock);

      // Call the input generators
      //
      for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
        std::string inputGenFuncName =
            "createInputsFor_" + forwardFuncOp.getName().str();

        // Get the input generator function
        //
        mlir::func::FuncOp inputGenFuncOp =
            module.lookupSymbol<mlir::func::FuncOp>(inputGenFuncName);

        // Call the input generator function
        //
        func::CallOp createdTensors = rewriter.create<mlir::func::CallOp>(
            forwardFuncOp->getLoc(), inputGenFuncOp, ValueRange());

        rewriter.create<mlir::func::CallOp>(forwardFuncOp->getLoc(),
                                            forwardFuncOp,
                                            createdTensors->getResults());
      }

      // Return 0
      //
      // func::ReturnOp requires a Value to be returned, which means that an SSA
      // needs to be returned, hence create a constant 0 via arith::ConstantOp
      //
      Value constantZero = rewriter.create<arith::ConstantOp>(
          rewriter.getUnknownLoc(), rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(0));
      rewriter.create<func::ReturnOp>(mainFuncOp->getLoc(), constantZero);
    }

    llvm::outs() << "e TTNNCreateInputGenerators::rOO\n";
  }
};

class TTNNModifySignaturesForDylib
    : public impl::TTNNModifySignaturesForDylibBase<
          TTNNModifySignaturesForDylib> {

public:
  using impl::TTNNModifySignaturesForDylibBase<
      TTNNModifySignaturesForDylib>::TTNNModifySignaturesForDylibBase;

  void runOnOperation() final {
    llvm::outs() << "s TTNNModifySignaturesForDylib::rOO\n";

    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region
    assert(module->getRegions().size() == 1);
    assert(module->getRegion(0).getBlocks().size() == 1);

    // Get the first block of the region at index 0
    //
    Block *firstBlock = module.getBody(0);

    // Find all the func.func ops in the module that are "forward" functions
    //
    SmallVector<func::FuncOp, 1> forwardFuncOps;
    for (mlir::Operation &op : firstBlock->getOperations()) {
      if (mlir::func::FuncOp funcOp = dyn_cast<func::FuncOp>(op)) {

        // Skip functions that are called elsewhere in the IR
        //
        // This will skip utility functions that are used by other functions,
        // only top-level "forward" functions should be considered
        //
        if (!funcOp->getUses().empty()) {
          continue;
        }

        forwardFuncOps.push_back(funcOp);
      }
    }

    // Iterate over all the func ops and modify the signatures
    //
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      // Replace the signature of the forward function so that all the tensor
      // arguments are packed into a single tuple, and device type is appended
      //
      mlir::FunctionType originalFuncType = forwardFuncOp.getFunctionType();
      assert(
          std::all_of(originalFuncType.getInputs().begin(),
                      originalFuncType.getInputs().end(),
                      [](Type t) { return mlir::isa<RankedTensorType>(t); }) &&
          "Expected all inputs must be of type RankedTensorType");

      // Find device op
      //
      ttnn::GetDeviceOp getDeviceOp = nullptr;
      forwardFuncOp.walk([&](ttnn::GetDeviceOp currGDOp) {
        assert(!getDeviceOp &&
               "Only one device expected, but found more than one!");
        getDeviceOp = currGDOp;
      });

      // Create Type objects for modified function signature:
      // 1. tuplifiedInputTensors: TupleType of all input tensors
      // 2. deviceType: DeviceType
      // 3. tuplifiedOutputTensors: TupleType of all output tensors
      //
      mlir::TupleType tuplifiedInputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getInputs());
      tt::DeviceType deviceType = getDeviceOp.getResult().getType();
      mlir::TupleType tuplifiedOutputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getResults());

      // Create modified function type (signature) that takes the input tuple
      // and device as operands, and returns the output tuple
      //
      FunctionType modifiedFuncType = originalFuncType.clone(
          {tuplifiedInputTensors, deviceType}, tuplifiedOutputTensors);
      rewriter.modifyOpInPlace(forwardFuncOp,
                               [&forwardFuncOp, &modifiedFuncType]() {
                                 forwardFuncOp.setType(modifiedFuncType);
                               });

      // First block of the function (often referred to as "entry block") needs
      // its arguments updated as well - the args need to match the containing
      // func's arguments; this is implemented here by first inserting the tuple
      // as the first argument of the block, inserting GetTupleElementOp ops to
      // start of the block in order to unpack tuple elements, and then
      // replacing all uses of the original block arguments with the
      // GetTupleElementOp results - after this it's finally safe to remove
      // original block arguments as they have no live uses anymore
      //
      // Additionally, the Device is added as the second argument, and the
      // GetDeviceOp that creates Device is removed
      //
      // The return statement is modified to return a tuple
      //
      Block &entryBlock = forwardFuncOp.getBlocks().front();
      entryBlock.insertArgument(/*index=*/0u, tuplifiedInputTensors,
                                forwardFuncOp.getLoc());
      entryBlock.insertArgument(/*index=*/1u, deviceType,
                                forwardFuncOp.getLoc());

      rewriter.setInsertionPointToStart(&entryBlock);
      for (size_t idx = 0; idx < originalFuncType.getInputs().size(); idx++) {
        ::mlir::tt::GetTupleElementOp getTupleElementOp =
            rewriter.create<mlir::tt::GetTupleElementOp>(
                forwardFuncOp.getLoc(), forwardFuncOp.getArgument(0), idx);

        rewriter.replaceAllUsesWith(entryBlock.getArgument(2 + idx),
                                    getTupleElementOp);
      }

      // Erase original arguments
      //
      entryBlock.eraseArguments(2, originalFuncType.getInputs().size());

      // Remove device usage and remove the original GetDeviceOp
      //
      rewriter.replaceAllUsesWith(getDeviceOp.getResult(),
                                  entryBlock.getArgument(1));
      rewriter.eraseOp(getDeviceOp);

      // Find return statement and replace with tuple
      //
      forwardFuncOp->walk([&](mlir::func::ReturnOp returnOp) {
        rewriter.setInsertionPointAfter(returnOp);
        TupleOp tupleOp = rewriter.create<mlir::tt::TupleOp>(
            returnOp.getLoc(), returnOp.getOperands());

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp,
                                                          tupleOp.getResult());
      });
    }
    llvm::outs() << "e TTNNModifySignaturesForDylib::rOO\n";
  }
};

} // namespace mlir::tt::ttnn
