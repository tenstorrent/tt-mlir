// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MPROMOTE1DTO2D
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Check if a tensor type is 1D (has exactly one dimension).
static bool is1DTensor(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    return tensorType.getRank() == 1;
  }
  return false;
}

/// Create a 2D tensor type from a 1D tensor type by adding a leading dimension
/// of 1.
static RankedTensorType promote1DTo2D(RankedTensorType type1D) {
  assert(type1D.getRank() == 1 && "Expected 1D tensor");
  SmallVector<int64_t> shape2D = {1, type1D.getShape()[0]};
  return RankedTensorType::get(shape2D, type1D.getElementType(),
                               type1D.getEncoding());
}

/// Promote a type from 1D to 2D if it's a 1D tensor, otherwise return as-is.
static Type promoteTypeIfNeeded(Type type) {
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type);
      tensorType && is1DTensor(tensorType)) {
    return promote1DTo2D(tensorType);
  }
  return type;
}

/// Create a reshape op from 1D [N] to 2D [1, N].
static Value createReshape1Dto2D(OpBuilder &builder, Location loc,
                                 Value input1D) {
  auto inputType = mlir::cast<RankedTensorType>(input1D.getType());
  auto outputType = promote1DTo2D(inputType);

  SmallVector<int32_t> shape2D = {
      1, static_cast<int32_t>(inputType.getShape()[0])};

  return builder
      .create<ttir::ReshapeOp>(loc, outputType, input1D,
                               builder.getI32ArrayAttr(shape2D))
      .getResult();
}

/// Create a reshape op from 2D [1, N] to 1D [N].
static Value createReshape2Dto1D(OpBuilder &builder, Location loc,
                                 Value input2D, int64_t size) {
  auto inputType = mlir::cast<RankedTensorType>(input2D.getType());
  auto outputType = RankedTensorType::get({size}, inputType.getElementType());

  SmallVector<int32_t> shape1D = {static_cast<int32_t>(size)};

  return builder
      .create<ttir::ReshapeOp>(loc, outputType, input2D,
                               builder.getI32ArrayAttr(shape1D))
      .getResult();
}

class D2MPromote1DTo2D : public impl::D2MPromote1DTo2DBase<D2MPromote1DTo2D> {
public:
  using D2MPromote1DTo2DBase::D2MPromote1DTo2DBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      // Skip external functions
      if (funcOp.isExternal()) {
        return;
      }

      // Collect indices of 1D tensor inputs
      SmallVector<unsigned> input1DIndices;
      for (auto [idx, inputType] : llvm::enumerate(funcOp.getArgumentTypes())) {
        if (is1DTensor(inputType)) {
          input1DIndices.push_back(idx);
        }
      }

      // Check if any results are 1D tensors
      bool has1DOutputs = llvm::any_of(funcOp.getResultTypes(), is1DTensor);

      if (input1DIndices.empty() && !has1DOutputs) {
        return;
      }

      OpBuilder funcBuilder(funcOp.getContext());

      // Step 1: Insert reshape ops for 1D inputs at function entry
      Block &entryBlock = funcOp.getBody().front();
      funcBuilder.setInsertionPointToStart(&entryBlock);

      // Map from original 1D values to their 2D reshaped versions
      IRMapping valueMapping;

      // Track all values that have been promoted to 2D (for downstream
      // propagation)
      DenseSet<Value> promotedValues;

      for (unsigned idx : input1DIndices) {
        BlockArgument arg = entryBlock.getArgument(idx);

        // Check if all uses of this 1D argument are reshape ops.
        // If so, skip inserting a promotion reshape since the existing
        // reshapes already handle the dimension change.
        bool allUsesAreReshapes =
            llvm::all_of(arg.getUses(), [](OpOperand &use) {
              return mlir::isa<ttir::ReshapeOp>(use.getOwner());
            });

        if (allUsesAreReshapes) {
          // Skip creating promotion reshape - existing reshapes handle it
          continue;
        }

        Value reshaped2D = createReshape1Dto2D(funcBuilder, arg.getLoc(), arg);
        valueMapping.map(arg, reshaped2D);
        promotedValues.insert(reshaped2D);
      }

      // Track the reshape ops we just created so we don't process them
      DenseSet<Operation *> createdReshapes;
      for (auto [arg, reshaped] : valueMapping.getValueMap()) {
        if (auto *definingOp = reshaped.getDefiningOp()) {
          createdReshapes.insert(definingOp);
        }
      }

      // Step 2: Update all ops in the function to use 2D types
      // We need to update ops that consume 1D values to use the 2D versions
      SmallVector<Operation *> opsToProcess;
      for (Operation &op : entryBlock.getOperations()) {
        // Skip the reshape ops we just created and the terminator
        if (createdReshapes.contains(&op) ||
            op.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        opsToProcess.push_back(&op);
      }

      // Collect reshape ops that become redundant after operand remapping
      SmallVector<ttir::ReshapeOp> redundantReshapes;

      for (Operation *op : opsToProcess) {
        // Check if any operand needs to be remapped or has been promoted
        bool needsUpdate = false;
        for (Value operand : op->getOperands()) {
          if (valueMapping.contains(operand) ||
              promotedValues.contains(operand)) {
            needsUpdate = true;
            break;
          }
        }

        if (!needsUpdate) {
          continue;
        }

        // Remap operands
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          Value operand = op->getOperand(i);
          if (valueMapping.contains(operand)) {
            op->setOperand(i, valueMapping.lookup(operand));
          }
        }

        // For reshape ops, check if they become redundant after remapping
        // (i.e., input and output shapes are now the same)
        if (auto reshapeOp = mlir::dyn_cast<ttir::ReshapeOp>(op)) {
          auto inputType =
              mlir::dyn_cast<RankedTensorType>(reshapeOp.getInput().getType());
          auto outputType =
              mlir::dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
          if (inputType && outputType &&
              inputType.getShape() == outputType.getShape()) {
            // This reshape is now a no-op, mark it for removal
            redundantReshapes.push_back(reshapeOp);
          }
          // Don't update result types for reshape ops - they have explicit
          // output shapes
          continue;
        }

        // Update result types: if input was promoted 1D->2D, output should be
        // too
        for (unsigned i = 0; i < op->getNumResults(); ++i) {
          Value result = op->getResult(i);
          Type newType = promoteTypeIfNeeded(result.getType());
          if (newType != result.getType()) {
            result.setType(newType);
            // Track this promoted result so downstream ops will be updated
            promotedValues.insert(result);
          }
        }
      }

      // Remove redundant reshapes
      for (ttir::ReshapeOp reshapeOp : redundantReshapes) {
        reshapeOp.getResult().replaceAllUsesWith(reshapeOp.getInput());
        reshapeOp.erase();
      }

      // Step 3: Handle return - insert reshape ops to convert 2D back to 1D
      auto *terminator = entryBlock.getTerminator();
      auto returnOp = mlir::dyn_cast<func::ReturnOp>(terminator);
      if (!returnOp) {
        return;
      }

      funcBuilder.setInsertionPoint(returnOp);

      SmallVector<Value> newReturnOperands;
      bool changed = false;

      for (auto [idx, operand] : llvm::enumerate(returnOp.getOperands())) {
        auto operandType = mlir::dyn_cast<RankedTensorType>(operand.getType());
        auto expectedType =
            mlir::dyn_cast<RankedTensorType>(funcOp.getResultTypes()[idx]);

        // If the return type should be 1D but the operand is 2D [1, N],
        // insert reshape
        if (operandType && expectedType && expectedType.getRank() == 1 &&
            operandType.getRank() == 2 && operandType.getShape()[0] == 1) {
          Value reshaped1D =
              createReshape2Dto1D(funcBuilder, returnOp.getLoc(), operand,
                                  expectedType.getShape()[0]);
          newReturnOperands.push_back(reshaped1D);
          changed = true;
        } else {
          newReturnOperands.push_back(operand);
        }
      }

      if (changed) {
        funcBuilder.create<func::ReturnOp>(returnOp.getLoc(),
                                           newReturnOperands);
        returnOp.erase();
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
