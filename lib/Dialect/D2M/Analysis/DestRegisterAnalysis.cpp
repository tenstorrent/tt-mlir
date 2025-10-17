// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DestRegisterAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::tt::d2m {

DestRegisterAnalysis::DestRegisterAnalysis(Operation *op) {
  op->walk([&](linalg::GenericOp genericOp) {
    DstRegisterInfo info;

    // Track destination register state.
    int nextAvailableIndex = 0;
    llvm::DenseMap<Operation *, int> opToDstIndex;
    llvm::DenseMap<Value, int> valueToDstIndex;

    // Phase 1: Collect all input values (from block args and non-compute ops).
    llvm::SmallVector<Value> inputValues;
    genericOp->walk([&](Operation *innerOp) {
      if (innerOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        for (Value operand : innerOp->getOperands()) {
          if (Operation *definingOp = operand.getDefiningOp()) {
            if (!definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
              if (valueToDstIndex.find(operand) == valueToDstIndex.end()) {
                inputValues.push_back(operand);
              }
            }
          } else {
            if (valueToDstIndex.find(operand) == valueToDstIndex.end()) {
              inputValues.push_back(operand);
            }
          }
        }
      }
    });

    // Phase 2: Load all inputs at the beginning.
    for (Value input : inputValues) {
      int inputIndex = nextAvailableIndex++;
      valueToDstIndex[input] = inputIndex;
      // Record input indices for DST allocation
      info.dstSliceIndices.push_back(inputIndex);
    }

    // Phase 3: Process compute ops.
    genericOp->walk([&](Operation *innerOp) {
      if (innerOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        int numOperands = innerOp->getNumOperands();

        // Check if this is an in-place operation.
        bool isInPlace = false;
        if (auto loadStoreInterface =
                llvm::dyn_cast<OperandLoadStoreRegisterOpInterface>(innerOp)) {
          isInPlace = loadStoreInterface.getDstRegInPlace();
        }

        // Get the input destination index (for in-place ops).
        int inputIndex = -1;
        if (isInPlace && numOperands > 0) {
          Value operand = innerOp->getOperand(0);
          if (Operation *definingOp = operand.getDefiningOp()) {
            if (definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
              inputIndex = opToDstIndex[definingOp];
            } else {
              inputIndex = valueToDstIndex[operand];
            }
          } else {
            inputIndex = valueToDstIndex[operand];
          }
        }

        // Allocate or reuse destination index.
        int outputIndex;
        if (isInPlace) {
          outputIndex = inputIndex;
        } else {
          outputIndex = nextAvailableIndex++;
        }
        opToDstIndex[innerOp] = outputIndex;
        // Store the intermediate DST slice indices for this compute op.
        info.dstSliceIndices.push_back(outputIndex);
      }
    });

    info.dstMaxUsage = std::max(info.dstMaxUsage, nextAvailableIndex);
    genericOpMap[genericOp] = info;
  });
}

} // namespace mlir::tt::d2m
