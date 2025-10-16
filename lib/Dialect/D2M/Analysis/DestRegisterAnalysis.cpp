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

    // Phase 2: Process compute ops.
    genericOp->walk([&](Operation *innerOp) {
      if (innerOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        // Get the operands that must be loaded from DST for this op
        llvm::SmallVector<int64_t> dstOperandIndices;
        if (auto loadStoreInterface =
                llvm::dyn_cast<OperandLoadStoreRegisterOpInterface>(innerOp)) {
          dstOperandIndices =
              loadStoreInterface.getOperandsLoadFromDstRegister();
        }

        // Allocate DST indices for operands that need to be loaded from DST
        for (int64_t operandIdx : dstOperandIndices) {
          if (operandIdx < static_cast<int64_t>(innerOp->getNumOperands())) {
            Value operand = innerOp->getOperand(operandIdx);
            if (valueToDstIndex.find(operand) == valueToDstIndex.end()) {
              Operation *definingOp = operand.getDefiningOp();
              if (!definingOp ||
                  !definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
                // This is an external input, allocate a DST index.
                valueToDstIndex[operand] = nextAvailableIndex++;
                info.dstSliceIndices.push_back(valueToDstIndex[operand]);
                llvm::errs() << "    Allocated external input DST index: "
                             << valueToDstIndex[operand] << "\n";
              }
            }
          }
        }

        // Check if this is an in-place operation.
        bool isInPlace = false;
        if (auto loadStoreInterface =
                llvm::dyn_cast<OperandLoadStoreRegisterOpInterface>(innerOp)) {
          isInPlace = loadStoreInterface.getDstRegInPlace();
        }

        // Allocate or reuse destination index for the output.
        int outputIndex;
        if (isInPlace && innerOp->getNumOperands() > 0) {
          Value operand = innerOp->getOperand(0);
          if (Operation *definingOp = operand.getDefiningOp()) {
            if (definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
              outputIndex = opToDstIndex[definingOp];
            } else {
              outputIndex = valueToDstIndex[operand];
            }
          } else {
            outputIndex = valueToDstIndex[operand];
          }
        } else {
          outputIndex = nextAvailableIndex++;
        }
        opToDstIndex[innerOp] = outputIndex;
        // Store the intermediate DST slice indices for this compute op.
        info.dstSliceIndices.push_back(outputIndex);
      }
    });

    info.dstMaxUsage = std::max(info.dstMaxUsage, nextAvailableIndex);
    dstRegisterInfoList.push_back(info);
  });
}

} // namespace mlir::tt::d2m
