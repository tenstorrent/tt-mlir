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
    int nextAvailableSlice = 0;
    llvm::DenseMap<Operation *, int> opToDstSlice;
    llvm::DenseMap<Value, int> valueToDstSlice;

    // Process compute ops.
    genericOp->walk([&](Operation *innerOp) {
      if (innerOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        // Get the operands that must be loaded from DST for this op
        // and whether this is an in-place operation.
        llvm::SmallVector<int64_t> dstOperandIndices;
        bool isInPlace = false;
        if (auto loadStoreInterface =
                llvm::dyn_cast<OperandLoadStoreRegisterOpInterface>(innerOp)) {
          dstOperandIndices =
              loadStoreInterface.getOperandsLoadFromDstRegister();
          isInPlace = loadStoreInterface.getDstRegInPlace();
        }

        // Allocate DST indices for operands that need to be loaded from DST.
        for (int64_t operandIdx : dstOperandIndices) {
          Value operand = innerOp->getOperand(operandIdx);
          if (valueToDstSlice.find(operand) == valueToDstSlice.end()) {
            if (Operation *definingOp = operand.getDefiningOp();
                !definingOp ||
                !definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
              // This is an external input, allocate a DST slice.
              valueToDstSlice[operand] = nextAvailableSlice++;
              info.dstSliceIndices.push_back(valueToDstSlice[operand]);
            }
          }
        }

        // Allocate or reuse destination slice for the output.
        int outputSlice = nextAvailableSlice;
        if (isInPlace) {
          assert(dstOperandIndices.size() == 1);
          Value operand = innerOp->getOperand(dstOperandIndices[0]);
          if (Operation *definingOp = operand.getDefiningOp();
              definingOp &&
              definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
            outputSlice = opToDstSlice[definingOp];
          } else {
            outputSlice = valueToDstSlice[operand];
          }
        } else {
          ++nextAvailableSlice;
        }
        opToDstSlice[innerOp] = outputSlice;
        // Store the intermediate DST slice indices for this compute op.
        info.dstSliceIndices.push_back(outputSlice);
      }
    });

    info.dstMaxUsage = nextAvailableSlice;
    dstRegisterInfoList.push_back(info);
  });
}

} // namespace mlir::tt::d2m
