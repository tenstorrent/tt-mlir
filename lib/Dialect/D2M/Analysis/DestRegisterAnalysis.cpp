// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DestRegisterAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {

int findDstCapacity(linalg::GenericOp genericOp) {
  int dstCapacity = 8;
  return dstCapacity;
}

DestRegisterAnalysis::DestRegisterAnalysis(Operation *op) {
  // TODO: Implement analysis logic
  op->walk([&](linalg::GenericOp genericOp) {
    llvm::errs() << "Processing GenericOp: " << genericOp << "\n";

    DstRegisterInfo info;

    // Track destination register state
    int nextAvailableIndex = 0;
    llvm::DenseMap<Operation *, int>
        opToDstIndex; // Track which index each op uses
    llvm::DenseMap<Value, int>
        valueToDstIndex; // Track which index each value uses

    // Phase 1: Collect all input values (from block args and non-compute ops)
    llvm::SmallVector<Value> inputValues;
    llvm::errs() << "  Phase 1: Collecting input values...\n";

    genericOp->walk([&](Operation *innerOp) {
      if (innerOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        for (Value operand : innerOp->getOperands()) {
          if (Operation *definingOp = operand.getDefiningOp()) {
            // Only collect if it's not from a compute op
            if (!definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
              if (valueToDstIndex.find(operand) == valueToDstIndex.end()) {
                inputValues.push_back(operand);
              }
            }
          } else {
            // Block argument
            if (valueToDstIndex.find(operand) == valueToDstIndex.end()) {
              inputValues.push_back(operand);
            }
          }
        }
      }
    });

    // Phase 2: Load all inputs at the beginning
    llvm::errs() << "  Phase 2: Loading " << inputValues.size()
                 << " inputs...\n";
    for (Value input : inputValues) {
      int inputIndex = nextAvailableIndex++;
      valueToDstIndex[input] = inputIndex;
      llvm::errs() << "    Loaded input to dst index " << inputIndex << "\n";
    }

    // Phase 3: Process compute ops
    llvm::errs() << "  Phase 3: Processing compute ops...\n";
    genericOp->walk([&](Operation *innerOp) {
      // Check if this operation has the compute op trait
      if (innerOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        int numOperands = innerOp->getNumOperands();

        // Check if this is an in-place operation
        bool isInPlace = false;
        if (auto loadStoreInterface =
                llvm::dyn_cast<OperandLoadStoreRegisterOpInterface>(innerOp)) {
          isInPlace = loadStoreInterface.getDstRegInPlace();
        }

        // Get the input destination index (for in-place ops)
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
            // Block argument
            inputIndex = valueToDstIndex[operand];
          }
        }

        // Show which dst indices the inputs are using
        for (Value operand : innerOp->getOperands()) {
          if (Operation *definingOp = operand.getDefiningOp()) {
            if (definingOp->hasTrait<D2MGenericRegionComputeOpTrait>()) {
              // Input comes from previous compute op
              int idx = opToDstIndex[definingOp];
              llvm::errs() << "    Input from compute op "
                           << definingOp->getName() << " using dst index "
                           << idx << "\n";
            } else {
              // Input from non-compute op (already loaded)
              int idx = valueToDstIndex[operand];
              llvm::errs() << "    Input from pre-loaded value using dst index "
                           << idx << "\n";
            }
          } else {
            // Block argument (already loaded)
            int idx = valueToDstIndex[operand];
            llvm::errs()
                << "    Input from pre-loaded block arg using dst index " << idx
                << "\n";
          }
        }

        // Allocate or reuse destination index
        int outputIndex;
        if (isInPlace) {
          // Reuse the input's destination register
          outputIndex = inputIndex;
          llvm::errs() << "  In-place operation, reusing dst index "
                       << outputIndex << "\n";
        } else {
          // Allocate new destination register
          outputIndex = nextAvailableIndex++;
        }
        opToDstIndex[innerOp] = outputIndex;
        info.computeOpMap[innerOp] = outputIndex;

        std::string locStr;
        llvm::raw_string_ostream locStream(locStr);
        innerOp->getLoc().print(locStream);
        locStream.flush();

        if (numOperands == 1) {
          llvm::errs() << "  Found unary compute op: " << innerOp->getName()
                       << " at " << innerOp << " location: " << locStr
                       << " -> dst index " << outputIndex
                       << (isInPlace ? " (in-place)" : "") << "\n";
        } else if (numOperands == 2) {
          llvm::errs() << "  Found binary compute op: " << innerOp->getName()
                       << " at " << innerOp << " location: " << locStr
                       << " -> dst index " << outputIndex << "\n";
        } else {
          llvm::errs() << "  Found compute op with " << numOperands
                       << " operands: " << innerOp->getName() << " at "
                       << innerOp << " location: " << locStr << " -> dst index "
                       << outputIndex << "\n";
        }
      }
    });

    llvm::errs() << "  Peak dst register usage: " << nextAvailableIndex << "\n";

    info.dstMaxUsage = std::max(info.dstMaxUsage, nextAvailableIndex);
    llvm::errs() << "  Stored dstMaxUsage: " << info.dstMaxUsage << "\n";

    llvm::errs() << "  [DEBUG] Storing " << info.computeOpMap.size()
                 << " compute ops in analysis map for GenericOp at "
                 << genericOp << "\n";
    for (const auto &entry : info.computeOpMap) {
      std::string entryLocStr;
      llvm::raw_string_ostream entryLocStream(entryLocStr);
      entry.first->getLoc().print(entryLocStream);
      entryLocStream.flush();

      llvm::errs() << "    [DEBUG] Storing op " << entry.first->getName()
                   << " at " << entry.first << " location: " << entryLocStr
                   << " with index " << entry.second << "\n";
    }
    llvm::errs() << "\n";

    genericOpMap[genericOp] = info;
  });

  // Print summary of all GenericOps analyzed
  llvm::errs() << "========================================\n";
  llvm::errs() << "DestRegisterAnalysis Summary:\n";
  llvm::errs() << "Total GenericOps analyzed: " << genericOpMap.size() << "\n";
  for (const auto &entry : genericOpMap) {
    llvm::errs() << "  GenericOp at " << entry.first
                 << " -> dstMaxUsage = " << entry.second.dstMaxUsage << "\n";
    llvm::errs() << "    Compute ops (" << entry.second.computeOpMap.size()
                 << "):\n";
    for (const auto &opEntry : entry.second.computeOpMap) {
      std::string summaryLocStr;
      llvm::raw_string_ostream summaryLocStream(summaryLocStr);
      opEntry.first->getLoc().print(summaryLocStream);
      summaryLocStream.flush();

      llvm::errs() << "      " << opEntry.first->getName() << " at "
                   << opEntry.first << " location: " << summaryLocStr
                   << " -> dst index = " << opEntry.second << "\n";
    }
  }
  llvm::errs() << "========================================\n";
}

} // namespace mlir::tt::d2m
