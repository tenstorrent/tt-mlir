// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNWorkaroundInterface.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h"
#include "ttmlir/Utils.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn::wa {
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNWorkaroundInterface.cpp.inc"

// Verifier function for TTNN Workaround Interface.
mlir::LogicalResult verifyTTNNWorkaroundInterface(mlir::Operation *op) {

  // Verify that the number of input and output operand workarounds is the same
  // as the number of tensor operands and tensor results.
  size_t cntTensorInputs =
      llvm::count_if(op->getOperands(), ttmlir::utils::isRankedTensor);
  size_t cntTensorResults =
      llvm::count_if(op->getResults(), ttmlir::utils::isRankedTensor);

  TTNNWorkaroundInterface workaroundOp =
      mlir::cast<TTNNWorkaroundInterface>(op);

  TTNNOperandsWorkarounds workarounds = workaroundOp.getOperandsWorkarounds();

  if (workarounds.getInputOperandWorkarounds().size() != cntTensorInputs) {
    return op->emitOpError()
           << "Number of input operand workarounds "
           << workarounds.getInputOperandWorkarounds().size()
           << " does not match the number of tensor inputs " << cntTensorInputs;
  }

  if (workarounds.getOutputOperandWorkarounds().size() != cntTensorResults) {
    return op->emitOpError() << "Number of output operand workarounds "
                             << " does not match the number of tensor results "
                             << cntTensorResults;
  }

  // For DPS ops, verify that the output workaround is the same as the input
  // init workaround.
  if (mlir::isa<DestinationStyleOpInterface>(op)) {
    DestinationStyleOpInterface dpsOp =
        mlir::cast<DestinationStyleOpInterface>(op);

    // Go through all the operands and for each DPS init operand, check if the
    // output workaround is the same.
    int dpsDestinationIndex = 0;
    for (int64_t i = 0; i < op->getNumOperands(); i++) {
      OpOperand &operand = op->getOpOperand(i);

      // Skip if the output result isn't a tensor.
      if (!ttmlir::utils::isRankedTensor(operand.get())) {
        dpsDestinationIndex++;
        continue;
      }

      // Skip if the operand is not a DPS init.
      if (!dpsOp.isDpsInit(&operand)) {
        dpsDestinationIndex++;
        continue;
      }

      // Get the tied output result for the DPS destination operand.
      OpResult tiedOutputResult = dpsOp.getTiedOpResult(&operand);

      // Check if the output workaround is the same as the input DPS destination
      // workaround.
      if (workarounds.getOutputOperandWorkarounds()[tiedOutputResult
                                                        .getResultNumber()] !=
          workarounds.getInputOperandWorkarounds()[dpsDestinationIndex]) {
        return op->emitOpError()
               << "DPS output workaround does not match "
                  "the input DPS destination operand workaround "
               << tiedOutputResult.getResultNumber() << " and "
               << dpsDestinationIndex;
      }

      dpsDestinationIndex++;
    }
  }

  // All checks passed, return success.
  return mlir::success();
}
} // namespace mlir::tt::ttnn::wa
