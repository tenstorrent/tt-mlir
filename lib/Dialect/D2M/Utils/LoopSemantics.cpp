// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/LoopSemantics.h"
#include "mlir/IR/AffineExpr.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m::utils {

LoopSemanticsAnalyzer::LoopSemanticsAnalyzer(GenericOp genericOp)
    : genericOp(genericOp), numInputs(genericOp.getInputs().size()),
      numOutputs(genericOp.getOutputs().size()) {
  analyzeDimensions();
}

void LoopSemanticsAnalyzer::analyzeDimensions() {
  // Get iterator types from GenericOp
  auto iteratorTypes = genericOp.getIteratorTypesValue();
  dimInfo.numDimensions = iteratorTypes.size();
  dimInfo.iteratorTypes = iteratorTypes;

  // Classify dimensions by iterator type
  for (unsigned i = 0; i < dimInfo.numDimensions; ++i) {
    switch (iteratorTypes[i]) {
    case ttcore::IteratorType::Parallel:
      dimInfo.parallelDims.insert(i);
      break;
    case ttcore::IteratorType::Reduction:
      dimInfo.reductionDims.insert(i);
      break;
    }
  }
}

OperandAccessInfo
LoopSemanticsAnalyzer::getOperandAccessInfo(unsigned operandIndex) const {
  OperandAccessInfo info;
  info.operandIndex = operandIndex;
  info.isOutput = operandIndex >= numInputs;

  // GenericOp methods aren't const, so we need to const_cast here
  // This is safe because we're only reading, not modifying
  auto mutableOp = const_cast<GenericOp &>(genericOp);

  // Get the indexing map for this operand
  info.indexingMap = mutableOp.getIndexingMap(operandIndex);

  // Get participating and non-participating dimensions
  // These methods return which loop dimensions the operand accesses
  info.participatingDims = mutableOp.getParticipatingLoopDims(operandIndex);
  info.nonParticipatingDims =
      mutableOp.getNonParticipatingLoopDims(operandIndex);

  return info;
}

llvm::SmallVector<OperandAccessInfo>
LoopSemanticsAnalyzer::getOutputAccessInfos() const {
  llvm::SmallVector<OperandAccessInfo> outputInfos;
  unsigned totalOperands = numInputs + numOutputs;

  for (unsigned i = numInputs; i < totalOperands; ++i) {
    outputInfos.push_back(getOperandAccessInfo(i));
  }

  return outputInfos;
}

llvm::SmallVector<unsigned>
LoopSemanticsAnalyzer::getPrologueEpilogueDims(unsigned outputIndex) const {
  // Get the operand index (outputs start after inputs)
  unsigned operandIndex = numInputs + outputIndex;

  // GenericOp methods aren't const, so we need to const_cast here
  auto mutableOp = const_cast<GenericOp &>(genericOp);

  // Get participating dimensions for this output
  auto participatingDims = mutableOp.getParticipatingLoopDims(operandIndex);

  // Convert to unsigned for consistency
  llvm::SmallVector<unsigned> dims;
  dims.reserve(participatingDims.size());
  for (int64_t dim : participatingDims) {
    dims.push_back(static_cast<unsigned>(dim));
  }

  return dims;
}

llvm::SmallVector<unsigned>
LoopSemanticsAnalyzer::getGuardDims(unsigned operandIndex) const {
  // GenericOp methods aren't const, so we need to const_cast here
  auto mutableOp = const_cast<GenericOp &>(genericOp);

  // Get non-participating dimensions
  auto nonParticipatingDims =
      mutableOp.getNonParticipatingLoopDims(operandIndex);

  // Convert to unsigned
  llvm::SmallVector<unsigned> guardDims;
  guardDims.reserve(nonParticipatingDims.size());
  for (int64_t dim : nonParticipatingDims) {
    guardDims.push_back(static_cast<unsigned>(dim));
  }

  return guardDims;
}

} // namespace mlir::tt::d2m::utils
