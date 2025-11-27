// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_LOOPSEMANTICS_H
#define TTMLIR_DIALECT_D2M_UTILS_LOOPSEMANTICS_H

#include "mlir/IR/AffineMap.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

namespace utils {

/// Information about loop dimensions and their semantic roles in a GenericOp.
///
/// This struct captures metadata about the loop dimensions that will be
/// generated from a GenericOp (i.e., the loops described by the GenericOp's
/// iterator_types attribute, not any outer loops that contain the GenericOp).
/// It classifies each dimension as parallel or reduction, enabling semantic
/// analysis of which dimensions are needed for prologue/epilogue operations.
struct LoopDimensionInfo {
  /// Total number of loop dimensions
  unsigned numDimensions;

  /// Iterator type for each dimension (parallel, reduction, etc.)
  llvm::SmallVector<ttcore::IteratorType> iteratorTypes;

  /// Which dimensions are parallel
  llvm::SmallDenseSet<unsigned> parallelDims;

  /// Which dimensions are reduction
  llvm::SmallDenseSet<unsigned> reductionDims;

  /// Check if a dimension is parallel
  bool isParallel(unsigned dim) const { return parallelDims.contains(dim); }

  /// Check if a dimension is reduction
  bool isReduction(unsigned dim) const { return reductionDims.contains(dim); }
};

/// Information about which dimensions an operand accesses in a GenericOp.
///
/// This struct describes the access pattern of a specific operand (input or
/// output) by identifying which loop dimensions it participates in based on
/// its indexing map.
struct OperandAccessInfo {
  /// Operand index within the GenericOp
  unsigned operandIndex;

  /// Which loop dimensions this operand participates in (accesses)
  llvm::SmallVector<int64_t> participatingDims;

  /// Which loop dimensions this operand doesn't participate in
  llvm::SmallVector<int64_t> nonParticipatingDims;

  /// The affine map for this operand (maps loop dims to operand indices)
  mlir::AffineMap indexingMap;

  /// Whether this is an output operand
  bool isOutput;

  /// Check if this operand uses a specific dimension
  bool usesDimension(unsigned dim) const {
    return llvm::is_contained(participatingDims, static_cast<int64_t>(dim));
  }

  /// Check if this operand doesn't use a specific dimension
  bool doesNotUseDimension(unsigned dim) const {
    return llvm::is_contained(nonParticipatingDims, static_cast<int64_t>(dim));
  }
};

/// Analyzer for loop semantics from GenericOp metadata.
///
/// This class analyzes a GenericOp to extract semantic information about its
/// loop nest structure and operand access patterns. It provides a high-level
/// interface for querying which dimensions are parallel vs reduction, which
/// dimensions each operand accesses, and which dimensions should be included
/// in prologue/epilogue loops.
///
/// Example usage:
/// \code
///   LoopSemanticsAnalyzer analyzer(genericOp);
///
///   // Query dimension information
///   const LoopDimensionInfo &dimInfo = analyzer.getDimensionInfo();
///   // For matmul: dimInfo.parallelDims = {0, 1}, reductionDims = {2}
///
///   // Query output access pattern
///   unsigned outputIdx = genericOp.getOutputs().size() - 1;
///   OperandAccessInfo outputInfo = analyzer.getOperandAccessInfo(outputIdx);
///   // For matmul output C[i,j]: participatingDims = {0, 1}
///
///   // Get dimensions for prologue/epilogue
///   llvm::SmallVector<unsigned> prologueDims =
///       analyzer.getPrologueEpilogueDims(outputIdx);
///   // For matmul: {0, 1} - only i and j, not k (reduction)
/// \endcode
class LoopSemanticsAnalyzer {
public:
  /// Construct analyzer from a GenericOp
  explicit LoopSemanticsAnalyzer(GenericOp genericOp);

  /// Get overall loop dimension information
  const LoopDimensionInfo &getDimensionInfo() const { return dimInfo; }

  /// Get access information for a specific operand
  OperandAccessInfo getOperandAccessInfo(unsigned operandIndex) const;

  /// Get access information for all output operands
  llvm::SmallVector<OperandAccessInfo> getOutputAccessInfos() const;

  /// Get the dimensions that should be used for prologue/epilogue loops
  /// for a given output operand.
  ///
  /// This returns the dimensions actually accessed by the output operand,
  /// which is what prologue (L1→DST init) and epilogue (DST→L1 writeback)
  /// should iterate over.
  ///
  /// For a matmul with C[i,j] = A[i,k] * B[k,j], this returns {0, 1} for
  /// output C since it only uses dimensions i and j, not k.
  llvm::SmallVector<unsigned>
  getPrologueEpilogueDims(unsigned outputIndex) const;

  /// Get non-participating dimensions for an operand that should be guarded
  /// in prologue/epilogue.
  ///
  /// Returns dimensions that the operand doesn't participate in, which need
  /// guards (e.g., checking iteration index != 0 for reduction dimensions).
  llvm::SmallVector<unsigned> getGuardDims(unsigned operandIndex) const;

  /// Get the GenericOp being analyzed
  GenericOp getGenericOp() const { return genericOp; }

private:
  GenericOp genericOp;
  LoopDimensionInfo dimInfo;
  unsigned numInputs;
  unsigned numOutputs;

  /// Analyze dimensions and populate dimInfo
  void analyzeDimensions();
};

} // namespace utils
} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_UTILS_LOOPSEMANTICS_H
