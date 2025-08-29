// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_TOOLS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_TOOLS_H

#include "ttmlir/Dialect/TTIR/Analysis/Allocation/PlannerImpl.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <filesystem>

namespace mlir::tt::ttir::allocation {

/// An API for synthesizing and serializing `Planner` problems without
/// having to get an IR representation of them first.
class Tools {
public:
  /// A randomly generated Problem is based on a topological sort order of
  /// a DAG comprised of a sequence of "segments". Each segment describes
  /// a sub-DAG started by a linear chain of `neckLength` length and followed
  /// by a section that is guaranteed to create `conflictCount` liveness
  /// conflicts. Each such sub-DAG thus results in low liveness contention of
  /// controllable length followed by controllably high contention.
  ///
  /// Conceptually, nodes of such a DAG are meant to represent "ops" of a
  /// calculation graph. Each node maps to a `Planner::Variable` and each
  /// outgoing edge maps to a "tensor" result that is consumed by the
  /// corresponding successor node. Each result requires some randomly
  /// generated amount of "memory" and maps to a `Planner::VarRequest`.
  ///
  /// A liveness range extends from the moment of output "creation" to its
  /// last consuming node and uses "logical time" that is the consuming node's
  /// position in the topological ordering of the overall DAG.
  ///
  /// Note that the generator makes sure that the DAG contains nodes of output
  /// degree of at most 2. A random subset of the nodes is made into bound
  /// variables.

  struct GenerateSegmentParms {
    std::int32_t neckLength;
    std::int32_t conflictCount;
  };

  struct GenerateCfg {
    llvm::SmallVector<GenerateSegmentParms> segments;
    double bindFraction;
    std::uint64_t seed;

    GenerateCfg(llvm::SmallVector<GenerateSegmentParms> &&segments,
                double bindFraction, std::uint64_t seed)
        : segments(std::move(segments)), bindFraction(bindFraction),
          seed(seed) {}

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const GenerateCfg &obj);
  };

  using Problem = PlannerImpl::Problem;

  static Problem generate(const GenerateCfg &cfg);

  static llvm::Expected<Problem> read(std::istream &in);
  static llvm::Expected<Problem> read(const std::filesystem::path &file);

  static void write(const Problem &problem, std::ostream &out);
  static void write(const Problem &problem, const std::filesystem::path &file);

private:
  Tools() = default; // Non-instantiable.
};

} // namespace mlir::tt::ttir::allocation

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_TOOLS_H
