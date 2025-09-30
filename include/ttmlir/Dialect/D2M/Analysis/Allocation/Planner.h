// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_PLANNER_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_PLANNER_H

#include "ttmlir/Dialect/D2M/Analysis/Allocation/PlannerImpl.h"

namespace mlir::tt::d2m::allocation {

/// An API for static DSA ("dynamic storage allocation") algorithms.
///
/// Planner works with problem specifications that contain
/// two kinds of settable state:
///  - Memory "requests" are claims on `Scratch` memory address regions
///   that exist throughout a finite "live range". Liveness and size are
///   provided by the caller; the planner's `allocate()` assigns offsets within
///   the address space that is understood to start at zero `AllocSizeT`.
///  - Decision "variables" represent discrete "placement" choices of which
///   `Space` an entity like an operand is to reside in. Variables come with
///   resource claims that are groups of memory "requests". A variable can
///   contain more than one request. "Spilling", i.e. moving a variable from one
///   memory space to another, changes its `Scratch` claims according to
///   the variable's "domain" as specified by the problem. The planner's
///   `spillAllocate()` will attempt to assign variable placements and their
///   request offsets such that the `Scratch` space is in a feasible state. In
///   that sense, `spillAllocate()` could be seen as "decide which variables to
///   spill, followed by allocating remaining unspilled variables; do both
///   'optimally'".
///
/// The caller can mark some problem variables as "bound", in which case
/// the planner will consider them pinned to a particular space but will still
/// include their memory offsets in the allocatin/defragmentation part of its
/// task.
///
class Planner : public PlannerImpl {
private:
  using Impl = PlannerImpl;

public:
  /// @see PlannerImpl
  using Impl::AllocSizeT;
  using Impl::IndexT;
  using Impl::LiveRange;
  using Impl::SequenceT;

  using Impl::Algorithm;
  using Impl::Problem;

  using Impl::AllocateStats;
  using Impl::SpillStats;

  /// For all variables in `problem` that are not marked disabled (i.e. have
  /// non-"NA" placement), find an "optimal" assignment of request offsets. This
  /// optimality means to keep the resulting solution within as small
  /// `AllocateStats::memUsage` budget as possible.
  /// This call will mutate `problem` in place.
  /// @param problem expected to contain Variables with Requests that have
  /// `first`, `last`, and `size` fields set.
  /// @return `AllocateStats` with `maxSize` and `memUsage` fields set
  [[nodiscard]] static AllocateStats
  allocate(Problem &problem, Algorithm algorithm = Algorithm::Greedy);

  /// For all variables in `problem` that are not marked disabled (i.e. have
  /// non-"NA" placement), find an "optimal" set of variables to spill from
  /// `Space::Scratch` to `Space::Spill` and an "optimal" assignment of offsets
  /// for those requests that remain in `Space::Scratch` post-spill. The current
  /// implementation will heuristically choose which variables to spill until
  /// the problem is solveable as if by `allocate()` within the `memUsageLimit`
  /// budget.
  /// This call will mutate `problem` in place and the final solution is only
  /// valid if it is feasible, i.e `SpillStats::memUsage <= memUsageLimit`.
  /// @param problem expected to contain Variables with Requests that have
  /// `first`, `last`, and `size` fields set. Variables in `Problem::bound` are
  /// interpreted as ineligible for spill decisions by the planner but will
  /// participate in the defragmentation part of the problem.
  /// @param algorithm the `Algorithm` to use for the inner (allocation) part of
  /// the spill-and-allocate iteration loop.
  /// @return `SpillStats`, an extension of `AllocateStats` that will
  /// additionally report `stepsTaken` and `spillCount`.
  [[nodiscard]] static SpillStats
  spillAllocate(Problem &problem, AllocSizeT memUsageLimit,
                Algorithm algorithm = Algorithm::Greedy);

  /// Validate the allocation plan in `solution` for proper memory/liveness
  /// conflict resolution.
  /// @param problem Variables that are not marked disabled must contain
  /// Requests that have `first`, `last`, `size`, and `offset` fields set.
  /// @return `AllocateStats` with the same fields as done by `allocate()`
  /// and in addition with `maxLoad` computed as well (the ratio of
  /// `memUsage/maxLoad` acts as a measure of solution quality/defragmentation).
  [[nodiscard]] static AllocateStats verify(const Problem &solution);

private:
  Planner() = default; // Non-instantiable.
};

} // namespace mlir::tt::d2m::allocation

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_PLANNER_H
