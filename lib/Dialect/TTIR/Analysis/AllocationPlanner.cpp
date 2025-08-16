// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/AllocationPlanner.h"

#include "ttmlir/Dialect/TTIR/Analysis/AllocationTools.h"

#include "llvm/ADT/IntervalTree.h"

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <sstream>

namespace mlir::tt::ttir {

// ............................................................................

using std::int32_t;

using Request = AllocationPlanner::Request;
using AllocSizeT = AllocationPlanner::AllocSizeT;
using SequenceT = AllocationPlanner::SequenceT;
using IndexT = AllocationPlanner::IndexT;

// ............................................................................

bool AllocationPlanner::Problem::valid() const {
  const int32_t varCount = variables.size();
  const int32_t reqCount = requests.size();

  for (const VarRequest &req : requests) {
    if (req.size < 0) {
      TT_ALLOC_ERROR("invalid request.size: {}", req.size);
      return false;
    }
    if (req.varIndex < 0 || req.varIndex >= varCount) {
      TT_ALLOC_ERROR("request.variable out of bounds [0, {}): {}", varCount,
                     req.varIndex);
      return false;
    }
  }
  for (int32_t varIndex = 0; varIndex < varCount; ++varIndex) {
    const Variable &var = variables[varIndex];
    if (bound.contains(varIndex)) {
      if (!(var.placement < limit)) {
        TT_ALLOC_ERROR("bound variable #{} does not have its placement set: {}",
                       varIndex, static_cast<int32_t>(var.placement));
        return false;
      }
    }
    for (const auto &indices : var.domain) {
      for (const IndexT reqIndex : indices) {
        if (reqIndex < 0 || reqIndex >= reqCount) {
          TT_ALLOC_ERROR("variable request out of bounds [0, {}): {}", reqCount,
                         reqIndex);
          return false;
        }
      }
    }
  }

  return true;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const AllocationPlanner::Problem &obj) {
  std::stringstream s;
  AllocationTools::write(obj, s);
  return os << s.str();
}
// ............................................................................

AllocationPlanner::SolveStats
AllocationPlanner::spillAllocate(Problem &problem, AllocSizeT memUsageLimit) {
  TT_assert(problem.valid());

  const Algorithm algorithm = Algorithm::Greedy;
  TT_ALLOC_DEBUG("solution watermark set at {}, using '{}' base algorithm",
                 memUsageLimit, algorithm);

  // Start with every (unbound) variable placed into scratch space.
  problem.reset(Space::Scratch);

  AllocateStats stats = allocate(problem, algorithm);
  if (stats.memUsage <= memUsageLimit) {
    TT_ALLOC_DEBUG("no spilling was required");
    return {0, 0, stats};
  }

  Problem work = problem;
  const AnalysisStats metrics = analyze(work, memUsageLimit);

  struct SpillPriority {
    AllocSizeT worstSize = 0;
    int32_t worstMaxConflictSize = 0;
    bool inContentionGroup = false;
    // TODO total live range duration OR some cumsum of conflict counts
    // TODO delta b/w unspilled and spilled mem requirements?
    IndexT varIndex = -1;
  };

  std::vector<SpillPriority> priorities;

  // TODO templatize this comparator
  auto byPriority = [&](const SpillPriority &lhs, const SpillPriority &rhs) {
    return (lhs.inContentionGroup > rhs.inContentionGroup) ||
           ((lhs.inContentionGroup == rhs.inContentionGroup) &&
            ((lhs.worstSize > rhs.worstSize) ||
             ((lhs.worstSize == rhs.worstSize) &&
              (lhs.worstMaxConflictSize > rhs.worstMaxConflictSize))));
  };

  const int32_t varCount = work.variables.size();
  const int32_t freeVarCount = varCount - work.bound.size();

  priorities.reserve(freeVarCount);

  for (IndexT varIndex = 0; varIndex < varCount; ++varIndex) {
    if (work.bound.contains(varIndex)) {
      continue;
    }
    const AllocationPlanner::Variable &var = work.variable(varIndex);
    TT_assert_limit(var.placement, Space::limit);

    SpillPriority varPriority;
    {
      varPriority.varIndex = varIndex;

      for (const IndexT reqIndex : var.domain[var.placement]) {
        const AnalysisStats::RequestMetrics &reqMetrics =
            metrics.requestMetrics[reqIndex];

        varPriority.worstSize =
            std::max(varPriority.worstSize, work.request(reqIndex).size);
        varPriority.worstMaxConflictSize = std::max(
            varPriority.worstMaxConflictSize, reqMetrics.maxConflictSize);
        varPriority.inContentionGroup |= reqMetrics.inContentionGroup;
      }
    }
    priorities.emplace_back(varPriority);
  }
  TT_debug(static_cast<int32_t>(priorities.size()) == freeVarCount);

  std::sort(priorities.begin(), priorities.end(), byPriority);

  int32_t spilledCount = -1;
  int32_t stepCount;

  int32_t lo = 0;
  int32_t hi = priorities.size() - 1;

  for (stepCount = 1; lo <= hi; ++stepCount) {
    const int32_t mid = lo + ((hi - lo) >> 1);

    TT_ALLOC_DEBUG("[step {}] lo/hi = [{}/{}], mid = {}", stepCount, lo, hi,
                   mid);

    work.reset(Space::Scratch);
    for (int32_t k = 0; k <= mid; ++k) {
      TT_debug(!work.bound.contains(priorities[k].varIndex));
      work.variable(priorities[k].varIndex).placement = Space::Spill;
    }

    const AllocateStats midStats = allocate(work, algorithm);
    TT_ALLOC_DEBUG("[step {}] mem usage {}", stepCount, midStats.memUsage);

    if (midStats.memUsage <= memUsageLimit) {
      hi = mid - 1;

      problem = work;
      stats = midStats;
      spilledCount = mid + 1;
    } else {
      lo = mid + 1;
    }
  }

  if (spilledCount < 0) {
    TT_ALLOC_ERROR("failed to allocate below mem usage limit {} after spilling "
                   "all {} var(s)",
                   memUsageLimit, freeVarCount);
    return {stepCount, -1, stats};
  }

  // Post-condition: `problem` is set to the last midpoint solution.

  TT_debug(spilledCount > 0);
  TT_debug(stats.memUsage <= memUsageLimit);

  TT_ALLOC_DEBUG("converged after spilling {} var(s) out of {}", spilledCount,
                 priorities.size());

  return {stepCount, spilledCount, stats};
}
// ............................................................................

template <AllocationPlanner::Algorithm Algorithm>
class PlannerImpl {
public:
  static inline AllocationPlanner::AllocateStats
  allocate(AllocationPlanner::Problem &problem);
};
// ............................................................................

//
// A simple bumper allocator. It ignores live ranges and thus can't reuse
// memory.
//
template <>
inline AllocationPlanner::AllocateStats
PlannerImpl<AllocationPlanner::Algorithm::Simple>::allocate(
    AllocationPlanner::Problem &problem) {

  // Not required for correctness, but for easier result interpretation
  // visit allocation requests in IR preorder.

  auto &requests = problem.requests;

  auto lexicographic = [&](IndexT lhs, IndexT rhs) {
    return (requests[lhs].first > requests[rhs].first) ||
           ((requests[lhs].first == requests[rhs].first) &&
            (requests[lhs].last > requests[rhs].last));
  };

  std::priority_queue<IndexT, std::vector<IndexT>, decltype(lexicographic)> pq(
      lexicographic);
  [[maybe_unused]] int32_t varCount = 0;

  for (const AllocationPlanner::Variable &var : problem.variables) {
    if (var.placement == AllocationPlanner::Space::NA) {
      continue;
    }
    for (const auto reqIndex : var.domain[var.placement]) {
      pq.push(reqIndex);
    }
    ++varCount;
  }

  TT_ALLOC_DEBUG("allocating {} var(s), {} requests(s) ...", varCount,
                 pq.size());

  AllocSizeT memUsage = 0;
  AllocSizeT maxSize = 0;

  while (!pq.empty()) {
    const IndexT reqIndex = pq.top();
    pq.pop();
    Request &request = requests[reqIndex];
    TT_ALLOC_TRACE("request #{}: {}", reqIndex, request);

    request.offset = memUsage;
    memUsage += request.size;

    TT_ALLOC_TRACE("request #{}: placed at offset {}", reqIndex,
                   request.offset);

    maxSize = std::max(maxSize, request.size);
  }

  AllocationPlanner::AllocateStats stats = {maxSize, memUsage, 0};
  TT_ALLOC_DEBUG("allocation results: {}", stats);

  return stats;
}
// ............................................................................
//
// Greedy-by-size allocator:
//  1. visit requests in decreasing memory size order;
//  2. place each request into the tighest gap found within the conflict set
//     formed by already placed requests; if no such gap is found, extend
//     the solution makespan.
template <>
inline AllocationPlanner::AllocateStats
PlannerImpl<AllocationPlanner::Algorithm::Greedy>::allocate(
    AllocationPlanner::Problem &problem) {

  auto &requests = problem.requests;

  auto bySize = [&](IndexT lhs, IndexT rhs) {
    return (requests[lhs].size < requests[rhs].size);
  };

  std::priority_queue<IndexT, std::vector<IndexT>, decltype(bySize)> pq(bySize);
  [[maybe_unused]] int32_t varCount = 0;

  for (const AllocationPlanner::Variable &var : problem.variables) {
    if (var.placement == AllocationPlanner::Space::NA) {
      continue;
    }
    for (const auto reqIndex : var.domain[var.placement]) {
      pq.push(reqIndex);
    }
    ++varCount;
  }

  TT_ALLOC_DEBUG("allocating {} var(s), {} requests(s) ...", varCount,
                 pq.size());

  // An index of already visited requests in increasing offset order.
  std::multimap<AllocSizeT, IndexT>
      allocatedOrderedByOffset; // TODO unique w/index, switch to DenseMap

  AllocSizeT memUsage = 0;
  AllocSizeT maxSize = 0;

  while (!pq.empty()) {
    const IndexT reqIndex = pq.top();
    pq.pop();
    Request &request = requests[reqIndex];
    TT_ALLOC_TRACE("request #{}: {}", reqIndex, request);
    TT_assertv(request.offset < 0,
               "request should not be marked allocated yet: {}", request);

    AllocSizeT gapBest = std::numeric_limits<AllocSizeT>::max();
    AllocSizeT offsetPrev = 0;
    AllocSizeT offsetBest = -1;

    for (const auto &[kOffset, k] : allocatedOrderedByOffset) {
      const Request &allocatedRequest = requests[k];
      TT_debug((kOffset >= 0 && kOffset == allocatedRequest.offset));

      const SequenceT maxFirst =
          std::max(request.first, allocatedRequest.first);
      const SequenceT minLast = std::min(request.last, allocatedRequest.last);

      // If `allocatedRequest` conflicts with `request`, check if we can use the
      // gap between it and the previous conflict.

      if (maxFirst <= minLast) {
        TT_ALLOC_TRACE("\tconflict request #{}: {}", k, allocatedRequest);
        const AllocSizeT gap = allocatedRequest.offset - offsetPrev;
        if (gap >= allocatedRequest.size && gap < gapBest) {
          gapBest = gap;
          offsetBest = offsetPrev;
        }
        offsetPrev = std::max(offsetPrev,
                              allocatedRequest.offset + allocatedRequest.size);
      }
    }

    if (offsetBest < 0) {
      offsetBest = offsetPrev;
    }

    TT_ALLOC_TRACE("request #{}: placed at offset {}", reqIndex, offsetBest);

    request.offset = offsetBest;
    allocatedOrderedByOffset.emplace(offsetBest, reqIndex);

    memUsage = std::max(memUsage, offsetBest + request.size);
    maxSize = std::max(maxSize, request.size);
  }

  AllocationPlanner::AllocateStats stats = {maxSize, memUsage, 0};
  TT_ALLOC_DEBUG("allocation results: {}", stats);

  return stats;
}
// ............................................................................

AllocationPlanner::AllocateStats
AllocationPlanner::allocate(Problem &problem, Algorithm algorithm) {
  switch (algorithm) {
  case Algorithm::Simple:
    return PlannerImpl<Algorithm::Simple>::allocate(problem);
  case Algorithm::Greedy:
    return PlannerImpl<Algorithm::Greedy>::allocate(problem);
  }
}
// ............................................................................

AllocationPlanner::AnalysisStats
AllocationPlanner::analyze(const Problem &solution,
                           const AllocSizeT watermark) {
  TT_assert(solution.valid());

  // All variables placed into 'space' by 'solution' and marked for lowering by
  // 'watermark'. Also, requests of those variables that triggered the inclusion
  // (in general, a subset of all requests in variables' domains).
  AllocationPlanner::AnalysisStats analysis;

  std::vector<IndexT> requests; // TODO rename

  using IntervalTree = llvm::IntervalTree<SequenceT, IndexT>;

  IntervalTree::Allocator allocator;
  IntervalTree iv(allocator);

  for (const Variable &var : solution.variables) {
    if (var.placement == Space::NA) {
      continue;
    }
    for (const auto reqIndex : var.domain[var.placement]) {
      const Request &req = solution.request(reqIndex);

      requests.emplace_back(reqIndex);
      iv.insert(req.first, req.last, reqIndex);
    }
  }
  iv.create();

  analysis.requestMetrics.resize(solution.requests.size());

  for (const auto reqIndex : requests) {
    const Request &req = solution.request(reqIndex);
    TT_assertv(req.offset >= 0, "unexpected unallocated request {}", req);

    SequenceT position = req.first;

    // First, determine if mem usage at 'position' marks it as being inside a
    // contention group.

    AllocSizeT usage = 0;

    const auto conflicts = iv.getContaining(position);
    for (const auto &entry : conflicts) {
      const auto conflictIndex = entry->value();
      const Request &conflict = solution.request(conflictIndex);

      usage = std::max(usage, conflict.offset + conflict.size);
    }

    AnalysisStats::RequestMetrics &metrics = analysis.requestMetrics[reqIndex];

    metrics.inContentionGroup |= (usage > watermark);
    metrics.maxConflictSize =
        std::max<int32_t>(metrics.maxConflictSize, conflicts.size());
  }

  return analysis;
}
// ............................................................................

AllocationPlanner::AllocateStats
AllocationPlanner::verify(const Problem &solution) {
  // Use an interval tree to both verify interval conflicts and calculate max
  // load/usage.
  //
  // Note that even though LLVM's interval tree only supports interval stabbing
  // queries (find all intervals containing a given point), it is sufficient for
  // our verification purposes: for any two overlapping intervals it is true
  // that one of their starting points stabs the other interval and thus our
  // sweep is guaranteed to visit each such conflict edge, either when visiting
  // the first interval or when visiting the second. And hence no load
  // contributions will be missed by the sweep.

  using IntervalTree = llvm::IntervalTree<SequenceT, IndexT>;

  IntervalTree::Allocator allocator;
  IntervalTree iv(allocator);

  // All requests referenced by 'solution.variables' placed into 'space'.
  std::vector<AllocationPlanner::IndexT> requests; // TODO rename
  [[maybe_unused]] int32_t varCount = 0;

  for (const Variable &var : solution.variables) {
    if (var.placement == Space::NA) {
      continue;
    }
    for (const auto reqIndex : var.domain[var.placement]) {
      const Request &req = solution.request(reqIndex);

      requests.emplace_back(reqIndex);
      iv.insert(req.first, req.last, reqIndex);
    }
    ++varCount;
  }
  iv.create();

  const int32_t reqCount = requests.size();
  TT_ALLOC_TRACE("verifying {} var(s), {} request(s) ...", varCount, reqCount);
  if (!reqCount) {
    return {0, 0, 0};
  }

  AllocSizeT memUsage = 0;
  AllocSizeT maxSize = 0;
  AllocSizeT maxLoad = 0;

  for (const auto reqIndex : requests) {
    const Request &req = solution.request(reqIndex);
    TT_assertv(req.offset >= 0, "unexpected unallocated request {}", req);

    AllocSizeT load = req.size;

    const auto conflicts = iv.getContaining(req.first);
    for (const auto &entry : conflicts) {
      if (entry->value() != reqIndex) {
        const Request &conflict = solution.request(entry->value());

        // 'conflict' and 'request' must not overlap in memory space.
        // Note that a memory range is considered half-open, i.e. [offset,
        // offset+size).
        const bool memOverlap =
            std::max(req.offset, conflict.offset) <
            std::min(req.offset + req.size, conflict.offset + conflict.size);
        TT_assertv(!memOverlap, "memory overlap b/w {} and {}", req, conflict);

        load += conflict.size;
      }
    }

    TT_ALLOC_TRACE("request #{}: {} conflict(s), size {}, load {}", reqIndex,
                   (conflicts.size() - 1), req.size, load);

    memUsage = std::max(memUsage, req.offset + req.size);
    maxLoad = std::max(maxLoad, load);
    maxSize = std::max(maxSize, req.size);
  }

  TT_assertv(maxLoad > 0, "should have seen positive max load");
  TT_assertv(maxLoad <= memUsage,
             "inconsistent max load/mem usage metrics inferred");

  TT_ALLOC_TRACE("verified allocation plan with {} variable(s), {} request(s): "
                 "max alloc size {}, mem "
                 "usage {}, max load {} (ratio {})",
                 varCount, reqCount, maxSize, memUsage, maxLoad,
                 static_cast<double>(memUsage) / maxLoad);

  return {maxSize, memUsage, maxLoad};
}

} // namespace mlir::tt::ttir
