// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/AllocationPlanner.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/IntervalTree.h"

#include <algorithm>
#include <limits>
#include <map>
#include <queue>

namespace mlir::tt::ttir {

// Define some local convenience macros.

#define TT_ALLOC_DEBUG(/* fmt, args */...)                                     \
  TTMLIR_DEBUG(ttmlir::LogComponent::Allocator, __VA_ARGS__)

#define TT_ALLOC_TRACE(/* fmt, args */...)                                     \
  TTMLIR_TRACE(ttmlir::LogComponent::Allocator, __VA_ARGS__)

using Record = AllocationPlanner::Record;
using AllocSizeT = AllocationPlanner::AllocSizeT;
using SequenceT = AllocationPlanner::SequenceT;
using IndexT = std::int32_t;

void AllocationPlanner::Context::add(AllocSizeT size, SequenceT first,
                                     SequenceT last) {
  TT_assertv(size > 0, "expected positive size: {}", size);
  TT_assert((0 <= first && 0 <= last));
  TT_assert(first <= last);

  [[maybe_unused]] const auto &r =
      records.emplace_back(Record{-1, size, first, last});
  TT_ALLOC_TRACE("added request record #{}: {}", (records.size() - 1), r);
}

template <AllocationPlanner::Algorithm Algorithm>
class PlannerImpl {
public:
  static inline AllocationPlanner::Stats
  allocate(AllocationPlanner::Context &context);
};

//
// A simple bumper allocator. It ignores live ranges and thus can't reuse
// memory.
//
template <>
inline AllocationPlanner::Stats
PlannerImpl<AllocationPlanner::Algorithm::Simple>::allocate(
    AllocationPlanner::Context &context) {
  auto &records = context.records;
  const IndexT n = static_cast<IndexT>(records.size());
  TT_ALLOC_DEBUG("allocating {} record(s) ...", n);

  AllocSizeT memUsage = 0;
  AllocSizeT maxSize = 0;

  // Not required for correctness, but for easier result interpretation
  // visit allocation records in IR preorder.

  auto lexicographic = [&](IndexT lhs, IndexT rhs) {
    return (records[lhs].first > records[rhs].first) ||
           ((records[lhs].first == records[rhs].first) &&
            (records[lhs].last > records[rhs].last));
  };
  std::priority_queue<IndexT, std::vector<IndexT>, decltype(lexicographic)> pq(
      lexicographic);
  for (IndexT i = 0; i < n; ++i) {
    pq.push(i);
  }

  while (!pq.empty()) {
    const IndexT i = pq.top();
    pq.pop();
    Record &record = records[i];
    TT_ALLOC_TRACE("record #{}: {}", i, record);

    record.offset = memUsage;
    memUsage += record.size;

    TT_ALLOC_TRACE("record #{}: placed at offset {}", i, record.offset);

    maxSize = std::max(maxSize, record.size);
  }

  TT_ALLOC_DEBUG("allocated {} record(s): max alloc size {}, mem usage {}", n,
                 maxSize, memUsage);

  return {maxSize, memUsage, 0};
}
//
// Greedy-by-size allocator:
//  1. visit requests in decreasing memory size order;
//  2. place each request into the tighest gap found within the conflict set
//     formed by already placed requests; if no such gap is found, extend
//     the solution makespan.
template <>
inline AllocationPlanner::Stats
PlannerImpl<AllocationPlanner::Algorithm::Greedy>::allocate(
    AllocationPlanner::Context &context) {
  auto &records = context.records;
  const IndexT n = static_cast<IndexT>(records.size());
  TT_ALLOC_DEBUG("allocating {} record(s) ...", n);

  auto bySize = [&](IndexT lhs, IndexT rhs) {
    return (records[lhs].size < records[rhs].size);
  };
  std::priority_queue<IndexT, std::vector<IndexT>, decltype(bySize)> pq(bySize);
  for (IndexT i = 0; i < n; ++i) {
    pq.push(i);
  }

  // An index of already visited records in increasing offset order.
  std::multimap<AllocSizeT, IndexT> allocatedOrderedByOffset;

  AllocSizeT memUsage = 0;
  AllocSizeT maxSize = 0;

  while (!pq.empty()) {
    const IndexT i = pq.top();
    pq.pop();
    Record &record = records[i];
    TT_ALLOC_TRACE("record #{}: {}", i, record);
    TT_assertv(record.offset < 0,
               "record should not be marked allocated yet: {}", record);

    AllocSizeT gapBest = std::numeric_limits<AllocSizeT>::max();
    AllocSizeT offsetPrev = 0;
    AllocSizeT offsetBest = -1;

    for (const auto &[kOffset, k] : allocatedOrderedByOffset) {
      const Record &allocatedRecord = records[k];
      TT_debugv((kOffset >= 0 && kOffset == allocatedRecord.offset),
                "iterating over allocated records");

      const SequenceT maxFirst = std::max(record.first, allocatedRecord.first);
      const SequenceT minLast = std::min(record.last, allocatedRecord.last);

      // If `allocatedRecord` conflicts with `record`, check if we can use the
      // gap between it and the previous conflict.

      if (maxFirst <= minLast) {
        TT_ALLOC_TRACE("\tconflict record #{}: {}", k, allocatedRecord);
        const AllocSizeT gap = allocatedRecord.offset - offsetPrev;
        if (gap >= allocatedRecord.size && gap < gapBest) {
          gapBest = gap;
          offsetBest = offsetPrev;
        }
        offsetPrev =
            std::max(offsetPrev, allocatedRecord.offset + allocatedRecord.size);
      }
    }

    if (offsetBest < 0) {
      offsetBest = offsetPrev;
    }

    TT_ALLOC_TRACE("record #{}: placed at offset {}", i, offsetBest);

    record.offset = offsetBest;
    allocatedOrderedByOffset.emplace(offsetBest, i);

    memUsage = std::max(memUsage, offsetBest + record.size);
    maxSize = std::max(maxSize, record.size);
  }

  TT_ALLOC_DEBUG("allocated {} record(s): max alloc size {}, mem usage {}", n,
                 maxSize, memUsage);
  return {maxSize, memUsage, 0};
}

AllocationPlanner::Stats AllocationPlanner::allocate(Context &context,
                                                     Algorithm algorithm) {
  switch (algorithm) {
  case Algorithm::Simple:
    return PlannerImpl<Algorithm::Simple>::allocate(context);
  case Algorithm::Greedy:
    return PlannerImpl<Algorithm::Greedy>::allocate(context);
  }
}

AllocationPlanner::Stats AllocationPlanner::verify(const Context &context) {
  TT_ALLOC_TRACE("verifying {} allocation(s)", context.size());
  if (!context.size()) {
    return {0, 0, 0};
  }

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

  const IndexT n = static_cast<IndexT>(context.size());

  for (IndexT i = 0; i < n; ++i) {
    const Record &record = context[i];
    iv.insert(record.first, record.last, i);
  }
  // NOLINTNEXTLINE
  iv.create();

  AllocSizeT memUsage = 0;
  AllocSizeT maxSize = 0;
  AllocSizeT maxLoad = 0;

  for (IndexT i = 0; i < n; ++i) {
    const Record &record = context[i];

    TT_assertv(record.offset >= 0, "unallocated record: {}", record);
    AllocSizeT load = record.size;

    const auto conflicts = iv.getContaining(record.first);
    for (const auto &entry : conflicts) {
      if (entry->value() != i) {
        const Record &conflict = context[entry->value()];

        // 'conflict' and 'record' must not overlap in memory space.
        // Note that a memory range is considered half-open, i.e. [offset,
        // offset+size).
        [[maybe_unused]] const bool memOverlap =
            std::max(record.offset, conflict.offset) <
            std::min(record.offset + record.size,
                     conflict.offset + conflict.size);
        TT_assertv(!memOverlap, "memory overlap b/w {} and {}", record,
                   conflict);

        load += conflict.size;
      }
    }

    TT_ALLOC_TRACE("record #{}: {} conflict(s), size {}, load {}", i,
                   (conflicts.size() - 1), record.size, load);

    memUsage = std::max(memUsage, record.offset + record.size);
    maxLoad = std::max(maxLoad, load);
    maxSize = std::max(maxSize, record.size);
  }

  TT_assert(maxLoad > 0);
  TT_assert(maxLoad <= memUsage);

  TT_ALLOC_DEBUG(
      "verified allocation plan with {} record(s): max alloc size {}, mem "
      "usage {}, max load {} (ratio {})",
      context.size(), maxSize, memUsage, maxLoad,
      static_cast<double>(memUsage) / maxLoad);

  return {maxSize, memUsage, maxLoad};
}

#undef TT_ALLOC_TRACE
#undef TT_ALLOC_DEBUG

} // namespace mlir::tt::ttir
