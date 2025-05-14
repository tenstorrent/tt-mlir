// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATIONPLANNER_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATIONPLANNER_H

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <vector>

namespace mlir::tt::ttir {

/// An API for static DSA ("dynamic storage allocation") algorithms.
class AllocationPlanner final {
public:
  /// Enum for planning algorithms exposed by this API.
  /// @see AllocationPlanner::allocate()
  struct Algorithm {
    // clang-format off
    enum enum_t : std::int32_t {
      simple,   //! Monotonic "bumper" allocator, does not support dealloc.
      greedy    //! Greedy-by-size allocator.
    };
    // clang-format on
  };

  // Type for address-like values.
  using AllocSizeT = std::int64_t;
  // Type for liveness "logical time".
  using SequenceT = std::int32_t;

  /// An allocation descriptor, a range of memory usage `[offset, offset+size)`
  /// over a liveness range `[first, last]`.
  struct Record final {

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const Record &obj);

    // clang-format off
    AllocSizeT offset;  //! Start memory address (negative means N/A).
    AllocSizeT size;    //! Requested memory size.
    SequenceT first;    //! (Inclusive) start of live range.
    SequenceT last;     //! (Inclusive) end of live range.
    // clang-format on

  }; // end of class

  /// An allocation planning problem descriptor. Callers may extend to
  /// associate additional information.
  struct Context {
  public:
    Context() = default;

    /// @result count of request records
    [[nodiscard]] std::size_t size() const { return records.size(); }

    /// @result index'th allocation request as a `Record`
    [[nodiscard]] const Record &operator[](std::size_t index) const {
      return records[index];
    }

    /// Add an allocation request of given `size` and live range `[first,
    /// last]`.
    void add(AllocSizeT size, SequenceT first, SequenceT last);

  protected:
    friend class AllocationPlanner;
    template <AllocationPlanner::Algorithm::enum_t Algorithm>
    friend class PlannerImpl;

    std::vector<Record> records;

  }; // end of class

  /// Descriptor of allocation and verification outcomes.
  struct Stats final {

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const Stats &obj);

    /// @return `memUsage/maxLoad` ratio if `maxLoad` is non-zero,
    /// double NaN otherwise
    double usageRatio() const {
      return maxLoad ? static_cast<double>(memUsage) / maxLoad
                     : std::numeric_limits<double>::quiet_NaN();
    }

    /// Largest single-record memory size within an allocation context.
    AllocSizeT maxSize;
    /// Allocation needed `[0, memUsage)` memory strip to satisfy the
    /// allocation request (also known as "makespan").
    AllocSizeT memUsage;
    /// "Load" is the sum of sizes of all requests live at a given
    /// logical instant. This is the maximum such value across the
    /// entire logical time range. (0 indicates N/A)
    AllocSizeT maxLoad;

  }; // end of class

  /// Find a feasible allocation for Records in `context` (expected
  /// to have `size`, `first`, and `last` fields set).
  /// @return `Stats` with `maxSize` and `memUsage` fields set
  [[nodiscard]] static Stats
  allocate(Context &context, Algorithm::enum_t algorithm = Algorithm::greedy);

  /// Validate the allocation plan in `context` for proper memory/liveness
  /// conflict resolution and return stats with *all* fields recomputed.
  [[nodiscard]] static Stats verify(const Context &context);

private:
  AllocationPlanner() = default; // Non-instantiable.

}; // end of class
//............................................................................

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const AllocationPlanner::Record &obj) {
  return os << obj.size << " [" << obj.first << ", " << obj.last << "] @ "
            << obj.offset;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const AllocationPlanner::Stats &obj) {
  os << "{max size = " << obj.maxSize << ", mem usage = " << obj.memUsage;
  if (obj.maxLoad) {
    os << ", max load = " << obj.maxLoad;
  }
  return os << "}";
}

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATIONPLANNER_H
