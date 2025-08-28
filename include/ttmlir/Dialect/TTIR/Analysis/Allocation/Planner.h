// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_PLANNER_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_PLANNER_H

#include "ttmlir/Dialect/TTIR/Analysis/Allocation/Utils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>

namespace mlir::tt::ttir::allocation {

/// An API for static DSA ("dynamic storage allocation") algorithms.
class Planner {
public:
  /// Enum for planning algorithms exposed by this API.
  /// @see AllocationPlanner::allocate()
  enum class Algorithm : std::int32_t {
    Simple, //! Monotonic "bumper" allocator, does not support dealloc.
    Greedy  //! Greedy-by-size allocator.
  };

  friend std::string to_string(Algorithm e) {
    switch (e) {
    case Algorithm::Simple:
      return "simple";
    case Algorithm::Greedy:
      return "greedy";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Algorithm &obj) {
    return os << to_string(obj);
  }

  // TODO(vroubtsov) it probably makes more sense these types template
  // parameters for the planner API and let Allocate pass do their actual
  // selection.

  // Type for address-like values.
  using AllocSizeT = std::int64_t;
  // Type for liveness "logical time".
  using SequenceT = std::int32_t;
  // Type for indexing into variable/request collections.
  using IndexT = std::int32_t; // more compact index type than std::size_t

  struct LiveRange {
    SequenceT first = -1; //! (Inclusive) start of live range.
    SequenceT last = -1;  //! (Inclusive) end of live range.

    bool valid() const { return 0 <= first && first <= last; }

    friend bool operator==(const LiveRange &lhs, const LiveRange &rhs) {
      return (lhs.first == rhs.first) && (lhs.last == rhs.last);
    }
  };

  // Either an L1 alloc with its full live range or a DRAM alloc with
  // a set of L1 buffer allocs

  /// An allocation descriptor, a range of memory usage `[offset, offset+size)`
  /// over a liveness range `[first, last]`.
  struct Request : public LiveRange {

    AllocSizeT size = -1;   //! Requested memory size.
    AllocSizeT offset = -1; //! Start memory address (negative means N/A).

    friend bool operator==(const Request &lhs, const Request &rhs) {
      if (!(static_cast<const LiveRange &>(lhs) ==
            static_cast<const LiveRange &>(rhs))) {
        return false;
      }
      return (lhs.size == rhs.size) && (lhs.offset == rhs.offset);
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const Request &obj);
    template <bool Nested = false>
    void print(llvm::raw_ostream &os) const;
  };

  struct VarRequest : public Request {

    IndexT varIndex = -1; //! Parent variable (index into `Problem::variables`).

    friend bool operator==(const VarRequest &lhs, const VarRequest &rhs) {
      if (!(static_cast<const Request &>(lhs) ==
            static_cast<const Request &>(rhs))) {
        return false;
      }
      return (lhs.varIndex == rhs.varIndex);
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const VarRequest &obj);
    template <bool Nested = false>
    void print(llvm::raw_ostream &os) const;
  };

  enum Space : std::int32_t {
    first,
    Scratch = first, //! TODO "scratch" ?
    Spill,           //! TODO "RAM", "L2" ?
    limit,           //! TODO
    NA = limit       // TODO rm
  };

  friend Space operator++(Space &e) { return (e = static_cast<Space>(e + 1)); }

  friend std::string to_string(Space e) {
    switch (e) {
    case Space::Scratch:
      return "scratch";
    case Space::Spill:
      return "spill";
    default:
      return "NA";
    }
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Space &obj) {
    return os << to_string(obj);
  }

  using VariableBindings = llvm::DenseMap<IndexT, Space>;
  using VariableIndexSet = llvm::DenseSet<IndexT>;
  using RequestIndexSet = llvm::SmallSet<IndexT, 4>;

  struct Variable {
    std::array<RequestIndexSet, Space::limit> domain = {};
    Space placement =
        Space::NA; //! `domain[placement]` is valid/part of the solution

    friend bool operator==(const Variable &lhs, const Variable &rhs) {
      if (&lhs == &rhs) {
        return true;
      }
      if (lhs.placement != rhs.placement) {
        return false;
      }
      return (lhs.domain == rhs.domain);
    }
  };

  struct Problem; // forward

  // TODO this is kind of ugly
  struct VariableBuilder {
    IndexT request(Space space, AllocSizeT size, SequenceT first,
                   SequenceT last) {
      const auto reqIndex = (parent->requests.size() + requests.size());
      requests.emplace_back(Request{{first, last}, size, -1});
      spaces.emplace_back(space);
      return reqIndex;
    }

    VariableBuilder &place(Space space) {
      placement = space;
      return (*this);
    }

    VariableBuilder &bind(Space space) {
      const auto varIndex = static_cast<IndexT>(parent->variables.size());
      parent->bound.insert(varIndex);
      return place(space);
    }

  private:
    friend struct Problem;

    VariableBuilder(Problem &parent) : parent(&parent) {}

    // Complete new variable definition.
    IndexT add() {
      const auto varIndex = static_cast<IndexT>(parent->variables.size());
      Variable &variable = parent->variables.emplace_back();

      variable.placement = placement;

      TT_debug(requests.size() == spaces.size());
      for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto reqIndex = static_cast<IndexT>(parent->requests.size());
        parent->requests.emplace_back(
            VarRequest{std::move(requests[i]), varIndex});
        variable.domain[spaces[i]].insert(reqIndex);
      }

      return varIndex;
    }

    llvm::SmallVector<Request> requests;
    llvm::SmallVector<Space> spaces;
    Problem *parent;
    Space placement = Space::NA;
  };

  // Copy-constructible, copy-assignable.
  struct Problem {
    llvm::SmallVector<VarRequest>
        requests; // indexed into by `Variable::domain[*]`
    llvm::SmallVector<Variable> variables;
    VariableIndexSet bound;

    const Variable &variable(IndexT varIndex) const {
      TT_debug_limit(varIndex, variables.size());
      return variables[varIndex];
    }

    const VarRequest &request(IndexT reqIndex) const {
      TT_debug_limit(reqIndex, requests.size());
      return requests[reqIndex];
    }

    bool empty() const { return variables.empty(); }

    bool valid() const;

    friend bool operator==(const Problem &lhs, const Problem &rhs) {
      return (lhs.requests == rhs.requests) &&
             (lhs.variables == rhs.variables) && (lhs.bound == rhs.bound);
    }

    Variable &variable(IndexT varIndex) {
      return const_cast<Variable &>(
          (const_cast<const Problem *>(this))->variable(varIndex));
    }

    VarRequest &request(IndexT reqIndex) {
      return const_cast<VarRequest &>(
          (const_cast<const Problem *>(this))->request(reqIndex));
    }

    template <typename /* void(VariableBuilder &) */ F>
    IndexT def(F &&requests) {
      VariableBuilder builder{*this};
      requests(builder);
      return builder.add();
    };

    void reset(Space placement = Space::NA) {
      for (int32_t varIndex = 0;
           varIndex < static_cast<int32_t>(variables.size()); ++varIndex) {
        Variable &variable = variables[varIndex];
        if (!bound.contains(varIndex)) {
          variable.placement = placement;
        }
        for (Space placement = Space::first; placement < Space::limit;
             ++placement) {
          for (const auto index : variable.domain[placement]) {
            requests[index].offset = -1;
          }
        }
      }
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const Problem &obj);
  };

  /// Descriptor of allocation and verification outcomes.
  struct AllocateStats {
    /// Largest single-record memory size within an allocation problem.
    AllocSizeT maxSize;
    /// "Memory usage" is the minimum amount of memory needed to satisfy
    /// the allocation problem (also known as "makespan" in literature).
    AllocSizeT memUsage;
    /// "Load" is the sum of sizes of all requests live at a given
    /// logical instant. This is the maximum such value across the
    /// entire logical time range. (0 indicates N/A)
    AllocSizeT maxLoad;

    AllocateStats(AllocSizeT maxSize, AllocSizeT memUsage, AllocSizeT maxLoad)
        : maxSize(maxSize), memUsage(memUsage), maxLoad(maxLoad) {
      // TODO arg validation?
    }
    AllocateStats(const AllocateStats &rhs) = default;
    AllocateStats &operator=(const AllocateStats &rhs) = default;

    /// @return `memUsage/maxLoad` ratio if `maxLoad` is non-zero,
    /// double NaN otherwise
    double usageRatio() const {
      return maxLoad ? static_cast<double>(memUsage) / maxLoad
                     : std::numeric_limits<double>::quiet_NaN();
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const AllocateStats &obj);
  };

  struct SpillStats : public AllocateStats {
    std::int32_t stepsTaken;
    std::int32_t spillCount;

    SpillStats(std::int32_t stepsTaken, std::int32_t spillCount,
               const AllocateStats &spaceStats)
        : AllocateStats(spaceStats), stepsTaken(stepsTaken),
          spillCount(spillCount) {}

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const SpillStats &obj);
  };

  // TODO update/finish doc (planner modifies `problem`)
  /// Find a feasible allocation for Requests in `problem` (expected
  /// to have `size`, `first`, and `last` fields set).
  /// @return `Stats` with `maxSize` and `memUsage` fields set
  [[nodiscard]] static AllocateStats
  allocate(Problem &problem, Algorithm algorithm = Algorithm::Greedy);

  // TODO update/finish doc (planner modifies `problem`)
  // TODO doc that 'problem' is mutated
  [[nodiscard]] static SpillStats spillAllocate(Problem &problem,
                                                AllocSizeT memUsageLimit);

  /// Validate the allocation plan in `solution` for proper memory/liveness
  /// conflict resolution and TODO return a `SpaceStats` instance per each
  /// memory space.
  [[nodiscard]] static AllocateStats verify(const Problem &solution);

private:
  Planner() = default; // Non-instantiable.

  struct AnalysisStats {
    struct RequestMetrics {
      std::int32_t maxConflictSize = 0;
      bool inContentionGroup = false;
    };
    llvm::SmallVector<RequestMetrics> requestMetrics;
  };

  // TODO doc
  // this implicitly allocates within SCRATCH
  static AnalysisStats analyze(const Problem &solution, AllocSizeT watermark);
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Planner::Request &obj) {
  obj.print<false>(os);
  return os;
}

template <bool Nested>
void Planner::Request::print(llvm::raw_ostream &os) const {

  if (!Nested) {
    os << '{';
  }
  os << size << " [" << first << ", " << last << "] @ " << offset;
  if (!Nested) {
    os << '}';
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Planner::VarRequest &obj) {
  obj.print<false>(os);
  return os;
}

template <bool Nested>
void Planner::VarRequest::print(llvm::raw_ostream &os) const {

  if (!Nested) {
    os << '{';
  }
  os << '%' << varIndex << ": ";
  Request::print<true>(os);
  if (!Nested) {
    os << '}';
  }
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Planner::AllocateStats &obj) {
  os << "{max size = " << obj.maxSize << ", mem usage = " << obj.memUsage;
  if (obj.maxLoad) {
    os << ", max load = " << obj.maxLoad << " (ratio {"
       << (static_cast<double>(obj.memUsage) / obj.maxLoad) << "})";
  }
  return os << "}";
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Planner::SpillStats &obj) {
  os << "converged in " << obj.stepsTaken << " step(s), spilled "
     << obj.spillCount << " var(s): ";
  os << static_cast<const Planner::AllocateStats &>(obj);
  return os;
}

} // namespace mlir::tt::ttir::allocation

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_PLANNER_H
