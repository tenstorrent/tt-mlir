// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_PLANNERIMPL_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_PLANNERIMPL_H

#include "ttmlir/Dialect/TTIR/Analysis/Allocation/Utils.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <string>

namespace mlir::tt::ttir::allocation {

// A base class used for nesting API types within `Planner` (via inheritance,
// since we don't have `using enum` until c++20).
class PlannerBase {
public:
  /// Enum for planning algorithms exposed by this API.
  /// @see Planner::allocate()
  /// @see Planner::spillAllocate()
  enum class Algorithm : std::int32_t {
    Simple, ///< Monotonic "bumper" allocator, does not support dealloc.
    Greedy  ///< Greedy-by-size allocator.
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

  /// Enum for variable placement spaces as understood by `Planner`, i.e.
  /// without referencing hardware specifics like "L1" or "DRAM", etc.
  enum class Space : std::int32_t {
    begin,           ///< Iteration helper.
    Scratch = begin, ///< Space targeted by allocate(), etc.
    Spill,           ///< Space where "spilled" variables go.
    end,             ///< Iteration helper.
    NA = end         ///< Value alias for expressing "unset/not available".
  };

  friend Space operator++(Space &e) {
    return (e = static_cast<Space>(ordinal(e) + 1));
  }
  friend bool operator<(const Space lhs, const Space rhs) {
    return ordinal(lhs) < ordinal(rhs);
  }

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
};

class PlannerImpl : public PlannerBase {
public:
  /// Type for address-like values.
  using AllocSizeT = std::int64_t;
  /// Type for liveness "logical time".
  using SequenceT = std::int32_t;
  /// Type for indexing into variable/request collections
  /// (should be more compact that `std::size_t`).
  using IndexT = std::int32_t;

  struct LiveRange {
    SequenceT first = -1; ///< (Inclusive) start of live range.
    SequenceT last = -1;  ///< (Inclusive) end of live range.

    bool valid() const { return 0 <= first && first <= last; }

    friend bool operator==(const LiveRange &lhs, const LiveRange &rhs) {
      return (lhs.first == rhs.first) && (lhs.last == rhs.last);
    }
  };

  /// An allocation descriptor, a region of memory usage `[offset, offset+size)`
  /// reserved over a liveness range `[first, last]`.
  struct Request : LiveRange {
    AllocSizeT size = -1;   ///< Requested memory size.
    AllocSizeT offset = -1; ///< Start memory address (negative means N/A).

    friend bool operator==(const Request &lhs, const Request &rhs) {
      if (!(static_cast<const LiveRange &>(lhs) ==
            static_cast<const LiveRange &>(rhs))) {
        return false;
      }
      return (lhs.size == rhs.size) && (lhs.offset == rhs.offset);
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const Request &obj) {
      obj.print<false>(os);
      return os;
    }

    template <bool Nested = false>
    void print(llvm::raw_ostream &os) const {
      if (!Nested) {
        os << '{';
      }
      os << this->size << " [" << this->first << ", " << this->last << "] @ "
         << this->offset;
      if (!Nested) {
        os << '}';
      }
    }
  };

  /// A `Request` extension for a request that is associated with a `Variable`.
  struct VarRequest : Request {
    IndexT varIndex = -1; ///< Parent variable (`Problem::variables` index).

    friend bool operator==(const VarRequest &lhs, const VarRequest &rhs) {
      if (!(static_cast<const Request &>(lhs) ==
            static_cast<const Request &>(rhs))) {
        return false;
      }
      return (lhs.varIndex == rhs.varIndex);
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const VarRequest &obj) {
      obj.print<false>(os);
      return os;
    }

    template <bool Nested = false>
    void print(llvm::raw_ostream &os) const {
      if (!Nested) {
        os << '{';
      }
      os << '%' << varIndex << ": ";
      Request::print<true>(os);
      if (!Nested) {
        os << '}';
      }
    }
  };

  using VariableBindings = llvm::DenseMap<IndexT, Space>;
  using VariableIndexSet = llvm::DenseSet<IndexT>;
  using RequestIndexSet = llvm::SmallSet<IndexT, 4>;
  using VariableDomain = std::array<RequestIndexSet, ordinal(Space::end)>;

  /// A `Variable` represents a discrete decision for which of `Space`s to
  /// be placed.
  struct Variable {
    /// This variable's requests, a (possibly empty) set of those for each
    /// possible placement.
    VariableDomain domain = {};
    /// Current placement space (indexes into `domain`).
    Space placement = Space::NA;

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

  struct Problem; // Forward.

  struct VariableBuilder {
    IndexT request(Space space, AllocSizeT size, SequenceT first,
                   SequenceT last) {
      const IndexT reqIndex = (parent->requests.size() + requests.size());
      requests.emplace_back(Request{{first, last}, size, -1});
      spaces.emplace_back(space);
      return reqIndex;
    }

    VariableBuilder &place(Space space) {
      placement = space;
      return (*this);
    }

    VariableBuilder &bind(Space space) {
      const IndexT varIndex = parent->variables.size();
      parent->bound.insert(varIndex);
      return place(space);
    }

  private:
    friend struct Problem;

    VariableBuilder(Problem &parent) : parent(&parent) {}

    // Complete new variable definition.
    IndexT add() {
      const IndexT varIndex = parent->variables.size();
      Variable &variable = parent->variables.emplace_back();

      variable.placement = placement;

      TT_debug(requests.size() == spaces.size());
      for (std::size_t i = 0; i < requests.size(); ++i) {
        const IndexT reqIndex = parent->requests.size();
        parent->requests.emplace_back(
            VarRequest{std::move(requests[i]), varIndex});
        variable.domain[ordinal(spaces[i])].insert(reqIndex);
      }

      return varIndex;
    }

    llvm::SmallVector<Request> requests;
    llvm::SmallVector<Space> spaces;
    Problem *parent;
    Space placement = Space::NA;
  };

  /// Top-level descriptor of a memory planning problem.
  /// @note Copy-constructible, copy-assignable.
  /// @see allocation::Tools for randomized testing, JSON serialization, etc
  struct Problem {
    /// All requests of all variables in the problem; indexed into by
    /// `Variable::domain[*]`.
    llvm::SmallVector<VarRequest> requests;
    /// All variables in the problem.
    llvm::SmallVector<Variable> variables;
    /// A subset of `variables` that are bound as far as spilling heuristics are
    /// concerned.
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

    /// Syntactic sugar helper for adding variables and requests to a `Problem`.
    /// Use as in
    /// @code
    ///   problem.def([](Planner::VariableBuilder &b) {
    ///     b.request(Planner::Space::Scratch, scratchRequest, 1, 2);
    ///     b.request(Planner::Space::Spill, 10 * scratchRequest, 1, 2);
    ///     ...can b.place(), b.bind(), etc...
    ///   });
    /// @endcode
    template <typename /* void(VariableBuilder &) */ F>
    IndexT def(F &&requests) {
      VariableBuilder builder{*this};
      requests(builder);
      return builder.add();
    };

    // Reset placements and offsets for all unbound variables.
    void reset(Space placement = Space::NA) {
      for (IndexT varIndex = 0;
           varIndex < static_cast<IndexT>(variables.size()); ++varIndex) {
        Variable &variable = variables[varIndex];
        if (!bound.contains(varIndex)) {
          variable.placement = placement;
        }
        for (Space placement = Space::begin; placement < Space::end;
             ++placement) {
          for (const auto index : variable.domain[ordinal(placement)]) {
            requests[index].offset = -1;
          }
        }
      }
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const Problem &obj);
  };

  /// Descriptor of allocation and spill/allocation outcomes.
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
        : maxSize(maxSize), memUsage(memUsage), maxLoad(maxLoad) {}
    AllocateStats(const AllocateStats &rhs) = default;
    AllocateStats &operator=(const AllocateStats &rhs) = default;

    /// @return `memUsage/maxLoad` ratio if `maxLoad` is non-zero,
    /// double NaN otherwise
    double usageRatio() const {
      return maxLoad ? static_cast<double>(memUsage) / maxLoad
                     : std::numeric_limits<double>::quiet_NaN();
    }

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const AllocateStats &obj) {
      os << "{max size = " << obj.maxSize << ", mem usage = " << obj.memUsage;
      if (obj.maxLoad) {
        os << ", max load = " << obj.maxLoad << " (ratio "
           << llvm::format("%.3f",
                           static_cast<double>(obj.memUsage) / obj.maxLoad)
           << ")";
      }
      return os << "}";
    }
  };

  /// Extends `AllocateStats` with spill stats.
  struct SpillStats : AllocateStats {
    /// Count of bisection steps that was needed to find the smallest
    /// set of variables to spill.
    std::int32_t stepsTaken;
    /// Count of free variables that needed to be placed into `Spill`.
    std::int32_t spillCount;

    SpillStats(std::int32_t stepsTaken, std::int32_t spillCount,
               const AllocateStats &spaceStats)
        : AllocateStats(spaceStats), stepsTaken(stepsTaken),
          spillCount(spillCount) {}

    friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                         const SpillStats &obj) {
      os << "converged in " << obj.stepsTaken << " step(s), spilled "
         << obj.spillCount << " var(s): ";
      os << static_cast<const AllocateStats &>(obj);
      return os;
    }
  };

protected:
  /// Request-level metrics as computed by analyze(). Variable-level
  /// aggregation is done by the caller.
  struct AnalysisStats {
    struct RequestMetrics {
      std::int32_t maxConflictSize = 0;
      bool inContentionGroup = false;
    };
    llvm::SmallVector<RequestMetrics> requestMetrics;
  };

  template <Algorithm Algorithm>
  static AllocateStats allocateImpl(Problem &problem);

  static AnalysisStats analyze(const Problem &solution, AllocSizeT watermark);
};

} // namespace mlir::tt::ttir::allocation
#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_ALLOCATION_PLANNERIMPL_H
