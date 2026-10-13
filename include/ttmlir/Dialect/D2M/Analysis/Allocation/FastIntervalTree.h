// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_FASTINTERVALTREE_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_FASTINTERVALTREE_H

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <list>
#include <queue>
#include <type_traits>

//===----------------------------------------------------------------------===//
/// This implements a structure for both interval stabbing and intersection
/// queries as described in [Schmidt 2010] that is asymptotically optimal both
/// in time and space. It is particularly suitable for working with interval
/// bounds that are in some compact [0, N) range, e.g. op positions within an IR
/// scope. Is uses O(N) memory, O(NlogN) preprocessing time, and guaranteed O(K)
/// query times, where `K` is the number of intervals in the result. (This tight
/// bound is due to the query algorithm never visiting nodes that are not in the
/// result. In theory, this is great for very sparse interval sets.)
///
/// Additionally, all queries return found intervals in lexicographic order
/// (`a < b` if `a.left < b.left || (a.left == b.left && a.right < b.right)`)
/// at no extra cost and the overall interval set is stored in non-decreasing
/// `left` order, both properties useful in applications.
///
/// @see [Schmidt 2010] "Interval Stabbing Problems in Small Integer Ranges"
/// (https://www.researchgate.net/profile/Jens-Schmidt-4/publication/221543344_Interval_Stabbing_Problems_in_Small_Integer_Ranges/links/5500986e0cf2d61f820efe94/Interval-Stabbing-Problems-in-Small-Integer-Ranges.pdf)
//===----------------------------------------------------------------------===//
namespace mlir::tt::d2m::allocation {

template <typename, typename>
class FastIntervalTree; // Forward.

// Empty type used internally for interval type with no `value` payload field.
struct EmptyValue {};

//===----------------------------------------------------------------------===//
namespace detail {

// Base class to allow EBO for `FastIntervalTree` instantiations that
// don't need a value payload.
template <typename ValueT>
struct IntervalBase {

  ValueT value;

  IntervalBase(ValueT value) : value(value) {}
};

template <>
struct IntervalBase<EmptyValue> {
  IntervalBase() = default;
  IntervalBase(EmptyValue) {};
};
//===----------------------------------------------------------------------===//
// A tree node type representing a closed [`left`, `right`] interval as well as
// an optional `value` payload field.
//
// The tree structure has the following structure (full details in [Schmidt
// 2010]):
//
//  - `left` endpoints are (logically) "de-duplicated" such that the main query
// code path can work with unique `left` values;
//
//  - a node's parent is the unique node that has the rightmost `left` of all
//  thus unique'd nodes that cover this node (`a` covers `b` if `b` is a
//  subrange of `a`);
//
//  - a node can have an arbitrary number of children which are sorted in
//  increasing order of their `left` endpoints.
//
// Parent/child and sibling node links are implemented via "short pointers",
// i.e. `IndexT` indices into an array-like container.
template <typename SequenceT, typename ValueT, typename IndexT>
class Interval : public IntervalBase<ValueT> {

  static_assert(std::is_signed_v<SequenceT>);
  static_assert(std::is_signed_v<IndexT>);

  using Base = IntervalBase<ValueT>;

public:
  SequenceT left;
  SequenceT right;

  template <typename... Args>
  Interval(SequenceT left, SequenceT right, Args &&...args)
      : Base(std::forward<Args>(args)...), left(left), right(right),
        parent_(-1), leftSibling_(-1), rightmostChild_(-1) {}

  bool stabs(SequenceT point) const {
    return (left <= point && point <= right);
  }

  bool overlaps(SequenceT l, SequenceT r) const {
    return !((left > r) || (right < l));
  }

  bool overlaps(const Interval &iv) const {
    return overlaps(iv.left, iv.right);
  }

  bool covers(SequenceT l, SequenceT r) const {
    return (left <= l && right >= r);
  }

  bool covers(const Interval &iv) const { return covers(iv.left, iv.right); }

  // Lexicographic less.
  friend bool operator<(const Interval &lhs, const Interval &rhs) {
    return (lhs.left < rhs.left) ||
           ((lhs.left == rhs.left) && (lhs.right < rhs.right));
  }

  friend std::string to_string(const Interval &obj) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << obj;
    return s;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Interval &obj) {
    return os << '[' << obj.left << ", " << obj.right << ']';
  }

private:
  template <typename, typename>
  friend class allocation::FastIntervalTree;

  // Parent/child and sibling links, negative values are "null" markers:

  IndexT parent_;
  IndexT leftSibling_;
  IndexT rightmostChild_;
};
//===----------------------------------------------------------------------===//
// A query traversal begins at a certain leaf node (`Start` or `Start2` in the
// paper) that is pre-computed as a function on the query input. `Start2` is
// only needed if interval intersection API is enabled.
template <typename IndexT, bool IntervalIntersectionAPI = false>
struct StartData {
  IndexT start;
};

template <typename IndexT>
struct StartData<IndexT, true> {
  IndexT start;
  IndexT start2;
};
//===----------------------------------------------------------------------===//
// Iterable and random-accessible type returned by the
// `FastIntervalTree::query(...)` API.
template <typename IntervalT>
class IntervalList {
public:
  std::size_t size() const { return intervals_.size(); }

  bool empty() const { return (size() == 0); }

  // Correct lexicographic ordering of results is the (lazy) inverse
  // of the stack unwind logic in `FastIntervalTree::queryImpl()`:

  auto begin() const { return std::make_reverse_iterator(intervals_.end()); }
  auto end() const { return std::make_reverse_iterator(intervals_.begin()); }

  template <bool CHECK_BOUNDS = true>
  const IntervalT &at(std::size_t index) const {
    if constexpr (CHECK_BOUNDS) {
      TT_assert_within(index, size());
    }
    return *intervals_[intervals_.size() - 1 - index];
  }

  const IntervalT &operator[](std::size_t index) const {
    return at<false>(index);
  }

private:
  template <typename, typename>
  friend class allocation::FastIntervalTree;

  llvm::SmallVector<const IntervalT *> intervals_;
};

} // namespace detail
//===----------------------------------------------------------------------===//
/// Default trait type for `FastIntervalTree`. Any type `T` can be used as long
/// as it has the following valid expressions:
///  - `T::SequenceT`, (signed integral) type of endpoint values;
///  - `T::IndexT`, (signed integral) type of node links, must be wide enough to
///    support the expected tree size;
///  - `T::hasIntervalIntersectionAPI`, a compile-time bool flag for whether the
///    instantiated tree should enable the interval intersection API.
template <bool IntervalIntersectionAPI = false>
struct FastIntervalTreeTraits {
  using SequenceT = std::int32_t;
  using IndexT = std::int32_t;
  static constexpr bool hasIntervalIntersectionAPI = IntervalIntersectionAPI;
};
//===----------------------------------------------------------------------===//
/// A data structure for a set of closed intervals that supports efficient
/// queries for intervals intersecting a given point or a given interval.
/// Intervals can optionally contain a value of some payload type `ValueT`.
///
/// @see [Schmidt 2010] cited above
///
/// Synopsis:
/// @code
///  using Tree = FastIntervalTree<>;
///  Tree tree;
///  tree.insert(10, 101);
///  tree.insert(20, 201);
///  ...
///  tree.create();
///
///  auto rs1 = tree.query(15);    // intervals stabbed by point 15
///  for (const Tree::IntervalT * iv : rs1) { ... }
///
///  auto rs2 = tree.query(5, 30); // intervals intersecting interval [5, 30]
///  for (const Tree::IntervalT * iv : rs2) { ... }
/// @endcode
///
/// @tparam ValueT payload type, accessible as `value` field on the tree's
/// `IntervalT` type (elided if left at `EmptyValue` default)
/// @tparam Traits several parameters controlling the instantiated
/// implementation, @see FastIntervalTreeTraits
///
/// @note By default, the interval intersection query part of the API is not
/// enabled because it uses some extra memory not needed otherwise -- enable
/// it via the appropriate tree trait (`Traits::hasIntervalIntersectionAPI`).
///
template <typename ValueT = EmptyValue,
          typename Traits = FastIntervalTreeTraits<>>
class FastIntervalTree : noncopyable {
  using traits = Traits;
  // TODO(vroubtsov) allocator/obj pool for more efficient repeated querying
public:
  using SequenceT = typename traits::SequenceT;
  using IndexT = typename traits::IndexT;
  static constexpr bool hasIntervalIntersectionAPI =
      traits::hasIntervalIntersectionAPI;

  using IntervalT = detail::Interval<SequenceT, ValueT, IndexT>;
  using IntervalListT = detail::IntervalList<IntervalT>;

  using IntervalContainerT = llvm::SmallVector<IntervalT>;
  using reverse_iterator = typename IntervalContainerT::const_iterator;

  /// @param limit exclusive upper bound for the tree query domain `[0, limit)`.
  ///
  /// The default is a safe "infinite" value and there is no performance
  /// overhead for overshooting the range of endpoints in actual data. However,
  /// if known a priori a tight bound can be provided at construction time and
  /// will result in extra error checking during the `insert()`s.
  explicit FastIntervalTree(SequenceT limit = kDefaultLimit) : limit_{limit} {
    TT_assert(limit > 0);
  }

  /// @return count of intervals in the tree
  IndexT size() const {
    TT_debug(constructionCompleted());
    return (intervals_.size() - 2);
  }

  /// The tree is a container of `IntervalT`s which are stored in order of
  /// non-decreasing `left` endpoints.
  /// @return iterator range for the interval set in this guaranteed order
  llvm::iterator_range<reverse_iterator> intervals() const {
    TT_debug(constructionCompleted());
    return {std::next(intervals_.begin()), std::prev(intervals_.end())};
  }

  /// @return list of intervals intersecting `point`, in lexicographic order
  IntervalListT query(SequenceT point) const {
    TT_debug(constructionCompleted());
    TT_assert_limit(point, limit_);

    IntervalListT result;

    if (point >= extent_) {
      return result;
    }

    // Get the `Start(point)` leaf node that initiates the tree traversal:

    const IndexT leaf = start_[point].start;
    if (leaf == 0) {
      return result; // Fast path for known empty returns.
    }
    TT_debug(intervals_[leaf].stabs(point));

    traverse(point, leaf, result);

    return result;
  }

  /// @return list of intervals intersecting interval `[left, right]`, in
  /// lexicographic order
  template <bool _ = hasIntervalIntersectionAPI>
  auto query(SequenceT left,
             SequenceT right) const -> std::enable_if_t<_, IntervalListT> {
    TT_debug(constructionCompleted());
    TT_assert_limit(left, limit_);
    TT_assert_open_range(right, left, limit_);

    IntervalListT result;

    if (left >= extent_) {
      return result;
    }

    // For interval intersection, the starting leaf node is computed based
    // on `Start(left)` and `Start2(right)`:

    const IndexT j = start_[left].start;
    const IndexT j2 = start_[std::min<SequenceT>(right, extent_ - 1)].start2;

    const IntervalT &iv = intervals_[j];
    const IntervalT &iv2 = intervals_[j2];

    const IndexT leaf = (iv < iv2 && iv2.right >= left) ? j2 : j;
    TT_debug(intervals_[leaf].overlaps(left, right));

    traverse(left, leaf, result);

    return result;
  }

  /// @return exclusive upper bound on allowed interval endpoints as provided
  /// at construction time
  SequenceT limit() const { return limit_; }

  /// Add interval `[left, right]` during the tree building phase. If payload
  /// `ValueT` is anything other than `EmptyValue`, pass it as the 3rd
  /// parameter.
  template <typename... Args>
  auto insert(SequenceT left, SequenceT right,
              Args &&...value) -> std::enable_if_t<(sizeof...(Args) <= 1)> {
    TT_debug(!constructionCompleted());
    TT_assert_limit(left, limit_);
    TT_assert_open_range(right, left, limit_);

    const SequenceT newExtent = std::max(extent_, right);
    TT_assert_limit(newExtent, limit_);
    extent_ = newExtent;

    intervals_.emplace_back(left, right, std::forward<Args>(value)...);
  }

  /// Invoke to finish building the tree.
  void create() {
    TT_debug(!constructionCompleted());

    intervals_.emplace_back(-1, limit_, ValueT{});     // Add root node.
    intervals_.emplace_back(limit_, limit_, ValueT{}); // Add sentinel node.

    ++extent_; // Make this bound exclusive.

    // Sort `intervals` using primary order of increasing `left` and secondary
    // order of *decreasing* `right`. This naturally arranges them into
    // `Smaller` sublists without needing to use dedicated `Smaller` links
    // as seems to be done in the paper.
    std::sort(intervals_.begin(), intervals_.end(),
              [&](const IntervalT &lhs, const IntervalT &rhs) {
                return (lhs.left < rhs.left) ||
                       ((lhs.left == rhs.left) && (lhs.right > rhs.right));
              });
    // The root and the sentinel stay in first/last slots by design:
    TT_debug((intervals_.front().left < 0 && intervals_.back().left == limit_));

    // Do a sweep [0, limit_) to set all tree links and fill in `start`:

    using EventListT = llvm::SmallVector<IndexT>;

    // Populate `events`: `events[q]` is a list of interval open/close events
    // at sequence point `q`, in order of strictly increasing `left` endpoints:
    llvm::SmallVector<EventListT> events(extent_);

    // Iterate over the sequence of unique `left`s that are the heads of
    // `Smaller` sublists:
    SequenceT prev = -1;
    for (IndexT i = 1; i < static_cast<IndexT>(intervals_.size()) - 1; ++i) {
      const IntervalT &iv = intervals_[i];
      if (iv.left != prev) {
        prev = iv.left;

        if (iv.right != iv.left) {
          events[iv.right].push_back(i);
        }
        events[iv.left].push_back(i); // at most one of these per `q`
      }
    }

    start_.resize(extent_); // allocate `Start(q)/Start2(q)`

    // A list of intervals stabbing current position of the sweep line:

    using IndexList = std::list<IndexT>;

    IndexList active;
    llvm::SmallVector<typename IndexList::iterator> in_active(
        intervals_.size());

    active.push_back(0);

    for (SequenceT q = 0; q < extent_; ++q) {
      TT_debug(!active.empty()); // there is always at least the root
      // All intervals in `active` are stabbing `q`, the latest of them
      // is thus the `Start(q)` candidate unless overwritten by an open event
      // later:
      start_[q].start = active.back();

      // Processing events in reverse order so we see an interval open
      // before close if it is length-1:
      for (auto e = events[q].rbegin(); e != events[q].rend(); ++e) {
        const IndexT i = *e;
        TT_assert_open_range(i, 1, intervals_.size());
        const IntervalT &iv = intervals_[i];

        if (q == iv.left) {
          // This is an interval open event, the (unique) interval opening at
          // `q` automatically becomes the rightmost stabber of `q`, i.e.
          // `Start(q)`:
          start_[q].start = i;
          in_active[i] = active.insert(active.end(), i);
        }

        if (q == iv.right) {
          // This is an interval close event, at which point we have all
          // the information necessary to know the interval's parent: it is
          // its predecessor in `active` (which can be the root):
          TT_debug(in_active[i] != active.begin());
          const auto predecessor = std::prev(in_active[i]);

          appendChild(*predecessor, i);
          active.erase(in_active[i]);
        }
      }
    }
    TT_debug(active.size() == 1u);

    if constexpr (hasIntervalIntersectionAPI) {
      // To populate `start2` (`start2[q]` is the rightmost interval with `left
      // <= q`), do another sweep over the sequence of unique `left`s:

      SequenceT q = 0;
      IndexT j = 0;

      SequenceT prev = -1;
      for (IndexT i = 1; i < static_cast<IndexT>(intervals_.size()) - 1; ++i) {
        const IntervalT &iv = intervals_[i];
        if (iv.left != prev) {
          prev = iv.left;

          for (; q < prev; ++q) {
            start_[q].start2 = j;
          }
          j = i;
        }
      }
      for (; q < extent_; ++q) {
        start_[q].start2 = j;
      }
    }
  }

#if defined(TT_BUILD_DEBUG)
  // Internal consistency check invoked by Debug-built tests.
  bool check() const {
    TT_assert(extent_ <= limit_);

    SequenceT prev = -1;
    for (IndexT i = 1; i < static_cast<IndexT>(intervals_.size()) - 1; ++i) {
      const IntervalT &iv = intervals_[i];
      if (iv.left != prev) {
        prev = iv.left;

        TT_assert(iv.parent_ >= 0);
        const IntervalT &parent = intervals_[iv.parent_];
        TT_assert(parent.covers(iv));

        if (iv.leftSibling_ >= 0) {
          const IntervalT &leftSibling = intervals_[iv.leftSibling_];
          TT_assert(leftSibling.left < iv.left);
          TT_assert(!leftSibling.covers(iv));
        }

        if (iv.rightmostChild_ >= 0) {
          const IntervalT &rightmostChild = intervals_[iv.rightmostChild_];
          TT_assert(iv.covers(rightmostChild));
        }
      }
    }

    for (SequenceT q = 0; q < extent_; ++q) {
      // `Start(q)` points to a valid parent node (which could be root):
      TT_assert(start_[q].start >= 0);
      TT_assert(intervals_[start_[q].start].stabs(q));

      if constexpr (hasIntervalIntersectionAPI) {
        // `Start2(q)` points to a valid parent node (which could be root):
        TT_assert(start_[q].start2 >= 0);
        TT_assert(intervals_[start_[q].start2].left <= q);
      }
    }

    return true;
  }
#endif // TT_BUILD_DEBUG

private:
  using StartData = detail::StartData<IndexT, hasIntervalIntersectionAPI>;

  // Collect all intervals stabbing `q` via a tree traversal starting
  // from leaf node `i`.
  //
  // Symbols/names used in code comments follow [Schmidt 2010].
  void traverse(SequenceT q, IndexT i, IntervalListT &result) const {
    std::deque<IndexT> queue;

    // Start by computing path `P` from `i` to root (not including the root):
    while (i > 0) {
      queue.push_front(i);
      i = intervals_[i].parent_;
    }

    // Now drain `queue`:
    while (!queue.empty()) {
      const IndexT j = queue.back();
      queue.pop_back();
      TT_debug(j > 0);

      const IntervalT *const jiv = &intervals_[j];
      result.intervals_.push_back(jiv);

      // If interval node @ `i` has a `Smaller` sublist,
      // all stabbed intervals within it can be added to `result`
      // directly.
      // This sublist is a (possibly empty) sequence of interval slots
      // starting at `i+1` and having the same `left`. At construction
      // time this sublist was also ordered by decreasing `right`, which
      // we take advantage of here.
      //
      // Note that using a sentinel node at the end of `intervals_` also
      // saves us an are-we-at-the-end-of-vector check.

      for (IndexT s = j + 1; true; ++s) {
        // 's' is a valid index of a non-root node:
        TT_debug_open_range(s, 1, intervals_.size());
        const IntervalT *const smaller = &intervals_[s];

        if ((smaller->left != jiv->left) || (smaller->right < q)) {
          break;
        }
        result.intervals_.push_back(smaller);
      }

      // Compute the rightmost path `R(iv)`, which is the path
      // from `jiv`s left sibling (if it exists and is stabbed) to
      // the rightmost stabbed node in the subtree rooted at that sibling.
      // Note that the path contains only stabbed nodes so we stop at the
      // first one that isn't.

      IndexT r = jiv->leftSibling_;
      while (r > 0) {
        const IntervalT *const riv = &intervals_[r];
        if (riv->right < q) {
          break;
        }
        queue.push_back(r);
        r = riv->rightmostChild_;
      }
    }

    // `result.intervals` are in reverse order to what the API promises;
    // the correct order is restored lazily, via reverse iteration on `result`
    // by the caller.
  }

  // Check that create() has been called.
  bool constructionCompleted() const {
    return (!intervals_.empty() && intervals_.front().left < 0);
  }

  // Make `child` the new right-most child of `parent`.
  void appendChild(IndexT parent, IndexT child) {
    TT_debug(parent >= -1);
    TT_debug(child >= 0);

    IntervalT &p = intervals_[parent];
    IntervalT &c = intervals_[child];

    c.parent_ = parent;

    c.leftSibling_ = p.rightmostChild_;
    p.rightmostChild_ = child;
  }

  IntervalContainerT intervals_;       ///< Tree of intervals
  llvm::SmallVector<StartData> start_; ///< `Start` and `Start2` data
  SequenceT extent_{0}; ///< "Effective" limit (1 + largest endpoint inserted)
  SequenceT limit_;     ///< Query domain limit provided at construction time

  static constexpr SequenceT kDefaultLimit =
      (std::numeric_limits<SequenceT>::max() - /* root and sentinel */ 2);
};

} // namespace mlir::tt::d2m::allocation

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_ALLOCATION_FASTINTERVALTREE_H
