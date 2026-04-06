// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <functional>
#include <iterator>

namespace mlir {
namespace tt {
namespace ttnn {

/// Search space for Conv3d blocking parameters.
///
/// Block semantics (from conv3d_program_factory.cpp):
///   - C_in_block:  input channels per iteration.  Smaller → smaller weight
///                   shard in L1.  Must divide C_in, be L1-aligned.
///                   0 means "full C_in" (default, fastest).
///   - C_out_block: output channels per matmul.  Smaller → smaller output CB.
///                   Must be tile-aligned (32) and divide padded_C_out.
///                   0 means "full padded_C_out" (default, fastest).
///   - T/H/W_out_block: spatial output blocking.  num_patches = T×H×W
///                   determines the matmul M dimension.  Larger = faster but
///                   needs more L1 for the input shard and intermediates.
///                   Default is 1×1×1 (minimum L1).
///
/// Constructed from a Conv3dOp and its base config.  Values that are already
/// fixed in the base config are not searched.  Search order is large-to-small
/// (0 first where applicable) so the generator yields the most performant
/// configs first.
struct Conv3dConfigSearchSpace {
  llvm::SmallVector<uint32_t> cInBlock;
  llvm::SmallVector<uint32_t> cOutBlock;
  llvm::SmallVector<uint32_t> tOutBlock;
  llvm::SmallVector<uint32_t> wOutBlock;
  llvm::SmallVector<uint32_t> hOutBlock;

  /// Build from a Conv3dOp and the base config already set on it.
  Conv3dConfigSearchSpace(ttnn::Conv3dOp conv3dOp, Conv3dConfigAttr baseConfig);

  bool isCInBlockSetForSearch() const { return !cInBlock.empty(); }
  bool isCOutBlockSetForSearch() const { return !cOutBlock.empty(); }
  bool isTOutBlockSetForSearch() const { return !tOutBlock.empty(); }
  bool isWOutBlockSetForSearch() const { return !wOutBlock.empty(); }
  bool isHOutBlockSetForSearch() const { return !hOutBlock.empty(); }

  bool isAnyFieldSetForSearch() const {
    return isCInBlockSetForSearch() || isCOutBlockSetForSearch() ||
           isTOutBlockSetForSearch() || isWOutBlockSetForSearch() ||
           isHOutBlockSetForSearch();
  }
};

/// A cycling cursor over a fixed sequence of uint32_t values.
///
/// Supports `operator*` for the current value and `advance()` which
/// moves to the next value, cycling back to the start when exhausted.
/// `advance()` returns true on wrap-around, enabling odometer-style
/// carry logic in the config generator.
class CyclingCursor {
public:
  explicit CyclingCursor(const llvm::SmallVector<uint32_t> &values)
      : values(values) {
    assert(!values.empty() && "Search space must not be empty");
  }

  uint32_t operator*() const { return values[index]; }

  /// Advance to the next value. Returns true if wrapped around to the start.
  bool advance() {
    if (++index >= values.size()) {
      index = 0;
      return true;
    }
    return false;
  }

private:
  llvm::SmallVector<uint32_t> values;
  size_t index = 0;
};

/// Generates Conv3d config combinations from a search space.
///
/// Supports C++ input-iterator semantics for range-based for loops:
///
///   Conv3dConfigGenerator gen(&op, base, space, filter);
///   for (Conv3dConfigAttr cfg : gen) { ... }
///
/// Each dereference yields the current config; incrementing advances the
/// odometer. The sentinel (end) is a default-constructed iterator.
class Conv3dConfigGenerator {
public:
  Conv3dConfigGenerator(
      ttnn::Conv3dOp *op, Conv3dConfigAttr baseConfig,
      const Conv3dConfigSearchSpace &space,
      std::function<bool(const Conv3dConfigAttr &)> filterOutFn);

  // --- C++ input iterator ---------------------------------------------------

  class iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using value_type = Conv3dConfigAttr;
    using difference_type = std::ptrdiff_t;
    using pointer = const Conv3dConfigAttr *;
    using reference = const Conv3dConfigAttr &;

    // Sentinel (end) iterator.
    iterator() = default;

    reference operator*() const { return current; }
    pointer operator->() const { return &current; }

    iterator &operator++();
    iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator &lhs, const iterator &rhs) {
      return !lhs.current && !rhs.current;
    }
    friend bool operator!=(const iterator &lhs, const iterator &rhs) {
      return !(lhs == rhs);
    }

  private:
    friend class Conv3dConfigGenerator;
    explicit iterator(Conv3dConfigGenerator *gen);

    Conv3dConfigGenerator *gen = nullptr;
    Conv3dConfigAttr current = nullptr;
  };

  iterator begin() { return iterator(this); }
  static iterator end() { return iterator(); }

private:
  Conv3dConfigAttr generateCurrent() const;
  void advanceOdometer();
  static Conv3dConfigAttr advanceToNextValid(Conv3dConfigGenerator *gen);

  /// One dimension of the search space: a cycling cursor over candidate values
  /// paired with a function that applies the current value to a config.
  struct ActiveFieldEntry {
    CyclingCursor cursor;
    std::function<Conv3dConfigAttr(Conv3dConfigAttr, uint32_t)> applyValue;

    ActiveFieldEntry(
        CyclingCursor &&cursor,
        std::function<Conv3dConfigAttr(Conv3dConfigAttr, uint32_t)> &&apply)
        : cursor(std::move(cursor)), applyValue(std::move(apply)) {}
  };

  [[maybe_unused]] ttnn::Conv3dOp *op;
  Conv3dConfigAttr baseConfig;
  Conv3dConfigSearchSpace searchSpace;
  llvm::SmallVector<ActiveFieldEntry> activeSearchFields;
  std::function<bool(const Conv3dConfigAttr &)> filterOutFn;
  bool isDone = false;
};

} // namespace ttnn
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGSEARCHSPACE_H
