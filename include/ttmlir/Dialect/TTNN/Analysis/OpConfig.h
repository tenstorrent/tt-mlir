// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H

#include "OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/FormatVariadicDetails.h"

#include <variant>

namespace mlir::tt::ttnn {

struct OpConfig {
  // Desired output layout for the op.
  TTNNLayoutAttr outputLayout;
  // Holds attributes for the op. For most cases, a new type should be
  // added to the following std::variant.
  using OpSpecificAttrs = std::variant<UninitializedAttrs, Conv2dAttrs>;
  OpSpecificAttrs opSpecificAttrs;

  // Default Config Constructors.
  OpConfig() = default;
  OpConfig(TTNNLayoutAttr outputLayout)
      : outputLayout(outputLayout), opSpecificAttrs(UninitializedAttrs{}) {}
  OpConfig(TTNNLayoutAttr outputLayout, OpSpecificAttrs attrs)
      : outputLayout(outputLayout), opSpecificAttrs(std::move(attrs)) {}
  // Op Specific Constructors.
  OpConfig(TTNNLayoutAttr outputLayout, Conv2dAttrs config)
      : outputLayout(outputLayout), opSpecificAttrs(std::move(config)) {}

  // Some utility functions.
  bool isAttrUninitialized() const {
    // This function is helpful to determine whether opSpecificAttrs has been
    // initialized with an actual T or not. If this function returns true, it's
    // safe to ignore/override the content of opSpecificAttrs. This function
    // is provided so that the caller doesn't need to worry about std::nullopt.
    return std::holds_alternative<UninitializedAttrs>(opSpecificAttrs);
  }
  bool operator==(const OpConfig &other) const {
    if (outputLayout != other.outputLayout) {
      return false;
    }
    // Compare variants using std::visit with a generic comparison.
    return std::visit(
        [](const auto &lhs, const auto &rhs) -> bool {
          // This requires that both types are the same and have operator==
          // defined.
          using T = std::decay_t<decltype(lhs)>;
          if constexpr (std::is_same_v<T, std::decay_t<decltype(rhs)>>) {
            // Use the type's own operator==.
            return lhs == rhs;
          }
          return false; // Different types are not equal.
        },
        opSpecificAttrs, other.opSpecificAttrs);
  }
  bool operator!=(const OpConfig &other) const { return !(*this == other); }
  void dump() const {
    if (outputLayout) {
      outputLayout.dump();
    }
    std::visit([](const auto &config) { config.dump(); }, opSpecificAttrs);
  }
};

} // namespace mlir::tt::ttnn

// DenseMapInfo specialization for OpSpecificAttrs variant.
namespace llvm {
template <>
struct DenseMapInfo<mlir::tt::ttnn::OpConfig::OpSpecificAttrs> {
  using OpSpecificAttrs = mlir::tt::ttnn::OpConfig::OpSpecificAttrs;
  using UninitializedAttrs = mlir::tt::ttnn::UninitializedAttrs;
  using Conv2dAttrs = mlir::tt::ttnn::Conv2dAttrs;

  static inline OpSpecificAttrs getEmptyKey() {
    // Create a unique Conv2dAttrs with special pointer value that will never
    // occur in normal usage.
    static const intptr_t EmptyKeyVal = static_cast<intptr_t>(-1);
    Conv2dAttrs empty;
    empty.conv2dConfig = mlir::tt::ttnn::Conv2dConfigAttr::getFromOpaquePointer(
        reinterpret_cast<void *>(EmptyKeyVal));
    empty.deviceComputeKernelConfig = std::nullopt;
    return OpSpecificAttrs{std::move(empty)};
  }

  static inline OpSpecificAttrs getTombstoneKey() {
    // Create a unique Conv2dAttrs with special pointer value that will never
    // occur in normal usage.
    static const intptr_t TombstoneKeyVal = static_cast<intptr_t>(-2);
    Conv2dAttrs tombstone;
    tombstone.conv2dConfig =
        mlir::tt::ttnn::Conv2dConfigAttr::getFromOpaquePointer(
            reinterpret_cast<void *>(TombstoneKeyVal));
    tombstone.deviceComputeKernelConfig = std::nullopt;
    return OpSpecificAttrs{std::move(tombstone)};
  }

  static unsigned getHashValue(const OpSpecificAttrs &attrs) {
    // For special keys, return their unique hash values.
    if (attrs == getEmptyKey()) {
      return static_cast<unsigned>(-1);
    }

    if (attrs == getTombstoneKey()) {
      return static_cast<unsigned>(-2);
    }

    return std::visit(
        [](const auto &attr) -> unsigned {
          using T = std::decay_t<decltype(attr)>;
          if constexpr (std::is_same_v<T, UninitializedAttrs>) {
            return 0; // UninitializedAttrs all hash to same value.
          } else if constexpr (std::is_same_v<T, Conv2dAttrs>) {
            unsigned h1 = 0;
            unsigned h2 = 0;

            if (attr.conv2dConfig.has_value() && attr.conv2dConfig.value()) {
              // Hash the MLIR attribute using its internal hash.
              h1 = static_cast<unsigned>(
                  mlir::hash_value(attr.conv2dConfig.value()));
            }

            if (attr.deviceComputeKernelConfig.has_value() &&
                attr.deviceComputeKernelConfig.value()) {
              h2 = static_cast<unsigned>(
                  mlir::hash_value(attr.deviceComputeKernelConfig.value()));
            }

            // Combine hashes using LLVM's method.
            return hash_combine(h1, h2);
          }
          // Default case for unknown types.
          return 1;
        },
        attrs);
  }

  static bool isEqual(const OpSpecificAttrs &lhs, const OpSpecificAttrs &rhs) {
    // First, handle all special key combinations.
    if (lhs == getEmptyKey()) {
      return rhs == getEmptyKey();
    }

    if (lhs == getTombstoneKey()) {
      return rhs == getTombstoneKey();
    }

    if (rhs == getEmptyKey() || rhs == getTombstoneKey()) {
      return false;
    }

    // Use the existing operator== for normal values.
    return lhs == rhs;
  }
};

// Format provider specialization for OpConfig::OpSpecificAttrs variant.
template <>
struct format_provider<mlir::tt::ttnn::OpConfig::OpSpecificAttrs> {
  static void format(const mlir::tt::ttnn::OpConfig::OpSpecificAttrs &variant,
                     raw_ostream &os, StringRef options) {
    std::visit([&os](const auto &attr) { os << attr.toString(); }, variant);
  }
};
} // namespace llvm

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
