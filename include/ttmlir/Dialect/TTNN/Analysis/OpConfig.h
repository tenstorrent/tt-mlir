// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H

#include "OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include <variant>

namespace mlir::tt::ttnn {

struct OpConfig {
  // Desired output layout for the op.
  TTNNLayoutAttr outputLayout;
  // Holds attributes for the op. Note: Please prevent using the DefaultAttrs
  // unless it's absolutely necessary. For most cases, a new type should be
  // added to the following std::variant.
  using OpSpecificAttrs =
      std::variant<DefaultAttrs, Conv2dAttrs, Uninitialized>;
  OpSpecificAttrs opSpecificAttrs;

  // Default Config Constructors:
  OpConfig() = default;
  OpConfig(TTNNLayoutAttr outputLayout)
      : outputLayout(outputLayout), opSpecificAttrs(Uninitialized{}) {}
  OpConfig(TTNNLayoutAttr outputLayout, OpSpecificAttrs attrs)
      : outputLayout(outputLayout), opSpecificAttrs(std::move(attrs)) {}
  OpConfig(TTNNLayoutAttr outputLayout, Attribute attr)
      : outputLayout(outputLayout), opSpecificAttrs(DefaultAttrs{attr}) {}
  OpConfig(TTNNLayoutAttr outputLayout, DefaultAttrs config)
      : outputLayout(outputLayout), opSpecificAttrs(std::move(config)) {}
  // Op Specific Constructors:
  OpConfig(TTNNLayoutAttr outputLayout, Conv2dAttrs config)
      : outputLayout(outputLayout), opSpecificAttrs(std::move(config)) {}

  // Some utility functions:
  bool attrIsUninitialized() const {
    // This function is helpful to determine whether opSpecificAttrs has been
    // initialized with an actual T or not. If this function returns true, it's
    // safe to ignore/override the content of opSpecificAttrs. I decided to
    // provide this function instead of using std::optional<std::variant<T,...>>
    // because I don't want the caller to worry about std::nullopt.
    return std::holds_alternative<Uninitialized>(opSpecificAttrs);
  }
  bool operator==(const OpConfig &other) const {
    if (outputLayout != other.outputLayout) {
      return false;
    }
    // Compare variants using std::visit with a generic comparison
    return std::visit(
        [](const auto &lhs, const auto &rhs) -> bool {
          // This requires that both types are the same and have operator==
          // defined
          using T = std::decay_t<decltype(lhs)>;
          if constexpr (std::is_same_v<T, std::decay_t<decltype(rhs)>>) {
            return lhs == rhs; // Use the type's own operator==
          }
          return false; // Different types are not equal
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

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIG_H
