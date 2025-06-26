// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

/*
- This file contains all different implementations of OpConfig::OpSpecificAttrs.
- Please add a new struct here if you need to support a new op.
- Each struct should provide three public methods:
  1. bool operator==(const T &) const { ... }
  2. bool operator!=(const T &) const { ... }
  3. void dump() const { ... }
*/

// This is useful for detecting whether OpConfig::OpSpecificAttrs has been
// initialized or not.
struct UninitializedAttrs {
  bool operator==(const UninitializedAttrs &) const { return true; }
  bool operator!=(const UninitializedAttrs &) const { return false; }
  void dump() const {}
};

struct Conv2dAttrs {
  std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig;
  std::optional<mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
      deviceComputeKernelConfig;

  bool operator==(const Conv2dAttrs &other) const {
    return conv2dConfig == other.conv2dConfig &&
           deviceComputeKernelConfig == other.deviceComputeKernelConfig;
  }
  bool operator!=(const Conv2dAttrs &other) const { return !(*this == other); }
  void dump() const {
    if (conv2dConfig.has_value() && conv2dConfig.value()) {
      conv2dConfig->dump();
    }
    if (deviceComputeKernelConfig.has_value() &&
        deviceComputeKernelConfig.value()) {
      deviceComputeKernelConfig->dump();
    }
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H
