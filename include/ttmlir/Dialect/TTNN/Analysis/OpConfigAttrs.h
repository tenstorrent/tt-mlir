// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

struct DefaultAttrs {
  Attribute attr;

  bool operator==(const DefaultAttrs &other) const {
    return attr == other.attr;
  }
  bool operator!=(const DefaultAttrs &other) const { return !(*this == other); }
  void dump() const {
    if (attr) {
      attr.dump();
    }
  }
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
    if (conv2dConfig) {
      conv2dConfig->dump();
    }
    if (deviceComputeKernelConfig) {
      deviceComputeKernelConfig->dump();
    }
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H
