// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H

#include "mlir/IR/Attributes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "llvm/Support/raw_ostream.h"

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
  std::string toString() const { return "UninitializedAttrs"; }
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

  std::string toString() const {
    std::string result = "Conv2dAttrs{";
    if (conv2dConfig.has_value() && conv2dConfig.value()) {
      std::string attrStr;
      llvm::raw_string_ostream stream(attrStr);

      // Use MLIR's operator<< which calls attr.print(os)
      stream << conv2dConfig.value();
      stream.flush();

      result += "conv2dConfig=" + attrStr;
    } else {
      result += "conv2dConfig=<null>";
    }
    if (deviceComputeKernelConfig.has_value() &&
        deviceComputeKernelConfig.value()) {
      std::string attrStr;
      llvm::raw_string_ostream stream(attrStr);

      stream << deviceComputeKernelConfig.value();
      stream.flush();

      result += ", deviceComputeKernelConfig=" + attrStr;
    } else {
      result += ", deviceComputeKernelConfig=<null>";
    }
    result += "}";
    return result;
  }

  void dump() const { llvm::outs() << toString(); }
};

struct MatmulAttrs {
  std::optional<mlir::Attribute> matmulProgramConfig;
  std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig;

  bool operator==(const MatmulAttrs &other) const {
    return matmulProgramConfig == other.matmulProgramConfig &&
           computeKernelConfig == other.computeKernelConfig;
  }
  bool operator!=(const MatmulAttrs &other) const { return !(*this == other); }

  std::string toString() const {
    std::string result = "MatmulAttrs{";
    if (matmulProgramConfig.has_value() && matmulProgramConfig.value()) {
      std::string attrStr;
      llvm::raw_string_ostream stream(attrStr);
      stream << matmulProgramConfig.value();
      stream.flush();
      result += "matmulProgramConfig=" + attrStr;
    } else {
      result += "matmulProgramConfig=<null>";
    }
    result += ", ";
    if (computeKernelConfig.has_value() && computeKernelConfig.value()) {
      std::string attrStr;
      llvm::raw_string_ostream stream(attrStr);
      stream << computeKernelConfig.value();
      stream.flush();
      result += "computeKernelConfig=" + attrStr;
    } else {
      result += "computeKernelConfig=<null>";
    }
    result += "}";
    return result;
  }

  void dump() const { llvm::outs() << toString(); }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGATTRS_H
