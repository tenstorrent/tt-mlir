// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H
#define TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H

#include <llvm/Support/CommandLine.h>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

struct OptionNames {

  static constexpr StringRef optimizerPassEnabled = "enable-optimizer";
  static constexpr StringRef overrideInputLayout = "insert-memreconfig";
  static constexpr StringRef overrideOutputLayout = "override-output-layout";
  static constexpr StringRef memoryLayoutAnalysisEnabled =
      "memory-layout-analysis-enabled";
  static constexpr StringRef memReconfigEnabled = "memreconfig-enabled";
  static constexpr StringRef memoryLayoutAnalysisPolicy =
      "memory-layout-analysis-policy";
  static constexpr StringRef systemDescPath = "system-desc-path";
  static constexpr StringRef maxLegalLayouts = "max-legal-layouts";
  static constexpr StringRef meshShape = "mesh-shape";
};

struct OutputLayoutOverrideParams {
  std::optional<SmallVector<int64_t, 2>> grid = std::nullopt;
  std::optional<BufferType> bufferType = std::nullopt;
  std::optional<TensorMemoryLayout> tensorMemoryLayout =
      std::nullopt; // INTERLEAVED / SHARDED etc...
  std::optional<Layout> memoryLayout = std::nullopt; // ROW_MAJOR / TILE
  std::optional<tt::DataType> dataType = std::nullopt;

  // Check if all layout parameters that are generated in LegalLayoutAnalysis
  // are overridden. DataType is the only that is not.
  bool fullLayoutOverride() const {
    return grid.has_value() && bufferType.has_value() &&
           tensorMemoryLayout.has_value() && memoryLayout.has_value();
  }

  bool operator==(const OutputLayoutOverrideParams &rhs) const {
    if (grid.has_value() != rhs.grid.has_value()) {
      return false;
    }

    if (grid.has_value() && rhs.grid.has_value()) {
      if (grid.value().size() != rhs.grid.value().size()) {
        return false;
      }
      for (std::size_t i = 0; i < grid.value().size(); i++) {
        if (grid.value()[i] != rhs.grid.value()[i]) {
          return false;
        }
      }
    }

    if (bufferType.has_value() != rhs.bufferType.has_value()) {
      return false;
    }

    if (bufferType.has_value() && rhs.bufferType.has_value()) {
      if (bufferType.value() != rhs.bufferType.value()) {
        return false;
      }
    }

    if (tensorMemoryLayout.has_value() != rhs.tensorMemoryLayout.has_value()) {
      return false;
    }

    if (tensorMemoryLayout.has_value() && rhs.tensorMemoryLayout.has_value()) {
      if (tensorMemoryLayout.value() != rhs.tensorMemoryLayout.value()) {
        return false;
      }
    }

    if (memoryLayout.has_value() != rhs.memoryLayout.has_value()) {
      return false;
    }

    if (memoryLayout.has_value() && rhs.memoryLayout.has_value()) {
      if (memoryLayout.value() != rhs.memoryLayout.value()) {
        return false;
      }
    }

    if (dataType.has_value() != rhs.dataType.has_value()) {
      return false;
    }

    if (dataType.has_value() && rhs.dataType.has_value()) {
      if (dataType.value() != rhs.dataType.value()) {
        return false;
      }
    }

    return true;
  }

  bool operator!=(const OutputLayoutOverrideParams &rhs) const {
    return !(*this == rhs);
  }
};

struct InputLayoutOverrideParams {

  SmallVector<int64_t> operandIdxes;

  bool operator==(const InputLayoutOverrideParams &rhs) const {
    if (operandIdxes.size() != rhs.operandIdxes.size()) {
      return false;
    }
    for (std::size_t i = 0; i < operandIdxes.size(); i++) {
      if (operandIdxes[i] != rhs.operandIdxes[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const InputLayoutOverrideParams &rhs) const {
    return !(*this == rhs);
  }
};

struct OutputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>> {
public:
  OutputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<OutputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<OutputLayoutOverrideParams> &value);

  static std::string
  toString(const llvm::StringMap<OutputLayoutOverrideParams> &);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<OutputLayoutOverrideParams> &value);
};

struct InputLayoutOverrideParser
    : public llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>> {
public:
  InputLayoutOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<InputLayoutOverrideParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<InputLayoutOverrideParams> &value);

  static std::string
  toString(const llvm::StringMap<InputLayoutOverrideParams> &);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<InputLayoutOverrideParams> &value);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H
