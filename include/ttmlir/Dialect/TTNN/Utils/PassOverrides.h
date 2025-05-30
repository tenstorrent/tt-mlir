// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H
#define TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/Support/CommandLine.h"

#include <optional>

namespace mlir::tt::ttnn {

struct OptionNames {

  static constexpr StringRef optimizerPassEnabled = "enable-optimizer";
  static constexpr StringRef insertMemReconfig = "insert-memreconfig";
  static constexpr StringRef overrideOutputLayout = "override-output-layout";
  static constexpr StringRef overrideConv2dConfig = "override-conv2d-config";
  static constexpr StringRef memoryLayoutAnalysisEnabled =
      "memory-layout-analysis-enabled";
  static constexpr StringRef memReconfigEnabled = "memreconfig-enabled";
  static constexpr StringRef memoryLayoutAnalysisPolicy =
      "memory-layout-analysis-policy";
  static constexpr StringRef systemDescPath = "system-desc-path";
  static constexpr StringRef mockSystemDescArch = "mock-system-desc-arch";
  static constexpr StringRef maxLegalLayouts = "max-legal-layouts";
  static constexpr StringRef meshShape = "mesh-shape";
};

struct Conv2dConfigOverrideParams {
  std::optional<tt::DataType> dtype = std::nullopt;
  std::optional<tt::DataType> weightsDtype = std::nullopt;
  std::optional<std::string> activation = std::nullopt;
  std::optional<bool> deallocateActivation = std::nullopt;
  std::optional<bool> reallocateHaloOutput = std::nullopt;
  std::optional<uint32_t> actBlockHOverride = std::nullopt;
  std::optional<uint32_t> actBlockWDiv = std::nullopt;
  std::optional<bool> reshardIfNotOptimal = std::nullopt;
  std::optional<bool> overrideShardingConfig = std::nullopt;
  std::optional<TensorMemoryLayout> shardLayout = std::nullopt;
  std::optional<CoreRangeSetAttr> coreGrid = std::nullopt;
  std::optional<bool> transposeShards = std::nullopt;
  std::optional<Layout> outputLayout = std::nullopt;
  std::optional<bool> preprocessWeightsOnDevice = std::nullopt;
  std::optional<bool> alwaysPreprocessWeights = std::nullopt;
  std::optional<bool> enableActDoubleBuffer = std::nullopt;
  std::optional<bool> enableWeightsDoubleBuffer = std::nullopt;
  std::optional<bool> enableSplitReader = std::nullopt;
  std::optional<bool> enableSubblockPadding = std::nullopt;

  bool empty() const {
    return !dtype.has_value() && !weightsDtype.has_value() &&
           !activation.has_value() && !deallocateActivation.has_value() &&
           !reallocateHaloOutput.has_value() &&
           !actBlockHOverride.has_value() && !actBlockWDiv.has_value() &&
           !reshardIfNotOptimal.has_value() &&
           !overrideShardingConfig.has_value() && !shardLayout.has_value() &&
           !coreGrid.has_value() && !transposeShards.has_value() &&
           !outputLayout.has_value() &&
           !preprocessWeightsOnDevice.has_value() &&
           !alwaysPreprocessWeights.has_value() &&
           !enableActDoubleBuffer.has_value() &&
           !enableWeightsDoubleBuffer.has_value() &&
           !enableSplitReader.has_value() && !enableSubblockPadding.has_value();
  }
};

struct OutputLayoutOverrideParams {
  std::optional<SmallVector<int64_t, 2>> grid = std::nullopt;
  std::optional<BufferType> bufferType = std::nullopt;
  std::optional<TensorMemoryLayout> tensorMemoryLayout =
      std::nullopt; // INTERLEAVED / SHARDED etc...
  std::optional<Layout> memoryLayout = std::nullopt; // ROW_MAJOR / TILE
  std::optional<tt::DataType> dataType = std::nullopt;

  bool empty() const {
    return !grid.has_value() && !bufferType.has_value() &&
           !tensorMemoryLayout.has_value() && !memoryLayout.has_value() &&
           !dataType.has_value();
  }

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

struct InsertMemReconfigParams {

  SmallVector<int64_t> operandIdxes;

  bool operator==(const InsertMemReconfigParams &rhs) const {
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

  bool operator!=(const InsertMemReconfigParams &rhs) const {
    return !(*this == rhs);
  }
};

struct Conv2dConfigOverrideParser
    : public llvm::cl::parser<llvm::StringMap<Conv2dConfigOverrideParams>> {
public:
  Conv2dConfigOverrideParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<Conv2dConfigOverrideParams>>(opt) {}

  // Parse string in override-conv2d-config format.
  // Return true if string format is invalid to indicate error.
  // Example of a valid string:
  // "conv2d_1=enable_weights_double_buffer#true:activation#none,conv2d_2=dtype#bf16"
  //
  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<Conv2dConfigOverrideParams> &value);

  // Return override-conv2d-config string represenation.
  //
  static std::string
  toString(const llvm::StringMap<Conv2dConfigOverrideParams> &);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<Conv2dConfigOverrideParams> &value);
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

struct InsertMemReconfigParser
    : public llvm::cl::parser<llvm::StringMap<InsertMemReconfigParams>> {
public:
  InsertMemReconfigParser(llvm::cl::Option &opt)
      : llvm::cl::parser<llvm::StringMap<InsertMemReconfigParams>>(opt) {}

  bool parse(llvm::cl::Option &opt, StringRef argName, StringRef arg,
             llvm::StringMap<InsertMemReconfigParams> &value);

  static std::string toString(const llvm::StringMap<InsertMemReconfigParams> &);

  static void print(llvm::raw_ostream &os,
                    const llvm::StringMap<InsertMemReconfigParams> &value);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_UTILS_PASSOVERRIDES_H
