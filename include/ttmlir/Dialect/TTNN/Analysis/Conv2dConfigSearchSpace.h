// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_CONV2DCONFIGSEARCHSPACE_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_CONV2DCONFIGSEARCHSPACE_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <optional>
#include <string>
#include <variant>

namespace mlir {
namespace tt {
namespace ttnn {

struct Conv2dConfigSearchSpace {
  llvm::SmallVector<ttcore::DataType> weightsDtype;
  llvm::SmallVector<UnaryOpType> activation;
  llvm::SmallVector<bool> deallocateActivation;
  llvm::SmallVector<bool> reallocateHaloOutput;
  llvm::SmallVector<uint32_t> actBlockHOverride;
  llvm::SmallVector<uint32_t> actBlockWDiv;
  llvm::SmallVector<bool> reshardIfNotOptimal;
  llvm::SmallVector<bool> overrideShardingConfig;
  llvm::SmallVector<::mlir::tt::ttnn::TensorMemoryLayout> shardLayout;
  llvm::SmallVector<::mlir::tt::ttnn::CoreRangeSetAttr> coreGrid;
  llvm::SmallVector<bool> transposeShards;
  llvm::SmallVector<::mlir::tt::ttnn::Layout> outputLayout;
  llvm::SmallVector<bool> enableActDoubleBuffer;
  llvm::SmallVector<bool> enableWeightsDoubleBuffer;

  // Constructor: All fields are empty by default.
  Conv2dConfigSearchSpace() = default;

  // Methods to check if field is set.
  bool isWeightsDtypeSetForSearch() const { return !weightsDtype.empty(); }
  bool isActivationSetForSearch() const { return !activation.empty(); }
  bool isDeallocateActivationSetForSearch() const {
    return !deallocateActivation.empty();
  }
  bool isReallocateHaloOutputSetForSearch() const {
    return !reallocateHaloOutput.empty();
  }
  bool isActBlockHOverrideSetForSearch() const {
    return !actBlockHOverride.empty();
  }
  bool isActBlockWDivSetForSearch() const { return !actBlockWDiv.empty(); }
  bool isReshardIfNotOptimalSetForSearch() const {
    return !reshardIfNotOptimal.empty();
  }
  bool isOverrideShardingConfigSetForSearch() const {
    return !overrideShardingConfig.empty();
  }
  bool isShardLayoutSetForSearch() const { return !shardLayout.empty(); }
  bool isCoreGridSetForSearch() const { return !coreGrid.empty(); }
  bool isTransposeShardsSetForSearch() const {
    return !transposeShards.empty();
  }
  bool isOutputLayoutSetForSearch() const { return !outputLayout.empty(); }
  bool isEnableActDoubleBufferSetForSearch() const {
    return !enableActDoubleBuffer.empty();
  }
  bool isEnableWeightsDoubleBufferSetForSearch() const {
    return !enableWeightsDoubleBuffer.empty();
  }

  // Helper to check if any field has been set with search values
  bool isAnyFieldSetForSearch() const {
    return isWeightsDtypeSetForSearch() || isActivationSetForSearch() ||
           isDeallocateActivationSetForSearch() ||
           isReallocateHaloOutputSetForSearch() ||
           isActBlockHOverrideSetForSearch() || isActBlockWDivSetForSearch() ||
           isReshardIfNotOptimalSetForSearch() ||
           isOverrideShardingConfigSetForSearch() ||
           isShardLayoutSetForSearch() || isCoreGridSetForSearch() ||
           isTransposeShardsSetForSearch() || isOutputLayoutSetForSearch() ||
           isEnableActDoubleBufferSetForSearch() ||
           isEnableWeightsDoubleBufferSetForSearch();
  }
};

// Helper struct for Conv2dConfigGenerator. It contains the search space for a
// single field in the Conv2dConfigAttr. It's generic over the type of the
// field.
struct Conv2dConfigGeneratorSearchFieldInfo {
  using SearchValueVariant =
      std::variant<llvm::SmallVector<ttcore::DataType>,
                   llvm::SmallVector<UnaryOpType>, llvm::SmallVector<uint32_t>,
                   llvm::SmallVector<bool>,
                   llvm::SmallVector<::mlir::tt::ttnn::TensorMemoryLayout>,
                   llvm::SmallVector<::mlir::tt::ttnn::Layout>,
                   llvm::SmallVector<::mlir::tt::ttnn::CoreRangeSetAttr>>;

  SearchValueVariant values;
  size_t currentIndex = 0;

  template <typename T>
  Conv2dConfigGeneratorSearchFieldInfo(const llvm::SmallVector<T> &vec)
      : values(vec) {
    assert(!std::visit([](const auto &v) { return v.empty(); }, values) &&
           "Search space must not be empty");
  }

  // Returns true if wrapped around all values.
  bool advance() {
    currentIndex++;
    if (currentIndex >=
        std::visit([](const auto &v) { return v.size(); }, values)) {
      // Exhausted all values, reset.
      currentIndex = 0;
      return true;
    }

    // Did not wrap.
    return false;
  }

  // Specific getters for each type in the variant.
  ttcore::DataType getCurrentDataType() const {
    return std::get<llvm::SmallVector<ttcore::DataType>>(values)[currentIndex];
  }
  UnaryOpType getCurrentUnaryOpType() const {
    return std::get<llvm::SmallVector<UnaryOpType>>(values)[currentIndex];
  }
  uint32_t getCurrentUint32() const {
    return std::get<llvm::SmallVector<uint32_t>>(values)[currentIndex];
  }
  bool getCurrentBool() const {
    return std::get<llvm::SmallVector<bool>>(values)[currentIndex];
  }
  ::mlir::tt::ttnn::TensorMemoryLayout getCurrentTensorMemoryLayout() const {
    return std::get<llvm::SmallVector<::mlir::tt::ttnn::TensorMemoryLayout>>(
        values)[currentIndex];
  }
  ::mlir::tt::ttnn::Layout getCurrentLayout() const {
    return std::get<llvm::SmallVector<::mlir::tt::ttnn::Layout>>(
        values)[currentIndex];
  }
  ::mlir::tt::ttnn::CoreRangeSetAttr getCurrentCoreRangeSetAttr() const {
    return std::get<llvm::SmallVector<::mlir::tt::ttnn::CoreRangeSetAttr>>(
        values)[currentIndex];
  }
};

class Conv2dConfigGenerator {
public:
  Conv2dConfigGenerator(
      ttnn::Conv2dOp *op, Conv2dConfigAttr baseConfig,
      const Conv2dConfigSearchSpace &space,
      std::function<bool(const Conv2dConfigAttr &)> filterOutFn);

  // Returns the next configuration in the search space.
  // Returns nullptr if all combinations have been exhausted.
  ::mlir::tt::ttnn::Conv2dConfigAttr getNextConfig();

  // Returns true if no search is left to be done.
  bool searchDone() const { return isDone; }

private:
  // Helper struct to store active search field data and its update logic.
  struct ActiveFieldEntry {
    // Information about the search field containing the values to search over.
    Conv2dConfigGeneratorSearchFieldInfo info;

    // Function to update the given conv2d config with the current value of the
    // search field in this entry.
    std::function<Conv2dConfigAttr(
        Conv2dConfigAttr, const Conv2dConfigGeneratorSearchFieldInfo &)>
        updateConfig;

    ActiveFieldEntry(
        Conv2dConfigGeneratorSearchFieldInfo &&i,
        std::function<Conv2dConfigAttr(
            Conv2dConfigAttr, const Conv2dConfigGeneratorSearchFieldInfo &)>
            &&updater)
        : info(std::move(i)), updateConfig(std::move(updater)) {}
  };

  // Operation for which the generator is created.
  [[maybe_unused]] ttnn::Conv2dOp *op;

  // Base config for conv2d config.
  Conv2dConfigAttr baseConfig;

  // Search space for conv2d config.
  Conv2dConfigSearchSpace searchSpace;

  // Active search fields are the ones that are enabled for search.
  llvm::SmallVector<ActiveFieldEntry> activeSearchFields;

  // Function to filter out invalid configurations.
  std::function<bool(const Conv2dConfigAttr &)> filterOutFn;

  // True if all combinations have been exhausted.
  bool isDone = false;
};

} // namespace ttnn
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_CONV2DCONFIGSEARCHSPACE_H
