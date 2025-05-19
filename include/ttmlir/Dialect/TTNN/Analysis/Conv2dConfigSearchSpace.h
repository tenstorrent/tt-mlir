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
  std::optional<llvm::SmallVector<ttcore::DataType>> dtype;
  std::optional<llvm::SmallVector<ttcore::DataType>> weightsDtype;
  std::optional<llvm::SmallVector<std::string>> activation;
  std::optional<llvm::SmallVector<bool>> deallocateActivation;
  std::optional<llvm::SmallVector<bool>> reallocateHaloOutput;
  std::optional<llvm::SmallVector<uint32_t>> actBlockHOverride;
  std::optional<llvm::SmallVector<uint32_t>> actBlockWDiv;
  std::optional<llvm::SmallVector<bool>> reshardIfNotOptimal;
  std::optional<llvm::SmallVector<bool>> overrideShardingConfig;
  std::optional<llvm::SmallVector<::mlir::tt::ttnn::TensorMemoryLayout>>
      shardLayout;
  std::optional<llvm::SmallVector<::mlir::tt::ttnn::CoreRangeSetAttr>> coreGrid;
  std::optional<llvm::SmallVector<bool>> transposeShards;
  std::optional<llvm::SmallVector<::mlir::tt::ttnn::Layout>> outputLayout;
  std::optional<llvm::SmallVector<bool>> preprocessWeightsOnDevice;
  std::optional<llvm::SmallVector<bool>> alwaysPreprocessWeights;
  std::optional<llvm::SmallVector<bool>> enableActDoubleBuffer;
  std::optional<llvm::SmallVector<bool>> enableWeightsDoubleBuffer;
  std::optional<llvm::SmallVector<bool>> enableSplitReader;
  std::optional<llvm::SmallVector<bool>> enableSubblockPadding;

  // Constructor: All fields are std::nullopt by default.
  Conv2dConfigSearchSpace() = default;

  // Methods to set search values.
  void setSearchDtype(llvm::SmallVector<ttcore::DataType> &&values) {
    dtype = std::move(values);
  }
  void setSearchWeightsDtype(llvm::SmallVector<ttcore::DataType> &&values) {
    weightsDtype = std::move(values);
  }
  void setSearchActivation(llvm::SmallVector<std::string> &&values) {
    activation = std::move(values);
  }
  void setSearchDeallocateActivation(llvm::SmallVector<bool> &&values) {
    deallocateActivation = std::move(values);
  }
  void setSearchReallocateHaloOutput(llvm::SmallVector<bool> &&values) {
    reallocateHaloOutput = std::move(values);
  }
  void setSearchActBlockHOverride(llvm::SmallVector<uint32_t> &&values) {
    actBlockHOverride = std::move(values);
  }
  void setSearchActBlockWDiv(llvm::SmallVector<uint32_t> &&values) {
    actBlockWDiv = std::move(values);
  }
  void setSearchReshardIfNotOptimal(llvm::SmallVector<bool> &&values) {
    reshardIfNotOptimal = std::move(values);
  }
  void setSearchOverrideShardingConfig(llvm::SmallVector<bool> &&values) {
    overrideShardingConfig = std::move(values);
  }
  void setSearchShardLayout(
      llvm::SmallVector<::mlir::tt::ttnn::TensorMemoryLayout> &&values) {
    shardLayout = std::move(values);
  }
  void setSearchCoreGrid(
      llvm::SmallVector<::mlir::tt::ttnn::CoreRangeSetAttr> &&values) {
    coreGrid = std::move(values);
  }
  void setSearchTransposeShards(llvm::SmallVector<bool> &&values) {
    transposeShards = std::move(values);
  }
  void
  setSearchOutputLayout(llvm::SmallVector<::mlir::tt::ttnn::Layout> &&values) {
    outputLayout = std::move(values);
  }
  void setSearchPreprocessWeightsOnDevice(llvm::SmallVector<bool> &&values) {
    preprocessWeightsOnDevice = std::move(values);
  }
  void setSearchAlwaysPreprocessWeights(llvm::SmallVector<bool> &&values) {
    alwaysPreprocessWeights = std::move(values);
  }
  void setSearchEnableActDoubleBuffer(llvm::SmallVector<bool> &&values) {
    enableActDoubleBuffer = std::move(values);
  }
  void setSearchEnableWeightsDoubleBuffer(llvm::SmallVector<bool> &&values) {
    enableWeightsDoubleBuffer = std::move(values);
  }
  void setSearchEnableSplitReader(llvm::SmallVector<bool> &&values) {
    enableSplitReader = std::move(values);
  }
  void setSearchEnableSubblockPadding(llvm::SmallVector<bool> &&values) {
    enableSubblockPadding = std::move(values);
  }

  // Methods to check if field is set.
  bool isDtypeSetForSearch() const {
    return dtype.has_value() && !dtype.value().empty();
  }
  bool isWeightsDtypeSetForSearch() const {
    return weightsDtype.has_value() && !weightsDtype.value().empty();
  }
  bool isActivationSetForSearch() const {
    return activation.has_value() && !activation.value().empty();
  }
  bool isDeallocateActivationSetForSearch() const {
    return deallocateActivation.has_value() &&
           !deallocateActivation.value().empty();
  }
  bool isReallocateHaloOutputSetForSearch() const {
    return reallocateHaloOutput.has_value() &&
           !reallocateHaloOutput.value().empty();
  }
  bool isActBlockHOverrideSetForSearch() const {
    return actBlockHOverride.has_value() && !actBlockHOverride.value().empty();
  }
  bool isActBlockWDivSetForSearch() const {
    return actBlockWDiv.has_value() && !actBlockWDiv.value().empty();
  }
  bool isReshardIfNotOptimalSetForSearch() const {
    return reshardIfNotOptimal.has_value() &&
           !reshardIfNotOptimal.value().empty();
  }
  bool isOverrideShardingConfigSetForSearch() const {
    return overrideShardingConfig.has_value() &&
           !overrideShardingConfig.value().empty();
  }
  bool isShardLayoutSetForSearch() const {
    return shardLayout.has_value() && !shardLayout.value().empty();
  }
  bool isCoreGridSetForSearch() const {
    return coreGrid.has_value() && !coreGrid.value().empty();
  }
  bool isTransposeShardsSetForSearch() const {
    return transposeShards.has_value() && !transposeShards.value().empty();
  }
  bool isOutputLayoutSetForSearch() const {
    return outputLayout.has_value() && !outputLayout.value().empty();
  }
  bool isPreprocessWeightsOnDeviceSetForSearch() const {
    return preprocessWeightsOnDevice.has_value() &&
           !preprocessWeightsOnDevice.value().empty();
  }
  bool isAlwaysPreprocessWeightsSetForSearch() const {
    return alwaysPreprocessWeights.has_value() &&
           !alwaysPreprocessWeights.value().empty();
  }
  bool isEnableActDoubleBufferSetForSearch() const {
    return enableActDoubleBuffer.has_value() &&
           !enableActDoubleBuffer.value().empty();
  }
  bool isEnableWeightsDoubleBufferSetForSearch() const {
    return enableWeightsDoubleBuffer.has_value() &&
           !enableWeightsDoubleBuffer.value().empty();
  }
  bool isEnableSplitReaderSetForSearch() const {
    return enableSplitReader.has_value() && !enableSplitReader.value().empty();
  }
  bool isEnableSubblockPaddingSetForSearch() const {
    return enableSubblockPadding.has_value() &&
           !enableSubblockPadding.value().empty();
  }

  // Helper to check if any field has been set with search values
  bool isAnyFieldSetForSearch() const {
    return dtype.has_value() || weightsDtype.has_value() ||
           activation.has_value() || deallocateActivation.has_value() ||
           reallocateHaloOutput.has_value() || actBlockHOverride.has_value() ||
           actBlockWDiv.has_value() || reshardIfNotOptimal.has_value() ||
           overrideShardingConfig.has_value() || shardLayout.has_value() ||
           coreGrid.has_value() || // Added coreGrid check
           transposeShards.has_value() || outputLayout.has_value() ||
           preprocessWeightsOnDevice.has_value() ||
           alwaysPreprocessWeights.has_value() ||
           enableActDoubleBuffer.has_value() ||
           enableWeightsDoubleBuffer.has_value() ||
           enableSplitReader.has_value() || enableSubblockPadding.has_value();
  }
};

struct Conv2dConfigSearchSpaceFactory {
  static Conv2dConfigSearchSpace get() {
    static Conv2dConfigSearchSpace searchSpace;

    // 0 is default and will use most memory. Must be multiple of 32. 32 is
    // recommended for memory savings.
    searchSpace.setSearchActBlockHOverride({0, 32, 64});

    // TODO(rpavlovicTT) we can not enable deallocation until we fix
    // https://github.com/tenstorrent/tt-mlir/issues/3383
    // searchSpace.setSearchDeallocateActivation({false, true});

    searchSpace.setSearchReshardIfNotOptimal({false, true});

    searchSpace.setSearchEnableSplitReader({false, true});

    return searchSpace;
  }
};

// Helper struct for Conv2dConfigGenerator. It contains the search space for a
// single field in the Conv2dConfigAttr. It's generic over the type of the
// field.
struct Conv2dConfigGeneratorSearchFieldInfo {
  using SearchValueVariant =
      std::variant<llvm::SmallVector<ttcore::DataType>,
                   llvm::SmallVector<std::string>, llvm::SmallVector<uint32_t>,
                   llvm::SmallVector<bool>,
                   llvm::SmallVector<::mlir::tt::ttnn::TensorMemoryLayout>,
                   llvm::SmallVector<::mlir::tt::ttnn::Layout>,
                   llvm::SmallVector<::mlir::tt::ttnn::CoreRangeSetAttr>>;

  SearchValueVariant values;
  size_t currentIndex = 0;

  template <typename T>
  Conv2dConfigGeneratorSearchFieldInfo(llvm::SmallVector<T> &&vec)
      : values(std::move(vec)) {
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
  std::string getCurrentString() const {
    return std::get<llvm::SmallVector<std::string>>(values)[currentIndex];
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
  Conv2dConfigGenerator(ttnn::Conv2dOp *op, Conv2dConfigAttr baseConfig,
                        const Conv2dConfigSearchSpace &space);

  // Returns the next configuration in the search space.
  // Returns nullptr if all combinations have been exhausted.
  ::mlir::tt::ttnn::Conv2dConfigAttr getNextConfig();

  // Returns true if given combination does not pass filter.
  bool filterOut(const Conv2dConfigAttr &config) const;

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

  // True if all combinations have been exhausted.
  bool isDone = false;
};

} // namespace ttnn
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_CONV2DCONFIGSEARCHSPACE_H
