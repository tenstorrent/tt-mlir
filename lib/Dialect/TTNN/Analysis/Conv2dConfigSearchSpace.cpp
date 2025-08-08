// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv2dConfigSearchSpace.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

Conv2dConfigGenerator::Conv2dConfigGenerator(
    ttnn::Conv2dOp *op, Conv2dConfigAttr baseConfig,
    const Conv2dConfigSearchSpace &space,
    std::function<bool(const Conv2dConfigAttr &)> filterOutFn)
    : op(op), baseConfig(baseConfig), searchSpace(space),
      filterOutFn(filterOutFn) {

  // Populate activeSearchFields from searchSpace.
  if (searchSpace.isWeightsDtypeSetForSearch() &&
      !baseConfig.hasWeightsDtype()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.weightsDtype),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withWeightsDtype(info.getCurrentDataType());
        });
  }
  if (searchSpace.isActivationSetForSearch() && !baseConfig.hasActivation()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.activation),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withActivation(info.getCurrentString());
        });
  }
  if (searchSpace.isDeallocateActivationSetForSearch() &&
      !baseConfig.hasDeallocateActivation()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.deallocateActivation),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withDeallocateActivation(info.getCurrentBool());
        });
  }
  if (searchSpace.isReallocateHaloOutputSetForSearch() &&
      !baseConfig.hasReallocateHaloOutput()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.reallocateHaloOutput),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withReallocateHaloOutput(info.getCurrentBool());
        });
  }
  if (searchSpace.isActBlockHOverrideSetForSearch() &&
      !baseConfig.hasActBlockHOverride()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.actBlockHOverride),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withActBlockHOverride(info.getCurrentUint32());
        });
  }
  if (searchSpace.isActBlockWDivSetForSearch() &&
      !baseConfig.hasActBlockWDiv()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.actBlockWDiv),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withActBlockWDiv(info.getCurrentUint32());
        });
  }
  if (searchSpace.isReshardIfNotOptimalSetForSearch() &&
      !baseConfig.hasReshardIfNotOptimal()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.reshardIfNotOptimal),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withReshardIfNotOptimal(info.getCurrentBool());
        });
  }
  if (searchSpace.isOverrideShardingConfigSetForSearch() &&
      !baseConfig.hasOverrideShardingConfig()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(
            searchSpace.overrideShardingConfig),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withOverrideShardingConfig(info.getCurrentBool());
        });
  }
  if (searchSpace.isShardLayoutSetForSearch() && !baseConfig.hasShardLayout()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.shardLayout),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withShardLayout(info.getCurrentTensorMemoryLayout());
        });
  }
  if (searchSpace.isCoreGridSetForSearch() && !baseConfig.hasCoreGrid()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.coreGrid),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withCoreGrid(info.getCurrentCoreRangeSetAttr());
        });
  }
  if (searchSpace.isTransposeShardsSetForSearch() &&
      !baseConfig.hasTransposeShards()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.transposeShards),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withTransposeShards(info.getCurrentBool());
        });
  }
  if (searchSpace.isOutputLayoutSetForSearch() &&
      !baseConfig.hasOutputLayout()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.outputLayout),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withOutputLayout(info.getCurrentLayout());
        });
  }
  if (searchSpace.isEnableActDoubleBufferSetForSearch() &&
      !baseConfig.hasEnableActDoubleBuffer()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.enableActDoubleBuffer),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withEnableActDoubleBuffer(info.getCurrentBool());
        });
  }
  if (searchSpace.isEnableWeightsDoubleBufferSetForSearch() &&
      !baseConfig.hasEnableWeightsDoubleBuffer()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(
            searchSpace.enableWeightsDoubleBuffer),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withEnableWeightsDoubleBuffer(info.getCurrentBool());
        });
  }
  if (searchSpace.isEnableSplitReaderSetForSearch() &&
      !baseConfig.hasEnableSplitReader()) {
    activeSearchFields.emplace_back(
        Conv2dConfigGeneratorSearchFieldInfo(searchSpace.enableSplitReader),
        [](Conv2dConfigAttr attr,
           const Conv2dConfigGeneratorSearchFieldInfo &info)
            -> Conv2dConfigAttr {
          return attr.withEnableSplitReader(info.getCurrentBool());
        });
  }

  // Initialize isDone to true if there are no active search fields.
  isDone = activeSearchFields.empty();
}

::mlir::tt::ttnn::Conv2dConfigAttr Conv2dConfigGenerator::getNextConfig() {
  // If isDone is true, it means all combinations for active fields were
  // exhausted.
  if (isDone) {
    return nullptr;
  }

  // Copy base config.
  Conv2dConfigAttr generatedAttr = baseConfig;

  // Override with current search values from activeSearchFields.
  for (const ActiveFieldEntry &fieldEntry : activeSearchFields) {
    generatedAttr = fieldEntry.updateConfig(generatedAttr, fieldEntry.info);
  }

  // Advance the state for the next call. This works like an odometer:
  // Start with the least significant "digit" (the last active search
  // field). Try to advance it.
  // - If it advances without wrapping (e.g., '1' becomes '2'), we're done
  //   with this iteration, and we have the next configuration.
  // - If it wraps (e.g., '9' becomes '0'), "carry over" to the next
  //   more significant "digit" (the previous active search field) and
  //   repeat the process.
  // If all "digits" wrap, it means we've exhausted all combinations.
  int currentFieldToAdvance = activeSearchFields.size() - 1;
  while (currentFieldToAdvance >= 0) {
    // Try to advance the current field. If it wraps, move to the next
    // field.
    if (activeSearchFields[currentFieldToAdvance].info.advance()) {
      currentFieldToAdvance--;
    } else {
      break;
    }
  }

  if (currentFieldToAdvance < 0) {
    isDone = true;
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Next conv2d config: {}",
               generatedAttr);

  if (filterOutFn(generatedAttr)) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Filtered out {}",
                 generatedAttr);
    return getNextConfig();
  }

  return generatedAttr;
}

} // namespace mlir::tt::ttnn
