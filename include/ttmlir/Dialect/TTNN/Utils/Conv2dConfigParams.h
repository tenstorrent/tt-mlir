// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_CONV2DCONFIGPARAMS_H
#define TTMLIR_DIALECT_TTNN_UTILS_CONV2DCONFIGPARAMS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <optional>

namespace mlir::tt::ttnn {

/// Unified configuration parameters for Conv2d operations.
/// This class serves dual purposes:
/// 1. As a mutable helper for Conv2dConfigAttr (immutable MLIR attribute)
/// 2. As a flexible override container for selective parameter overriding
///
/// All fields are std::optional to support partial configuration and selective
/// overrides.
struct Conv2dConfigParams {
  std::optional<ttcore::DataType> weightsDtype = std::nullopt;
  std::optional<UnaryOpType> activation = std::nullopt;
  std::optional<std::vector<FloatAttr>> activationParams = std::nullopt;
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
  std::optional<bool> enableActDoubleBuffer = std::nullopt;
  std::optional<bool> enableWeightsDoubleBuffer = std::nullopt;
  std::optional<bool> inPlace = std::nullopt;

  // Default constructor - all fields nullopt
  Conv2dConfigParams() = default;

  // Constructor from Conv2dConfigAttr (for mutability helper use case)
  Conv2dConfigParams(Conv2dConfigAttr attr, bool partial = true);

  // Conversion method to build MLIR attribute
  Conv2dConfigAttr buildConv2dConfigAttr(::mlir::MLIRContext *ctx) const;

  bool hasWeightsDtype() const { return weightsDtype.has_value(); }
  bool hasActivation() const { return activation.has_value(); }
  bool hasDeallocateActivation() const {
    return deallocateActivation.has_value();
  }
  bool hasReallocateHaloOutput() const {
    return reallocateHaloOutput.has_value();
  }
  bool hasActBlockHOverride() const { return actBlockHOverride.has_value(); }
  bool hasActBlockWDiv() const { return actBlockWDiv.has_value(); }
  bool hasReshardIfNotOptimal() const {
    return reshardIfNotOptimal.has_value();
  }
  bool hasOverrideShardingConfig() const {
    return overrideShardingConfig.has_value();
  }
  bool hasShardLayout() const { return shardLayout.has_value(); }
  bool hasCoreGrid() const { return coreGrid.has_value(); }
  bool hasTransposeShards() const { return transposeShards.has_value(); }
  bool hasOutputLayout() const { return outputLayout.has_value(); }
  bool hasEnableActDoubleBuffer() const {
    return enableActDoubleBuffer.has_value();
  }
  bool hasEnableWeightsDoubleBuffer() const {
    return enableWeightsDoubleBuffer.has_value();
  }
  bool hasInPlace() const { return inPlace.has_value(); }

  /// Check if all fields are unset (empty configuration)
  bool empty() const {
    return !hasWeightsDtype() && !hasActivation() &&
           !hasDeallocateActivation() && !hasReallocateHaloOutput() &&
           !hasActBlockHOverride() && !hasActBlockWDiv() &&
           !hasReshardIfNotOptimal() && !hasOverrideShardingConfig() &&
           !hasShardLayout() && !hasCoreGrid() && !hasTransposeShards() &&
           !hasOutputLayout() && !hasEnableActDoubleBuffer() &&
           !hasEnableWeightsDoubleBuffer() && !hasInPlace();
  }

  /// Check if all fields are set (complete configuration)
  bool fullConfigOverride() const {
    return hasWeightsDtype() && hasActivation() && hasDeallocateActivation() &&
           hasReallocateHaloOutput() && hasActBlockHOverride() &&
           hasActBlockWDiv() && hasReshardIfNotOptimal() &&
           hasOverrideShardingConfig() && hasShardLayout() && hasCoreGrid() &&
           hasTransposeShards() && hasOutputLayout() &&
           hasEnableActDoubleBuffer() && hasEnableWeightsDoubleBuffer() &&
           hasInPlace();
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const Conv2dConfigParams &params) {
    os << "weights_dtype#" << params.weightsDtype << ":activation#"
       << params.activation << ":deallocate_activation#"
       << params.deallocateActivation << ":reallocate_halo_output#"
       << params.reallocateHaloOutput << ":act_block_h_override#"
       << params.actBlockHOverride << ":act_block_w_div#" << params.actBlockWDiv
       << ":reshard_if_not_optimal#" << params.reshardIfNotOptimal
       << ":override_sharding_config#" << params.overrideShardingConfig
       << ":shard_layout#" << params.shardLayout << ":core_grid#"
       << params.coreGrid << ":transpose_shards#" << params.transposeShards
       << ":output_layout#" << params.outputLayout
       << ":enable_act_double_buffer#" << params.enableActDoubleBuffer
       << ":enable_weights_double_buffer#" << params.enableWeightsDoubleBuffer
       << params.inPlace;
    return os;
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_UTILS_CONV2DCONFIGPARAMS_H
