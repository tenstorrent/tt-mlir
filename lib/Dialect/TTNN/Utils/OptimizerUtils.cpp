// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <unordered_map>

namespace mlir::tt::ttnn::optimizer_utils {

std::vector<OpConfig::OpSpecificAttrs>
getUniqueOpSpecificAttrs(const std::vector<OpConfig> &configs) {
  llvm::DenseSet<OpConfig::OpSpecificAttrs> uniqueAttrs;
  std::vector<OpConfig::OpSpecificAttrs> attrVec;

  for (const OpConfig &config : configs) {
    if (uniqueAttrs.insert(config.opSpecificAttrs).second) {
      attrVec.push_back(config.opSpecificAttrs);
    }
  }
  return attrVec;
}

llvm::SmallVector<OpConfig> getUniqueTestConfigsForMatmulLinear(
    const std::vector<OpConfig> &consumerConfigs) {
  // Helper structs for tracking unique (bufferType, memLayout) pairs.
  struct BufferMemLayoutKey {
    BufferType bufferType;
    TensorMemoryLayout memLayout;

    bool operator==(const BufferMemLayoutKey &other) const {
      return bufferType == other.bufferType && memLayout == other.memLayout;
    }
  };

  struct BufferMemLayoutKeyHash {
    size_t operator()(const BufferMemLayoutKey &key) const {
      return llvm::hash_combine(key.bufferType, key.memLayout);
    }
  };

  // For each unique (bufferType, memLayout), collect:
  //   - A representative partial layout (with ignorePhysicalLayout=true)
  //   - The unique opSpecificAttrs from configs with that same memLayout
  //
  // MatmulProgramConfig depends on the tensor memory layout type
  // (width_sharded uses mcast_in0=true, height_sharded uses mcast_in0=false,
  // block_sharded uses a 2D config). Pairing a program config generated for
  // one memLayout type with a different memLayout would produce invalid
  // configs.
  struct LayoutGroup {
    TTNNLayoutAttr partialLayout;
    std::vector<OpConfig::OpSpecificAttrs> uniqueAttrs;
    llvm::DenseSet<OpConfig::OpSpecificAttrs> seenAttrs;
  };

  std::unordered_map<BufferMemLayoutKey, LayoutGroup, BufferMemLayoutKeyHash>
      groups;

  for (const OpConfig &config : consumerConfigs) {
    assert(config.outputLayout &&
           "Matmul/Linear configs must have valid output layout");

    BufferMemLayoutKey key{config.outputLayout.getBufferType(),
                           config.outputLayout.getMemLayout().getValue()};

    auto &group = groups[key];
    if (!group.partialLayout) {
      group.partialLayout = TTNNLayoutAttr::Builder(config.outputLayout)
                                .setIgnorePhysicalLayout(true);
    }
    if (group.seenAttrs.insert(config.opSpecificAttrs).second) {
      group.uniqueAttrs.push_back(config.opSpecificAttrs);
    }
  }

  // Build test configs: each partial layout is paired only with
  // opSpecificAttrs from configs of the same (bufferType, memLayout) group.
  llvm::SmallVector<OpConfig> testConfigs;
  for (const auto &[layoutKey, group] : groups) {
    for (const OpConfig::OpSpecificAttrs &attrs : group.uniqueAttrs) {
      testConfigs.push_back(OpConfig(group.partialLayout, attrs));
    }
  }

  return testConfigs;
}

llvm::SmallVector<OpConfig>
getUniqueTestConfigs(const std::vector<OpConfig> &consumerConfigs,
                     bool isMatmulOrLinear) {
  if (isMatmulOrLinear) {
    return getUniqueTestConfigsForMatmulLinear(consumerConfigs);
  }

  // For non-Matmul/Linear: only op-specific attrs matter, no output layout
  // needed.
  std::vector<OpConfig::OpSpecificAttrs> uniqueAttrs =
      getUniqueOpSpecificAttrs(consumerConfigs);
  llvm::SmallVector<OpConfig> testConfigs;
  for (const OpConfig::OpSpecificAttrs &attrs : uniqueAttrs) {
    testConfigs.push_back(OpConfig(/*outputLayout=*/nullptr, attrs));
  }
  return testConfigs;
}

} // namespace mlir::tt::ttnn::optimizer_utils
