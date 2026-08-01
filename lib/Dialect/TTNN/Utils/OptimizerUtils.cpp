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
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

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
  struct LayoutGeometryKey {
    BufferType bufferType;
    TensorMemoryLayout memLayout;
    std::vector<int64_t> gridShape;
    std::vector<int64_t> shardShape;
    std::vector<uint64_t> coreRangeSignature;

    bool operator<(const LayoutGeometryKey &other) const {
      return std::tie(bufferType, memLayout, gridShape, shardShape,
                      coreRangeSignature) <
             std::tie(other.bufferType, other.memLayout, other.gridShape,
                      other.shardShape, other.coreRangeSignature);
    }

    static LayoutGeometryKey get(TTNNLayoutAttr layout) {
      llvm::ArrayRef<int64_t> gridShape = layout.getGridShape();
      llvm::SmallVector<int64_t> shardShape = layout.getShardShape();
      LayoutGeometryKey key{
          layout.getBufferType(),
          layout.getMemLayout().getValue(),
          std::vector<int64_t>(gridShape.begin(), gridShape.end()),
          std::vector<int64_t>(shardShape.begin(), shardShape.end()),
          {}};
      if (CoreRangeSetAttr ranges = layout.getCoreRangeSet()) {
        key.coreRangeSignature.reserve(ranges.getCoreRanges().size() * 4);
        for (CoreRangeAttr range : ranges.getCoreRanges()) {
          key.coreRangeSignature.push_back(range.getStartCoord().getX());
          key.coreRangeSignature.push_back(range.getStartCoord().getY());
          key.coreRangeSignature.push_back(range.getEndCoord().getX());
          key.coreRangeSignature.push_back(range.getEndCoord().getY());
        }
      }
      return key;
    }
  };

  // For each unique output layout geometry, collect:
  //   - A representative physical layout
  //   - The unique opSpecificAttrs from configs with that same geometry
  //
  // MatmulProgramConfig depends on both the tensor memory layout type and its
  // physical grid/shard geometry. In particular, per_core_M/N size dynamic
  // circular buffers which are backed by the output tensor shard. Pairing
  // attrs generated for one grid with a representative layout from another
  // can request a CB larger than that layout's shard bank.
  struct LayoutGroup {
    TTNNLayoutAttr partialLayout;
    std::vector<OpConfig::OpSpecificAttrs> uniqueAttrs;
    llvm::DenseSet<OpConfig::OpSpecificAttrs> seenAttrs;
  };

  // Iteration order must be deterministic: downstream tie-breaks
  // (e.g., L1-spill's first-fit walk over fallback configs) commit to
  // the first entry, so a shuffled order produces different IR across
  // processes. std::unordered_map iteration depends on bucket layout
  // and varies between processes; std::map iterates in deterministic geometry
  // key order.
  std::map<LayoutGeometryKey, LayoutGroup> groups;

  for (const OpConfig &config : consumerConfigs) {
    assert(config.outputLayout &&
           "Matmul/Linear configs must have valid output layout");

    LayoutGeometryKey key = LayoutGeometryKey::get(config.outputLayout);

    LayoutGroup &group = groups[key];
    if (!group.partialLayout) {
      TTNNLayoutAttr layout = config.outputLayout;
      // Explicit matmul program configs size tensor-backed circular buffers
      // against this exact shard placement. Marking the representative as
      // ignorePhysicalLayout drops its ShardSpec during OpModel conversion and
      // can defer invalid dynamic-CB failures until after constraints return.
      group.partialLayout = layout.withIgnorePhysicalLayout(false);
    }
    if (group.seenAttrs.insert(config.opSpecificAttrs).second) {
      group.uniqueAttrs.push_back(config.opSpecificAttrs);
    }
  }

  // Build test configs: each partial layout is paired only with
  // opSpecificAttrs from configs of the same physical/shard geometry group.
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
