// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt::sharding_utils {

#if TTMLIR_ENABLE_STABLEHLO
struct MeshSharding {
  mlir::tt::MeshShardDirection shardDirection;
  mlir::tt::MeshShardType shardType;
  bool lastTileDimReplicate;
  llvm::SmallVector<int64_t> shardShape;
  llvm::SmallVector<int64_t> shardDims;
  llvm::SmallVector<int64_t> meshShape;
};

LogicalResult parseGSPMDShardingAttr(StringRef shardingStr,
                                     MeshSharding &meshSharding);
#endif

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
