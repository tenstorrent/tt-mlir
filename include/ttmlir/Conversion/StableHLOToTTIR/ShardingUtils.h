// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt::sharding_utils {

#if TTMLIR_ENABLE_STABLEHLO
typedef struct {
  mlir::tt::MeshShardDirection shardDirection;
  mlir::tt::MeshShardType shardType;
  bool lastTileDimReplicate;
  std::vector<int64_t> shardShape;
  std::vector<int64_t> shardDims;
  std::vector<int64_t> meshShape;
} MeshShardAttr;

LogicalResult parseGSPMDShardingAttr(StringRef shardingStr,
                                     MeshShardAttr &meshShardAttr);
#endif

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
