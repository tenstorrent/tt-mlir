// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::sharding_utils {

#if TTMLIR_ENABLE_STABLEHLO

class MeshSharding {
public:
  MeshSharding() {};
  ~MeshSharding() {};

  // Convert mhlo.sharding string to meshSharding.
  llvm::Expected<bool>
  convertGSPMDShardingToMeshSharding(StringRef shardingStr);

  // Convert sdy.sharding to meshSharding.
  llvm::Expected<bool>
  convertSdyShardingToMeshSharding(mlir::sdy::TensorShardingAttr sdySharding,
                                   mlir::sdy::MeshAttr mesh,
                                   mlir::tt::MeshShardDirection direction);

  // Getter functions.
  mlir::tt::MeshShardDirection getShardDirection(void) {
    return shardDirection;
  };
  mlir::tt::MeshShardType getShardType(void) { return shardType; };
  llvm::SmallVector<int64_t> getShardShape(void) { return shardShape; };
  llvm::SmallVector<int64_t> getShardDims(void) { return shardDims; };
  llvm::SmallVector<int64_t> getMeshShape(void) { return meshShape; };

private:
  // Parse GSPMD devices string and fill out MeshSharding info.
  llvm::Expected<bool> parseGSPMDDevicesStr(const StringRef devicesStr);

  // Based on current MeshSharding info, finalize sharding dimensions.
  llvm::Expected<bool> determineGSPMDShardingDims();

  // Set sharyType other than devices and reset values.
  void setNonDevicesShardType(tt::MeshShardType targetShardType) {
    assert(targetShardType != tt::MeshShardType::Devices);
    shardType = targetShardType;
    shardShape = llvm::SmallVector<int64_t>{1};
    shardDims = llvm::SmallVector<int64_t>{-1};
    meshShape = llvm::SmallVector<int64_t>{-1};
  }

private:
  mlir::tt::MeshShardDirection shardDirection =
      mlir::tt::MeshShardDirection::ShardToFull;
  mlir::tt::MeshShardType shardType = mlir::tt::MeshShardType::Manual;
  llvm::SmallVector<int64_t> shardShape{-1};
  llvm::SmallVector<int64_t> shardDims{-1};
  llvm::SmallVector<int64_t> meshShape{-1};
  llvm::SmallVector<int64_t> deviceIds{-1};
  bool lastTileDimReplicate = false;
};

#endif

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
