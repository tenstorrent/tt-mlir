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

  // Force dummy sharding op by setting shard_type to manual. The mesh_shard op
  // will be ignored at runtime by simply copying input tensor to output.
  void setDummyShardingOp() { shardType = mlir::tt::MeshShardType::Manual; }

  // Getter functions.
  mlir::tt::MeshShardDirection getShardDirection() const {
    return shardDirection;
  }
  mlir::tt::MeshShardType getShardType() const { return shardType; }
  llvm::ArrayRef<int64_t> getShardShape() const { return shardShape; }
  llvm::ArrayRef<int64_t> getShardDims() const { return shardDims; }
  llvm::ArrayRef<int64_t> getMeshShape() const { return meshShape; }

private:
  // Parse GSPMD devices string and fill out MeshSharding info.
  llvm::Expected<bool> parseGSPMDDevicesStr(StringRef devicesStr);

  // Based on current MeshSharding info, finalize sharding dimensions.
  llvm::Expected<bool> determineGSPMDShardingDims();

  // Set sharyType other than devices and reset values.
  void setNonDevicesShardType(tt::MeshShardType targetShardType) {
    assert(targetShardType != tt::MeshShardType::Devices);
    shardType = targetShardType;
    // Specific values are required to fill corresponding attributes in
    // mesh_shard operation.
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

// Sharding related string definitions from open-xla
// https://github.com/openxla/xla/blob/main/xla/service/spmd/shardy/constants.h

inline constexpr llvm::StringRef kShardingCustomCallTargetName = "Sharding";
inline constexpr llvm::StringRef kSPMDFullToShardShapeCallTargetName =
    "SPMDFullToShardShape";
inline constexpr llvm::StringRef kSPMDShardToFullShapeCallTargetName =
    "SPMDShardToFullShape";
inline constexpr llvm::StringRef kXlaShardingAttr = "mhlo.sharding";

#endif

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
