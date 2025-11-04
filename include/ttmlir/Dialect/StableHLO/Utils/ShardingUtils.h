// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGUTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::sharding_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// SDY sharding related string definitions.
inline constexpr llvm::StringRef kXlaSdyShardingAttr = "xla.sdy.sharding";
inline constexpr llvm::StringRef kXlaSdyMeshesAttr = "xla.sdy.meshes";
inline constexpr llvm::StringRef kDefaultMeshName = "mesh";

// Composite op flattening/re-outlining related string definitions.
inline constexpr llvm::StringLiteral kGroupAttr("reoutline.group");
inline constexpr llvm::StringLiteral kSeedAttr("reoutline.seed");
inline constexpr llvm::StringLiteral kOrigNameAttr("reoutline.orig_name");
inline constexpr llvm::StringLiteral kCompAttrsAttr("reoutline.comp_attrs");

inline const llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 7>
    SupportedMeshes = {
        {{1, 1}, {1, 2}, {1, 4}, {1, 8}, {2, 4}, {1, 32}, {8, 4}}};

// Check if the meshMap is valid.
// todo: https://github.com/tenstorrent/tt-mlir/issues/4668
// Currently we only support a few mesh shapes. We need to extend this to
// support more mesh shapes (like 1x3, 2x5 etc).
inline mlir::LogicalResult
checkValidMesh(llvm::SmallVector<int64_t> meshShape) {
  return mlir::success(
      llvm::is_contained(sharding_utils::SupportedMeshes, meshShape));
}

class MeshSharding {
public:
  // Constructors
  MeshSharding() = delete;
  MeshSharding(mlir::tt::ttcore::MeshShardDirection shardDirection,
               mlir::tt::ttcore::MeshShardType shardType,
               const llvm::SmallVector<int64_t> &shardShape,
               const llvm::SmallVector<int64_t> &shardDims,
               const llvm::SmallVector<int64_t> &meshShape,
               const llvm::SmallVector<int64_t> &deviceIds,
               mlir::tt::ttcore::ShardStatus shardStatus)
      : shardDirection(shardDirection), shardType(shardType),
        shardShape(shardShape), shardDims(shardDims), meshShape(meshShape),
        deviceIds(deviceIds) {}

  // Getters
  mlir::tt::ttcore::MeshShardDirection getShardDirection() const {
    return shardDirection;
  }
  mlir::tt::ttcore::MeshShardType getShardType() const { return shardType; }
  llvm::SmallVector<int64_t> getShardShape() const { return shardShape; }
  llvm::SmallVector<int64_t> getShardDims() const { return shardDims; }
  llvm::SmallVector<int64_t> getMeshShape() const { return meshShape; }
  llvm::SmallVector<int64_t> getDeviceIds() const { return deviceIds; }
  mlir::tt::ttcore::ShardStatus getShardStatus() const { return shardStatus; }

private:
  // Member variables
  mlir::tt::ttcore::MeshShardDirection shardDirection;
  mlir::tt::ttcore::MeshShardType shardType;
  llvm::SmallVector<int64_t> shardShape;
  llvm::SmallVector<int64_t> shardDims;
  llvm::SmallVector<int64_t> meshShape;
  llvm::SmallVector<int64_t> deviceIds;
  mlir::tt::ttcore::ShardStatus shardStatus;
};

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDINGUTILS_H
