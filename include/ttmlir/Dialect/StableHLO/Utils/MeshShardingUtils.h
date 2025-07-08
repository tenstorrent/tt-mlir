// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::mesh_sharding_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

inline const llvm::SmallVector<llvm::SmallVector<int64_t, 2>, 6>
    SupportedMeshes = {{{1, 1}, {1, 2}, {1, 8}, {2, 4}, {1, 32}, {8, 4}}};

// Check if the meshMap is valid.
inline mlir::LogicalResult
checkValidMesh(llvm::SmallVector<int64_t> meshShape) {
  if (!llvm::is_contained(mesh_sharding_utils::SupportedMeshes, meshShape)) {
    return mlir::failure();
  }

  return mlir::success();
}

class MeshSharding {
public:
  // Constructors
  MeshSharding() = delete;
  MeshSharding(mlir::tt::ttcore::MeshShardDirection shardDirection,
               mlir::tt::ttcore::MeshShardType shardType,
               llvm::SmallVector<int64_t> shardShape,
               llvm::SmallVector<int64_t> shardDims,
               llvm::SmallVector<int64_t> meshShape,
               llvm::SmallVector<int64_t> deviceIds,
               mlir::tt::ttcore::ShardStatus shardStatus)
      : shardDirection(shardDirection), shardType(shardType),
        shardShape(shardShape), shardDims(shardDims), meshShape(meshShape),
        deviceIds(deviceIds) {}

  // Getters
  mlir::tt::ttcore::MeshShardDirection getShardDirection() const {
    return shardDirection;
  }
  mlir::tt::ttcore::MeshShardType getShardType() const { return shardType; }
  llvm::ArrayRef<int64_t> getShardShape() const { return shardShape; }
  llvm::ArrayRef<int64_t> getShardDims() const { return shardDims; }
  llvm::ArrayRef<int64_t> getMeshShape() const { return meshShape; }
  llvm::ArrayRef<int64_t> getDeviceIds() const { return deviceIds; }
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

} // namespace mlir::tt::mesh_sharding_utils

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
