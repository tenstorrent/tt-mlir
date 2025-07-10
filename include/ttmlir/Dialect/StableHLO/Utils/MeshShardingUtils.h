// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

namespace mlir::tt::sharding_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Sharding related string definitions from open-xla
// https://github.com/openxla/xla/blob/main/xla/service/spmd/shardy/constants.h
inline constexpr llvm::StringRef kShardingCustomCallTargetName = "Sharding";
inline constexpr llvm::StringRef kSPMDFullToShardShapeCallTargetName =
    "SPMDFullToShardShape";
inline constexpr llvm::StringRef kSPMDShardToFullShapeCallTargetName =
    "SPMDShardToFullShape";
inline constexpr llvm::StringRef kXlaShardingAttr = "mhlo.sharding";

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

class GSPMDMeshSharding : public MeshSharding {
public:
  // Static factory method
  static llvm::Expected<GSPMDMeshSharding>
  generate(llvm::StringRef opShardingStr, llvm::StringRef operandShardingStr, mlir::tt::ttcore::MeshShardDirection shardDirection,
           mlir::tt::ttcore::ShardStatus shardStatus);
  GSPMDMeshSharding(mlir::tt::ttcore::MeshShardDirection shardDirection,
                    mlir::tt::ttcore::MeshShardType shardType,
                    llvm::SmallVector<int64_t> shardShape,
                    llvm::SmallVector<int64_t> shardDims,
                    llvm::SmallVector<int64_t> meshShape,
                    llvm::SmallVector<int64_t> deviceIds,
                    mlir::tt::ttcore::ShardStatus shardStatus,
                    std::string opShardingStr, std::string operandShardingStr,
                    bool lastTileDimReplicate)
      : MeshSharding(shardDirection, shardType, shardShape, shardDims,
                     meshShape, deviceIds, shardStatus),
        lastTileDimReplicate(lastTileDimReplicate),
        opShardingStr(opShardingStr), operandShardingStr(operandShardingStr) {}

  // Getters
  bool getLastTileDimReplicate() const { return lastTileDimReplicate; }
  std::string getOpShardingStr() const { return opShardingStr; }
  std::string getOperandShardingStr() const { return operandShardingStr; }

private:
  // Member variables
  bool lastTileDimReplicate;
  std::string opShardingStr;
  std::string operandShardingStr;
};

class ShardyMeshSharding : public MeshSharding {
public:
  // Static factory method
  static llvm::Expected<ShardyMeshSharding>
  generate(sdy::MeshAttr meshAttr, sdy::TensorShardingAttr sdySharding,
           ttcore::MeshShardDirection shardDirection,
           mlir::tt::ttcore::ShardStatus shardStatus);
  ShardyMeshSharding(mlir::tt::ttcore::MeshShardDirection shardDirection,
                     mlir::tt::ttcore::MeshShardType shardType,
                     llvm::SmallVector<int64_t> shardShape,
                     llvm::SmallVector<int64_t> shardDims,
                     llvm::SmallVector<int64_t> meshShape,
                     llvm::SmallVector<int64_t> deviceIds,
                     mlir::tt::ttcore::ShardStatus shardStatus,
                     sdy::MeshAttr meshAttr,
                     sdy::TensorShardingAttr sdySharding)
      : MeshSharding(shardDirection, shardType, shardShape, shardDims,
                     meshShape, deviceIds, shardStatus),
        meshAttr(meshAttr), sdySharding(sdySharding) {}

  // Getters
  mlir::sdy::MeshAttr getMeshAttr() const { return meshAttr; }
  mlir::sdy::TensorShardingAttr getSdySharding() const { return sdySharding; }

private:
  // Member variables
  mlir::sdy::MeshAttr meshAttr;
  mlir::sdy::TensorShardingAttr sdySharding;
};

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
