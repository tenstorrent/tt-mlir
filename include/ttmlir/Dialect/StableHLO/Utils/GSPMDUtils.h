// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_GSPMDUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_GSPMDUTILS_H

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/MeshShardingUtils.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#pragma clang diagnostic pop

namespace mlir::tt::gspmd_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Sharding related string definitions from open-xla
// https://github.com/openxla/xla/blob/main/xla/service/spmd/shardy/constants.h
inline constexpr llvm::StringRef kShardingCustomCallTargetName = "Sharding";
inline constexpr llvm::StringRef kSPMDFullToShardShapeCallTargetName =
    "SPMDFullToShardShape";
inline constexpr llvm::StringRef kSPMDShardToFullShapeCallTargetName =
    "SPMDShardToFullShape";
inline constexpr llvm::StringRef kXlaShardingAttr = "mhlo.sharding";

// Parse meshes from the GSPMD module.
llvm::Expected<llvm::SmallVector<llvm::SmallVector<int64_t>>>
parseMeshesFromGspmdModule(mlir::ModuleOp &module);

// Check if the module has any gspmd annotations.
bool gspmdAnnotationsExist(mlir::ModuleOp &module);

class GSPMDMeshSharding : public mesh_sharding_utils::MeshSharding {
public:
  // Static factory method
  static llvm::Expected<GSPMDMeshSharding>
  generate(llvm::StringRef opShardingStr, llvm::StringRef operandShardingStr,
           mlir::tt::ttcore::ShardStatus shardStatus);
  GSPMDMeshSharding(mlir::tt::ttcore::MeshShardDirection shardDirection,
                    mlir::tt::ttcore::MeshShardType shardType,
                    llvm::ArrayRef<int64_t> shardShape,
                    llvm::ArrayRef<int64_t> shardDims,
                    llvm::ArrayRef<int64_t> meshShape,
                    llvm::ArrayRef<int64_t> deviceIds,
                    mlir::tt::ttcore::ShardStatus shardStatus,
                    llvm::StringRef opShardingStr,
                    std::string operandShardingStr, bool lastTileDimReplicate)
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

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::gspmd_utils

#endif // TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_GSPMDUTILS_H
