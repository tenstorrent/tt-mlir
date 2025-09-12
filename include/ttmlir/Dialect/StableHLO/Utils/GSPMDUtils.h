// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_GSPMDUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_GSPMDUTILS_H

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

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
inline constexpr llvm::StringRef kFrontendAttributesAttr =
    "mhlo.frontend_attributes";

// Parse meshes from the GSPMD module.
llvm::Expected<llvm::SmallVector<llvm::SmallVector<int64_t>>>
parseMeshesFromGspmdModule(mlir::ModuleOp &module);

// Check if the module has any gspmd annotations.
bool gspmdAnnotationsExist(mlir::ModuleOp &module);

// Check if the module has frontend SDY attributes.
bool hasFrontendSdyAttributes(mlir::ModuleOp &module);

// Update @Sharding custom call with the shard status for the argument.
void updateShardStatusForArgument(MLIRContext *context,
                                  mlir::BlockArgument &arg,
                                  mlir::NamedAttribute shardStatusNamedAttr);

// Update @Sharding custom call with the shard status for the result.
void updateShardStatusForResult(MLIRContext *context, func::FuncOp &funcOp,
                                uint32_t resultIdx,
                                mlir::NamedAttribute shardStatusNamedAttr);

// Parse axis definitions from SDY mesh string format.
std::vector<std::pair<std::string, int64_t>>
parseAxisDefinitions(const std::string &axesContent);

// Parse mesh information from mhlo.frontend_attributes and create sdy.mesh.
mlir::LogicalResult parseMeshFromFrontendAttributes(mlir::ModuleOp &rootModule,
                                                    mlir::MLIRContext *context);

class GSPMDMeshSharding : public sharding_utils::MeshSharding {
public:
  // Static factory methods.
  static llvm::Expected<GSPMDMeshSharding> generateDefault();
  static llvm::Expected<GSPMDMeshSharding>
  generate(llvm::StringRef opShardingStr, llvm::StringRef operandShardingStr,
           mlir::tt::ttcore::ShardStatus shardStatus,
           mlir::tt::ttcore::MeshShardDirection shardDirection);
  GSPMDMeshSharding(mlir::tt::ttcore::MeshShardDirection shardDirection,
                    mlir::tt::ttcore::MeshShardType shardType,
                    const llvm::SmallVector<int64_t> &shardShape,
                    const llvm::SmallVector<int64_t> &shardDims,
                    const llvm::SmallVector<int64_t> &meshShape,
                    const llvm::SmallVector<int64_t> &deviceIds,
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

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_GSPMDUTILS_H
