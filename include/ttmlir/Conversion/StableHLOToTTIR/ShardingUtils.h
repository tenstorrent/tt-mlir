// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::sharding_utils {

#if defined(TTMLIR_ENABLE_STABLEHLO) && (TTMLIR_ENABLE_STABLEHLO != 0)

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

  // Check and update function arg sharding
  template <typename AttrType>
  void checkAndUpdateFuncArgSharding(mlir::PatternRewriter &rewriter,
                                     mlir::func::FuncOp funcOp, uint64_t argNum,
                                     AttrType shardingAttr,
                                     llvm::StringRef argShardingStrRef) {
    if (auto argShardingAttr =
            funcOp.getArgAttrOfType<AttrType>(argNum, argShardingStrRef)) {
      if (argShardingAttr == shardingAttr) {
        setDummyShardingOp();
        rewriter.modifyOpInPlace(
            funcOp, [&]() { funcOp.removeArgAttr(argNum, argShardingStrRef); });
      } else {
        llvm_unreachable(
            "MeshSharding operation and function argument shardings "
            "are different.");
      }
    }
  }

  // Check and update function ret sharding
  template <typename AttrType>
  void checkAndUpdateFuncReturnSharding(mlir::PatternRewriter &rewriter,
                                        mlir::func::FuncOp funcOp,
                                        uint64_t retNum, AttrType shardingAttr,
                                        llvm::StringRef retShardingStrRef) {
    if (auto retShardingAttr =
            funcOp.getResultAttrOfType<AttrType>(retNum, retShardingStrRef)) {
      if (retShardingAttr == shardingAttr) {
        setDummyShardingOp();
        rewriter.modifyOpInPlace(funcOp, [&]() {
          funcOp.removeResultAttr(
              retNum,
              mlir::StringAttr::get(rewriter.getContext(), retShardingStrRef));
        });
      } else {
        llvm_unreachable("MeshSharding operation and function return shardings "
                         "are different.");
      }
    }
  }

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

  // Force dummy sharding op by setting shard_type to manual. The mesh_shard op
  // will be ignored at runtime by simply copying input tensor to output.
  void setDummyShardingOp() { shardType = mlir::tt::MeshShardType::Manual; }

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
