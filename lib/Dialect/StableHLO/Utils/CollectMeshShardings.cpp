// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Utils/CollectMeshShardings.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::sharding_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

namespace {

// Resolve one func boundary (arg or result) into a MeshSharding by reading
// its attribute dictionary. Shardy wins when both annotation families are
// present. Falls back to a Replicate placeholder when neither is present.
// Shard status is taken from the explicit ttcore.shard_status attribute when
// set; otherwise it is deduced from sharding-annotation presence, matching
// what ApplyArgumentShardStatus would write.
llvm::Expected<MeshSharding>
resolveBoundary(mlir::DictionaryAttr attributes,
                mlir::sdy::MeshAttr meshAttribute,
                ttcore::MeshShardDirection direction) {
  auto lookup = [&](llvm::StringRef name) -> mlir::Attribute {
    return attributes ? attributes.get(name) : nullptr;
  };
  auto shardyAttribute = mlir::dyn_cast_or_null<mlir::sdy::TensorShardingAttr>(
      lookup(mlir::sdy::kShardingAttr));
  auto gspmdAttribute = mlir::dyn_cast_or_null<mlir::StringAttr>(
      lookup(gspmd_utils::kXlaShardingAttr));
  auto statusAttribute = mlir::dyn_cast_or_null<ttcore::ShardStatusAttr>(
      lookup(ttcore::ShardStatusAttr::name));

  ttcore::ShardStatus status = statusAttribute ? statusAttribute.getValue()
                               : (shardyAttribute || gspmdAttribute)
                                   ? ttcore::ShardStatus::Presharded
                                   : ttcore::ShardStatus::Unsharded;

  if (shardyAttribute && meshAttribute && !meshAttribute.empty()) {
    auto generated = shardy_utils::ShardyMeshSharding::generate(
        meshAttribute, shardyAttribute, status, direction);
    if (auto error = generated.takeError()) {
      return std::move(error);
    }
    return MeshSharding(*generated);
  }
  if (gspmdAttribute) {
    auto generated = gspmd_utils::GSPMDMeshSharding::generate(
        gspmdAttribute.getValue(), gspmdAttribute.getValue(), status,
        direction);
    if (auto error = generated.takeError()) {
      return std::move(error);
    }
    return MeshSharding(*generated);
  }
  return MeshSharding(direction, ttcore::MeshShardType::Replicate, {}, {}, {},
                      {}, status);
}

mlir::sdy::MeshAttr firstMeshAttribute(mlir::func::FuncOp funcOp) {
  mlir::ModuleOp module = funcOp->getParentOfType<mlir::ModuleOp>();
  llvm::SmallVector<mlir::sdy::MeshOp> meshOps =
      shardy_utils::getMeshOps(module);
  return meshOps.empty() ? nullptr : meshOps.front().getMesh();
}

} // namespace

llvm::Expected<llvm::SmallVector<MeshSharding, 0>>
collectArgMeshShardings(mlir::func::FuncOp funcOp) {
  mlir::sdy::MeshAttr meshAttribute = firstMeshAttribute(funcOp);
  llvm::SmallVector<MeshSharding, 0> result;
  result.reserve(funcOp.getNumArguments());
  for (unsigned argIndex = 0; argIndex < funcOp.getNumArguments(); ++argIndex) {
    auto resolved =
        resolveBoundary(funcOp.getArgAttrDict(argIndex), meshAttribute,
                        ttcore::MeshShardDirection::FullToShard);
    if (auto error = resolved.takeError()) {
      return std::move(error);
    }
    result.push_back(*resolved);
  }
  return result;
}

llvm::Expected<llvm::SmallVector<MeshSharding, 0>>
collectResultMeshShardings(mlir::func::FuncOp funcOp) {
  mlir::sdy::MeshAttr meshAttribute = firstMeshAttribute(funcOp);
  llvm::SmallVector<MeshSharding, 0> result;
  result.reserve(funcOp.getNumResults());
  for (unsigned resultIndex = 0; resultIndex < funcOp.getNumResults();
       ++resultIndex) {
    auto resolved =
        resolveBoundary(funcOp.getResultAttrDict(resultIndex), meshAttribute,
                        ttcore::MeshShardDirection::ShardToFull);
    if (auto error = resolved.takeError()) {
      return std::move(error);
    }
    result.push_back(*resolved);
  }
  return result;
}

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::sharding_utils
