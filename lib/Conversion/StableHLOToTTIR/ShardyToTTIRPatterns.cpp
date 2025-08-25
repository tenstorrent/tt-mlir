// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"

#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Utils/Mesh.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

// This class is used to cache analysis performed on manual computation op.
// It will cache the shard status of arguments and results of the manual
// computation op. This is useful to avoid re-analyzing the same manual
// computation op multiple times.
class ManualComputationAnalysisCache {
public:
  ManualComputationAnalysisCache(
      llvm::DenseMap<mlir::Value, mlir::tt::ttcore::ShardStatus>
          shardStatusCache)
      : shardStatusCache(shardStatusCache) {}

  static ManualComputationAnalysisCache
  generate(mlir::sdy::ManualComputationOp &op) {
    llvm::DenseMap<mlir::Value, mlir::tt::ttcore::ShardStatus> shardStatusMap;

    // Iterate through all the operands of the manual computation op and
    // determine the shard status of each argument.
    for (auto arg : op.getOperands()) {
      // We need to backtrace the argument to its owning operation to find the
      // shard status. The shard status is stored in the operation's
      // attribute dictionary.
      mlir::tt::ttcore::ShardStatus shardStatus =
          mlir::tt::ttcore::ShardStatus::Unsharded; // Default to unsharded.

      if (!mlir::isa<mlir::BlockArgument>(arg)) {
        shardStatusMap[arg] = shardStatus;
        continue;
      }

      mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(arg);
      mlir::Operation *owningOp = blockArg.getOwner()->getParentOp();
      if (auto funcOp = mlir::dyn_cast<mlir::FunctionOpInterface>(owningOp)) {
        unsigned argIndex = blockArg.getArgNumber();

        // Retrieve the attribute dictionary and store the shard status.
        mlir::DictionaryAttr argAttrs = mlir::DictionaryAttr::get(
            op.getContext(), funcOp.getArgAttrs(argIndex));
        if (argAttrs) {
          auto shardStatusAttr =
              argAttrs.get(mlir::tt::ttcore::ShardStatusAttr::name);
          if (shardStatusAttr) {
            shardStatus =
                mlir::cast<mlir::tt::ttcore::ShardStatusAttr>(shardStatusAttr)
                    .getValue();
          }
        }
      }

      shardStatusMap[arg] = shardStatus;
    }

    // Iterate through all the results of the manual computation op and
    // determine the shard status of each result.
    for (auto result : op.getResults()) {
      mlir::tt::ttcore::ShardStatus shardStatus =
          mlir::tt::ttcore::ShardStatus::Unsharded; // Default to unsharded.

      // We find all the users of the manual computation op result.
      // If the result is used in a return op, we can find the shard status
      // from the result attributes of the return op.
      for (mlir::Operation *user : result.getUsers()) {
        if (!mlir::isa<mlir::func::ReturnOp>(user)) {
          continue;
        }

        // Find the operand index in the return
        mlir::func::ReturnOp returnOp = mlir::cast<mlir::func::ReturnOp>(user);
        for (auto [i, operand] : llvm::enumerate(returnOp.getOperands())) {
          // Go up to the parent function
          auto funcOp = returnOp->getParentOfType<mlir::FunctionOpInterface>();
          if (!funcOp) {
            continue;
          }

          // Get the result attributes for that return index
          auto resultAttrs = mlir::DictionaryAttr::get(
              op.getContext(), funcOp.getResultAttrs(i));
          if (!resultAttrs) {
            continue;
          }

          // Look for shard status
          auto shardStatusAttr =
              resultAttrs.get(mlir::tt::ttcore::ShardStatusAttr::name);
          if (shardStatusAttr) {
            shardStatus =
                mlir::cast<mlir::tt::ttcore::ShardStatusAttr>(shardStatusAttr)
                    .getValue();
          }
        }

        shardStatusMap[result] = shardStatus;
      }
    }

    return ManualComputationAnalysisCache(shardStatusMap);
  }

  mlir::tt::ttcore::ShardStatus getShardStatus(mlir::Value arg) const {
    auto it = this->shardStatusCache.find(arg);
    if (it != this->shardStatusCache.end()) {
      return it->second;
    }
    return mlir::tt::ttcore::ShardStatus::Unsharded;
  }

public:
  llvm::DenseMap<mlir::Value, mlir::tt::ttcore::ShardStatus> shardStatusCache;
};

class ShardyToTTIRManualComputationOpConversionPattern
    : public mlir::OpConversionPattern<mlir::sdy::ManualComputationOp> {
  using mlir::OpConversionPattern<
      mlir::sdy::ManualComputationOp>::OpConversionPattern;

public:
  llvm::LogicalResult
  matchAndRewrite(mlir::sdy::ManualComputationOp srcOp,
                  mlir::sdy::ManualComputationOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = srcOp->getParentOfType<mlir::ModuleOp>();
    mlir::Location loc = srcOp.getLoc();

    // Cache all the shard status analysis performed on the manual computation
    // op for its arguments and results.
    ManualComputationAnalysisCache cache =
        ManualComputationAnalysisCache::generate(srcOp);

    // Get the sdy mesh from the module.
    llvm::SmallVector<mlir::sdy::MeshOp> parsedMeshOps =
        shardy_utils::getMeshOps(module);

    if (parsedMeshOps.size() > 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "TTMLIR compiler only supports single mesh.");
    }

    // Add mesh_shard (FullToShardShape) for inputs.
    rewriter.setInsertionPoint(srcOp);
    llvm::SmallVector<mlir::Value> fullToShardResults;
    for (auto [globalOperand, argSharding, localArgType] : llvm::zip_equal(
             adaptor.getOperands(), srcOp.getInShardings().getShardings(),
             srcOp.getBody().getArgumentTypes())) {

      // Once extracted, we can generate the ShardyMeshSharding object.
      llvm::Expected<mlir::tt::shardy_utils::ShardyMeshSharding>
          shardyMeshSharding =
              mlir::tt::shardy_utils::ShardyMeshSharding::generate(
                  parsedMeshOps[0].getMeshAttr(), argSharding,
                  cache.getShardStatus(globalOperand),
                  mlir::tt::ttcore::MeshShardDirection::FullToShard);
      if (auto err = shardyMeshSharding.takeError()) {
        return rewriter.notifyMatchFailure(
            srcOp, "Error trying to parse shardy annotation.");
      }

      // Create a new mesh shard op.
      auto outputType = mlir::cast<mlir::RankedTensorType>(
          getTypeConverter()->convertType(localArgType));
      auto meshShardOp = rewriter.create<mlir::tt::ttir::MeshShardOp>(
          loc, outputType, globalOperand, shardyMeshSharding->getShardType(),
          shardyMeshSharding->getShardDirection(),
          shardyMeshSharding->getShardShape(),
          shardyMeshSharding->getShardDims());
      fullToShardResults.push_back(meshShardOp.getResult());
    }

    // Add mesh_shard (ShardToFullShape) for outputs.
    rewriter.setInsertionPointAfter(srcOp);
    llvm::SmallVector<mlir::Value> shardToFullResults;
    mlir::Operation *sdyReturn = mlir::sdy::getBodyTerminator(srcOp);
    for (auto [retIdx, args] : llvm::enumerate(llvm::zip_equal(
             sdyReturn->getOpOperands(), srcOp.getOutShardings().getShardings(),
             srcOp.getResults()))) {
      auto [returnOperand, outSharding, opResult] = args;

      // Once extracted, we can generate the ShardyMeshSharding object.
      llvm::Expected<mlir::tt::shardy_utils::ShardyMeshSharding>
          shardyMeshSharding =
              mlir::tt::shardy_utils::ShardyMeshSharding::generate(
                  parsedMeshOps[0].getMeshAttr(), outSharding,
                  cache.getShardStatus(opResult),
                  mlir::tt::ttcore::MeshShardDirection::ShardToFull);
      if (auto err = shardyMeshSharding.takeError()) {
        return rewriter.notifyMatchFailure(
            srcOp, "Error trying to parse shardy annotation.");
      }

      // Create a new mesh shard op.
      auto outputType = mlir::cast<mlir::RankedTensorType>(
          getTypeConverter()->convertType(opResult.getType()));
      auto meshShardOp = rewriter.create<mlir::tt::ttir::MeshShardOp>(
          loc, outputType, returnOperand.get(),
          shardyMeshSharding->getShardType(),
          shardyMeshSharding->getShardDirection(),
          shardyMeshSharding->getShardShape(),
          shardyMeshSharding->getShardDims());
      shardToFullResults.push_back(meshShardOp.getResult());
    }

    // Inline inner block ops.
    rewriter.inlineBlockBefore(&srcOp.getBody().front(), srcOp,
                               fullToShardResults);
    rewriter.eraseOp(sdyReturn);
    rewriter.replaceOp(srcOp, shardToFullResults);

    return llvm::success();
  }
};

class ShardyToTTIRMeshOpConversionPattern
    : public mlir::OpConversionPattern<mlir::sdy::MeshOp> {
  using mlir::OpConversionPattern<mlir::sdy::MeshOp>::OpConversionPattern;

public:
  llvm::LogicalResult
  matchAndRewrite(mlir::sdy::MeshOp srcOp, mlir::sdy::MeshOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Create a ttir mesh attribute from the sdy mesh attribute.
    mlir::tt::ttcore::MeshAttr ttMeshAttr =
        shardy_utils::createTTMeshAttrFromSdyMeshOp(srcOp);

    // Add a list of ttir mesh attributes to the rootModule.
    mlir::ModuleOp module = srcOp->getParentOfType<mlir::ModuleOp>();
    llvm::SmallVector<mlir::tt::ttcore::MeshAttr> meshes;
    if (auto meshesAttr = module->getAttrOfType<mlir::tt::ttcore::MeshesAttr>(
            mlir::tt::ttcore::MeshesAttr::name)) {
      meshes =
          llvm::SmallVector<mlir::tt::ttcore::MeshAttr>(meshesAttr.getMeshes());
    }

    // Avoid adding the same mesh multiple times.
    if (llvm::all_of(meshes, [&](mlir::tt::ttcore::MeshAttr m) {
          return m.getName() != ttMeshAttr.getName();
        })) {
      meshes.push_back(mlir::tt::ttcore::MeshAttr::get(
          getContext(), ttMeshAttr.getName(), ttMeshAttr.getShape()));
      rewriter.modifyOpInPlace(module, [&]() {
        module->setAttr(
            mlir::tt::ttcore::MeshesAttr::name,
            mlir::tt::ttcore::MeshesAttr::get(getContext(), meshes));
      });
    }

    rewriter.eraseOp(srcOp);
    return llvm::success();
  }
};

} // namespace

namespace mlir::tt {

void populateShardyToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  patterns.add<ShardyToTTIRManualComputationOpConversionPattern>(typeConverter,
                                                                 ctx);
  patterns.add<ShardyToTTIRMeshOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
