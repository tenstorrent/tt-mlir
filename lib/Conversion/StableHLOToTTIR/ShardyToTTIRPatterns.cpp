// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"

#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/Utils/Mesh.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace {

class ShardyToTTIRManualComputationOpConversionPattern
    : public mlir::OpConversionPattern<mlir::sdy::ManualComputationOp> {
  using mlir::OpConversionPattern<
      mlir::sdy::ManualComputationOp>::OpConversionPattern;

public:
  llvm::LogicalResult
  matchAndRewrite(mlir::sdy::ManualComputationOp srcOp,
                  mlir::sdy::ManualComputationOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = srcOp->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      llvm_unreachable("mlir::sdy::ManualComputationOp requires module as one "
                       "of parent ops.");
    }
    mlir::SymbolTable symbolTable(module);
    mlir::Location loc = srcOp.getLoc();

    auto shardings = llvm::concat<const mlir::sdy::TensorShardingAttr>(
        srcOp.getInShardings().getShardings(),
        srcOp.getOutShardings().getShardings());
    if (shardings.begin() == shardings.end()) {
      // Inline the body with no in/out shardings.
      rewriter.eraseOp(getBodyTerminator(srcOp));
      rewriter.inlineBlockBefore(&srcOp.getBody().front(), srcOp,
                                 srcOp.getOperands());
      rewriter.eraseOp(srcOp);
      return llvm::success();
    }

    // ManualComputationOp include one mesh for all in/out shardings, so we can
    // pick up first sharding and get mesh info.
    mlir::sdy::TensorShardingAttr firstSharding = *shardings.begin();
    mlir::sdy::MeshAttr targetMesh = firstSharding.getMesh(symbolTable);
    if (!targetMesh) {
      llvm_unreachable(
          "mlir::sdy::TensorShardingAttr requires mesh definition.");
    }

    // Currently, sharding operation on device memory is not supported, so
    // remove any sharding in body of manual computation op and compute
    // with replicated tensor.
    srcOp.getBody().front().walk<mlir::WalkOrder::PreOrder>(
        [&](mlir::Operation *opInBody) {
          if (mlir::isa<mlir::sdy::ManualComputationOp>(opInBody)) {
            return mlir::WalkResult::skip();
          }
          mlir::sdy::TensorShardingPerValueAttr shardingPerValue =
              opInBody->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
                  mlir::sdy::kShardingAttr);
          if (!shardingPerValue) {
            return mlir::WalkResult::advance();
          }
          rewriter.modifyOpInPlace(opInBody, [&]() {
            opInBody->removeAttr(mlir::sdy::kShardingAttr);
          });
          return mlir::WalkResult::advance();
        });

    auto funcOp = srcOp->getParentOfType<mlir::func::FuncOp>();

    // Add mesh_shard (FullToShardShape) for inputs.
    llvm::SmallVector<mlir::Value> fullToShardResults;
    for (auto [globalOperand, argSharding, localArgType] : llvm::zip_equal(
             srcOp.getOperands(), srcOp.getInShardings().getShardings(),
             srcOp.getBody().getArgumentTypes())) {

      mlir::tt::sharding_utils::MeshSharding meshSharding;
      auto error = meshSharding.convertSdyShardingToMeshSharding(
          argSharding, targetMesh, mlir::tt::MeshShardDirection::FullToShard);
      if (auto e = error.takeError()) {
        return rewriter.notifyMatchFailure(srcOp, llvm::toString(std::move(e)));
      }

      // JAX automatic sharding pre-shards input tensors and provides multiple
      // buffers. Thus, we have to check if mesh shard op is sharding the
      // tensors twice. We create dummy mesh shard op if input and output
      // shapes are different or not create mesh shard op if they are
      // identical.
      bool shouldCreateMeshShardOp =
          meshSharding.checkAndUpdateShardyArgSharding(
              rewriter, funcOp, globalOperand, argSharding);
      if (shouldCreateMeshShardOp) {
        auto outputType = mlir::cast<mlir::RankedTensorType>(
            getTypeConverter()->convertType(localArgType));

        auto meshShardOp =
            ttmlir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
                rewriter, loc, outputType, globalOperand,
                meshSharding.getShardType(), meshSharding.getShardDirection(),
                meshSharding.getShardShape(), meshSharding.getShardDims());

        fullToShardResults.push_back(meshShardOp.getResult());
      } else {
        // Do not create mesh shard op if input and output shapes are
        // identical: frontend provides sharded input and shard type is
        // replicate.
        fullToShardResults.push_back(globalOperand);
      }
    }

    // Add mesh_shard (ShardToFullShape) for outputs.
    rewriter.setInsertionPointAfter(srcOp);
    mlir::Operation *sdyReturn = getBodyTerminator(srcOp);
    for (auto [retIdx, args] : llvm::enumerate(llvm::zip_equal(
             sdyReturn->getOpOperands(), srcOp.getOutShardings().getShardings(),
             srcOp.getResults()))) {
      auto [returnOperand, outSharding, opResult] = args;
      mlir::tt::sharding_utils::MeshSharding meshSharding;
      auto error = meshSharding.convertSdyShardingToMeshSharding(
          outSharding, targetMesh, mlir::tt::MeshShardDirection::ShardToFull);
      if (auto e = error.takeError()) {
        return rewriter.notifyMatchFailure(srcOp, llvm::toString(std::move(e)));
      }

      // JAX automatic sharding may expect pre-sharded output tensors. We should
      // check and update mesh shard op to match frontend's expectation. We may
      // create dummy mesh shard op even though frontend expect sharded return
      // in case input and output shapes of mesh shard op are different.
      bool shouldCreateMeshShardOp =
          meshSharding.checkAndUpdateShardyRetSharding(rewriter, funcOp, retIdx,
                                                       outSharding);
      if (shouldCreateMeshShardOp) {
        auto inputOperand = returnOperand.get();
        auto inputType = mlir::cast<mlir::RankedTensorType>(
            getTypeConverter()->convertType(inputOperand.getType()));
        if (inputType != inputOperand.getType()) {
          inputOperand.setType(inputType);
        }

        auto outputType = mlir::cast<mlir::RankedTensorType>(
            getTypeConverter()->convertType(opResult.getType()));

        auto meshShardOp =
            ttmlir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
                rewriter, loc, outputType, inputOperand,
                meshSharding.getShardType(), meshSharding.getShardDirection(),
                meshSharding.getShardShape(), meshSharding.getShardDims());

        rewriter.replaceAllUsesWith(opResult, meshShardOp.getResult());
      } else {
        // Do not create mesh shard op if input and output shapes are identical:
        // frontend expects sharded return and shard type is replicate.
        rewriter.replaceAllUsesWith(opResult, returnOperand.get());
      }
    }

    // Inline inner block ops.
    rewriter.inlineBlockBefore(&srcOp.getBody().front(), srcOp,
                               fullToShardResults);
    rewriter.eraseOp(sdyReturn);
    rewriter.eraseOp(srcOp);

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
    // The main goal of this conversion is to extract hardware mesh information
    // from sdy.mesh op and store it as module attribute.
    auto module = srcOp->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      llvm_unreachable(
          "mlir::sdy::MeshOp requires module as one of parent ops.");
    }

    mlir::StringAttr meshName = srcOp.getSymNameAttr();
    llvm::SmallVector<int64_t> meshShape;
    mlir::sdy::MeshAttr sdyMesh = srcOp.getMesh();
    for (auto meshAxisAttr : sdyMesh.getAxes()) {
      meshShape.push_back(meshAxisAttr.getSize());
    }
    mlir::tt::utils::addMeshToModuleAttribute(rewriter, module, meshName,
                                              meshShape);

    // Before erasing MeshOp, visit public functions and erase argument sharding
    // attributes that are not refered by ManualComputationOp. Ones that are
    // refered by ManualComputationOp are properly handled by
    // ShardyToTTIRManualComputationOpConversionPattern.
    module->walk([&](mlir::func::FuncOp funcOp) {
      if (funcOp.isPublic()) {
        for (auto arg : funcOp.getArguments()) {
          auto argIdx = arg.getArgNumber();
          auto argShardingAttr =
              funcOp.getArgAttrOfType<mlir::sdy::TensorShardingAttr>(
                  argIdx, mlir::sdy::kShardingAttr);
          if (!argShardingAttr) {
            continue;
          }
          if (llvm::any_of(arg.getUsers(), [&](mlir::Operation *user) {
                return mlir::dyn_cast_if_present<
                    mlir::sdy::ManualComputationOp>(*user);
              })) {
            continue;
          }
          rewriter.modifyOpInPlace(funcOp, [&]() {
            funcOp.removeArgAttr(argIdx, mlir::sdy::kShardingAttr);
          });
        }
      }
    });

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
