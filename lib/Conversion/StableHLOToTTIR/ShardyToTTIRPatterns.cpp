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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt {
// Helper functions for propagateTensorMeshSharding.

// Get callee funcOp from callOp.
static mlir::func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym =
      mlir::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym) {
    llvm_unreachable("Unable to find Callee of callOp.");
  }
  auto funcOp = mlir::dyn_cast_if_present<mlir::func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
  if (!funcOp) {
    llvm_unreachable("Unable to find funcOp.");
  }
  if (funcOp.isExternal()) {
    llvm_unreachable(
        "Unable to propagate TensorMeshShardingAttr to external function.");
  }
  return funcOp;
}

// Check if Type includes TensorMeshShardingAttr
static bool checkTypeIfTensorMeshShardingAttrExist(
    mlir::Type valueType,
    mlir::tt::TensorMeshShardingAttr incomingTensorMeshShardingAttr) {
  auto type = mlir::cast<RankedTensorType>(valueType);
  mlir::Attribute encoding = type.getEncoding();
  if (auto existingTensorMeshShardingAttr =
          dyn_cast_if_present<mlir::tt::TensorMeshShardingAttr>(encoding)) {
    if (existingTensorMeshShardingAttr.getName() !=
        incomingTensorMeshShardingAttr.getName()) {
      llvm_unreachable("Conflict mesh in TensorMeshShardingAttrs.");
    }
    return true;
  }
  return false;
}

// Check if FuncOp input/result include TensorMeshShardingAttr
static bool checkFuncOpIfTensorMeshShardingAttrExist(
    mlir::func::FuncOp funcOp,
    mlir::tt::TensorMeshShardingAttr incomingTensorMeshShardingAttr) {
  auto funcOpType = funcOp.getFunctionType();
  return (funcOpType.getInputs().size() > 0 &&
          checkTypeIfTensorMeshShardingAttrExist(
              funcOpType.getInput(0), incomingTensorMeshShardingAttr)) ||
         (funcOpType.getResults().size() > 0 &&
          checkTypeIfTensorMeshShardingAttrExist(
              funcOpType.getResult(0), incomingTensorMeshShardingAttr));
}

// Add tensorMeshShardingAttr to tensors and get updated types.
static llvm::SmallVector<Type> addTensorMeshShardingAttrToValues(
    mlir::ValueRange values, TensorMeshShardingAttr tensorMeshShardingAttr) {
  llvm::SmallVector<Type> types;
  if (values.size() == 0) {
    return types;
  }

  // If tensors already have TensorMeshShardingAttr, they should be identical
  // and we can skip adding it again.
  if (checkTypeIfTensorMeshShardingAttrExist(values[0].getType(),
                                             tensorMeshShardingAttr)) {
    return llvm::SmallVector<Type>(values.getTypes());
  }

  for (auto v : values) {
    types.push_back(mlir::tt::sharding_utils::addTensorMeshShardingAttrToValue(
        v, tensorMeshShardingAttr));
  }
  return types;
}

// Propagate TensorMeshShardingAttr to tensors in arg, body, return of
// manucalComputationOp and FuncOp. We assume that this function is being called
// inside body of rewriter.modifyOpInPlace().
template <typename OpTy>
void propagateTensorMeshSharding(
    OpTy srcOp, mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr) {
  static_assert(std::is_same_v<OpTy, mlir::sdy::ManualComputationOp> ||
                    std::is_same_v<OpTy, mlir::func::FuncOp>,
                "Only propagate TensorMeshSharding to "
                "mlir::sdy::ManualComputationOp or mlir::func::FuncOp.");

  // Propagate to sdy::ManualComputationOp args.
  if constexpr (std::is_same_v<OpTy, mlir::sdy::ManualComputationOp>) {
    addTensorMeshShardingAttrToValues(srcOp.getBody().getArguments(),
                                      tensorMeshShardingAttr);
  }

  // Visit ops in body and propagate TensorMeshShardingAttr.
  srcOp.getBody().front().template walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation *op) {
        if (mlir::isa<mlir::sdy::ManualComputationOp>(op) ||
            mlir::isa<mlir::func::ReturnOp>(op)) {
          return mlir::WalkResult::skip();
        }
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          auto funcOp = getCalledFunction(callOp);
          // Only visit the function that never visited.
          if (!checkFuncOpIfTensorMeshShardingAttrExist(
                  funcOp, tensorMeshShardingAttr)) {
            // Propagate to function args.
            auto inputTypes = addTensorMeshShardingAttrToValues(
                funcOp.getArguments(), tensorMeshShardingAttr);
            // Visit function in a recursive manner.
            mlir::tt::propagateTensorMeshSharding<mlir::func::FuncOp>(
                funcOp, tensorMeshShardingAttr);
            // Propagate to function returns.
            mlir::Operation *funcReturnOp =
                funcOp.getBody().front().getTerminator();
            llvm::SmallVector<Type> resultTypes;
            for (auto result : funcReturnOp->getOperands()) {
              resultTypes.push_back(result.getType());
            }
            // Update function signature.
            funcOp.setType(FunctionType::get(srcOp->getContext(), inputTypes,
                                             resultTypes));
          }
        }
        addTensorMeshShardingAttrToValues(op->getResults(),
                                          tensorMeshShardingAttr);
        return mlir::WalkResult::advance();
      });
}
} // namespace mlir::tt

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
    mlir::MLIRContext *context = rewriter.getContext();
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

    auto meshNameStrAttr =
        mlir::StringAttr::get(context, firstSharding.getMeshName());
    auto tensorMeshShardingAttr =
        mlir::tt::TensorMeshShardingAttr::get(context, meshNameStrAttr);
    rewriter.modifyOpInPlace(srcOp, [&]() {
      mlir::tt::propagateTensorMeshSharding<mlir::sdy::ManualComputationOp>(
          srcOp, tensorMeshShardingAttr);
    });

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
    llvm::SmallVector<mlir::Value> shardToFullResults;
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

        shardToFullResults.push_back(meshShardOp.getResult());
      } else {
        // Do not create mesh shard op if input and output shapes are identical:
        // frontend expects sharded return and shard type is replicate.
        shardToFullResults.push_back(returnOperand.get());
      }
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
    if (meshShape.size() < 2) {
      llvm_unreachable("1d hardware mesh is not supported.");
    }
    mlir::tt::utils::addMeshToModuleAttribute(rewriter, module, meshName,
                                              meshShape);

    // Before erasing MeshOp, visit public functions and properly handle
    // argument sharding attributes that are not used by ManualComputationOp or
    // MeshShardOp. Ones that are refered by ManualComputationOp are properly
    // handled by ShardyToTTIRManualComputationOpConversionPattern.
    module->walk([&](mlir::func::FuncOp funcOp) {
      if (!funcOp.isPublic()) {
        return mlir::WalkResult::skip();
      }
      for (auto arg : funcOp.getArguments()) {
        auto argIdx = arg.getArgNumber();
        // Check arguments with sdy sharding attribute.
        auto argShardingAttr =
            funcOp.getArgAttrOfType<mlir::sdy::TensorShardingAttr>(
                argIdx, mlir::sdy::kShardingAttr);
        if (!argShardingAttr) {
          continue;
        }
        if (llvm::any_of(arg.getUsers(), [&](mlir::Operation *user) {
              return mlir::isa<mlir::sdy::ManualComputationOp,
                               mlir::tt::ttir::MeshShardOp>(*user);
            })) {
          continue;
        }

        auto *firstUserOp = *arg.user_begin();
        rewriter.setInsertionPoint(firstUserOp);
        auto outputType = mlir::cast<mlir::RankedTensorType>(
            getTypeConverter()->convertType(arg.getType()));

        mlir::tt::sharding_utils::MeshSharding meshSharding;
        auto error = meshSharding.convertSdyShardingToMeshSharding(
            argShardingAttr, sdyMesh,
            mlir::tt::MeshShardDirection::ShardToFull);
        if (auto e = error.takeError()) {
          llvm_unreachable(llvm::toString(std::move(e)).c_str());
        }

        rewriter.modifyOpInPlace(funcOp, [&]() {
          funcOp.removeArgAttr(argIdx, mlir::sdy::kShardingAttr);
          mlir::tt::sharding_utils::addTensorMeshShardingAttrToFunctionArg(
              funcOp, argIdx, meshSharding.getTensorMeshShardingAttr(rewriter));
        });

        auto meshShardOp =
            ttmlir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
                rewriter, firstUserOp->getLoc(), outputType, arg,
                meshSharding.getShardType(), meshSharding.getShardDirection(),
                meshSharding.getShardShape(), meshSharding.getShardDims());

        rewriter.replaceAllUsesExcept(arg, meshShardOp.getResult(),
                                      meshShardOp);
      }
      return mlir::WalkResult::advance();
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
