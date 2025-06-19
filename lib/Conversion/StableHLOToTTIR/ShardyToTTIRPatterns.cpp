// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"

#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
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
    mlir::tt::ttcore::TensorMeshShardingAttr incomingTensorMeshShardingAttr) {
  auto type = mlir::cast<RankedTensorType>(valueType);
  mlir::Attribute encoding = type.getEncoding();
  if (auto existingTensorMeshShardingAttr =
          dyn_cast_if_present<mlir::tt::ttcore::TensorMeshShardingAttr>(
              encoding)) {
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
    mlir::tt::ttcore::TensorMeshShardingAttr incomingTensorMeshShardingAttr) {
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
    mlir::ValueRange values,
    mlir::tt::ttcore::TensorMeshShardingAttr tensorMeshShardingAttr) {
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
    OpTy srcOp,
    mlir::tt::ttcore::TensorMeshShardingAttr tensorMeshShardingAttr) {
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
    mlir::sdy::MeshAttr targetMesh =
        mlir::tt::sharding_utils::adjustSdyMeshAttr(
            srcOp, firstSharding.getMesh(symbolTable));
    if (!targetMesh) {
      llvm_unreachable(
          "mlir::sdy::TensorShardingAttr requires mesh definition.");
    }

    auto meshNameStrAttr =
        mlir::StringAttr::get(context, firstSharding.getMeshName());
    auto tensorMeshShardingAttr =
        mlir::tt::ttcore::TensorMeshShardingAttr::get(context, meshNameStrAttr);
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
             adaptor.getOperands(), srcOp.getInShardings().getShardings(),
             srcOp.getBody().getArgumentTypes())) {

      mlir::tt::sharding_utils::MeshSharding meshSharding;
      auto error = meshSharding.convertSdyShardingToMeshSharding(
          argSharding, targetMesh,
          mlir::tt::ttcore::MeshShardDirection::FullToShard);
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
            ttir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
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
          outSharding, targetMesh,
          mlir::tt::ttcore::MeshShardDirection::ShardToFull);
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
            mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
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
    mlir::sdy::MeshAttr sdyMesh =
        mlir::tt::sharding_utils::adjustSdyMeshAttr(srcOp, srcOp.getMesh());
    if (!sdyMesh.empty()) {
      llvm::SmallVector<int64_t> meshShape;
      for (auto meshAxisAttr : sdyMesh.getAxes()) {
        meshShape.push_back(meshAxisAttr.getSize());
      }
      mlir::tt::ttcore::utils::addMeshToModuleAttribute(rewriter, module,
                                                        meshName, meshShape);
    }

    // Before erasing MeshOp, visit public functions and properly handle
    // argument and return sharding attributes that are not used or defined by
    // ManualComputationOp, respectively. Ones that are refered by
    // ManualComputationOp are properly handled by
    // ShardyToTTIRManualComputationOpConversionPattern.
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
              return mlir::isa<mlir::sdy::ManualComputationOp>(*user);
            })) {
          continue;
        }

        mlir::tt::sharding_utils::MeshSharding meshSharding;
        auto error = meshSharding.convertSdyShardingToMeshSharding(
            argShardingAttr, sdyMesh,
            mlir::tt::ttcore::MeshShardDirection::ShardToFull);
        if (auto e = error.takeError()) {
          llvm_unreachable(llvm::toString(std::move(e)).c_str());
        }

        mlir::tt::sharding_utils::checkAndRemoveFuncArgSharding<
            mlir::sdy::TensorShardingAttr>(
            rewriter, funcOp, argIdx, argShardingAttr,
            meshSharding.getTensorMeshShardingAttr(rewriter),
            mlir::sdy::kShardingAttr);
      }

      mlir::Operation *funcReturnOp = funcOp.getBody().front().getTerminator();
      for (auto [retIdx, ret] : llvm::enumerate(funcReturnOp->getOperands())) {
        // Check arguments with sdy sharding attribute.
        auto retShardingAttr =
            funcOp.getResultAttrOfType<mlir::sdy::TensorShardingAttr>(
                retIdx, mlir::sdy::kShardingAttr);
        if (!retShardingAttr) {
          continue;
        }

        if (mlir::isa_and_present<mlir::sdy::ManualComputationOp>(
                ret.getDefiningOp())) {
          continue;
        }

        mlir::tt::sharding_utils::MeshSharding meshSharding;
        auto error = meshSharding.convertSdyShardingToMeshSharding(
            retShardingAttr, sdyMesh,
            mlir::tt::ttcore::MeshShardDirection::FullToShard);
        if (auto e = error.takeError()) {
          llvm_unreachable(llvm::toString(std::move(e)).c_str());
        }

        mlir::tt::sharding_utils::checkAndRemoveFuncReturnSharding<
            mlir::sdy::TensorShardingAttr>(
            rewriter, funcOp, retIdx, retShardingAttr,
            meshSharding.getTensorMeshShardingAttr(rewriter),
            mlir::sdy::kShardingAttr);
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

namespace mlir::tt {

// Check matching of input and result tensor annotations and fix if there is any
// issue.
llvm::LogicalResult
analyzeSingleOpAndFixWrongTensorAnnotation(mlir::tt::ttcore::MeshesAttr meshes,
                                           mlir::Operation *srcOp,
                                           mlir::OpBuilder &builder) {
  if (srcOp->getNumResults() == 0 || srcOp->getNumOperands() == 0) {
    return llvm::success();
  }

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType());
  if (auto tensorMeshShardingAttr =
          mlir::dyn_cast_if_present<mlir::tt::ttcore::TensorMeshShardingAttr>(
              resultType.getEncoding())) {
    // Result is multi device tensor and thus expects args to be multi device
    // tensors. If args are not multi device tensor, propagate the missing
    // TensorMeshShardingAttr to args.
    for (auto arg : srcOp->getOperands()) {
      auto argType = mlir::cast<mlir::RankedTensorType>(arg.getType());
      if (auto argTensorMeshShardingAttr = mlir::dyn_cast_if_present<
              mlir::tt::ttcore::TensorMeshShardingAttr>(
              argType.getEncoding())) {
        // If pre-existing TensorMeshShardingAttr is different from the
        // expected one, then fail.
        if (argTensorMeshShardingAttr.getName() !=
            tensorMeshShardingAttr.getName()) {
          srcOp->emitError()
              << "Inconsistency found in TensorMeshShardingAttr : expected ("
              << tensorMeshShardingAttr.getName() << ") and current ("
              << argTensorMeshShardingAttr.getName() << ") are different.";
        }
        continue;
      }
      mlir::Type elementType = argType.getElementType();
      llvm::ArrayRef<int64_t> shape = argType.getShape();
      arg.setType(mlir::RankedTensorType::get(shape, elementType,
                                              tensorMeshShardingAttr));
    }
  } else {
    // Result is single device tensor and thus expects args to be single
    // device tensors. If we find inconsistency due to one of the args with
    // multi-device tensor, insert MeshShardOp before the op and change the
    // tensor to single device tensor.
    for (auto arg : srcOp->getOperands()) {
      auto argType = mlir::cast<mlir::RankedTensorType>(arg.getType());
      if (auto argTensorMeshShardingAttr = mlir::dyn_cast_if_present<
              mlir::tt::ttcore::TensorMeshShardingAttr>(
              argType.getEncoding())) {
        mlir::tt::sharding_utils::MeshSharding meshSharding;
        meshSharding.extractMeshShardingFromTensorMeshShardingAttr(
            meshes.getMesh(argTensorMeshShardingAttr.getName().str()),
            argTensorMeshShardingAttr,
            mlir::tt::ttcore::MeshShardDirection::ShardToFull);

        mlir::Type elementType = argType.getElementType();
        llvm::ArrayRef<int64_t> shape = argType.getShape();

        builder.setInsertionPoint(srcOp);

        auto meshShardOp =
            mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
                builder, srcOp->getLoc(),
                mlir::RankedTensorType::get(shape, elementType), arg,
                meshSharding.getShardType(), meshSharding.getShardDirection(),
                meshSharding.getShardShape(), meshSharding.getShardDims());

        srcOp->replaceUsesOfWith(arg, meshShardOp.getResult());
      }
    }
  }
  return llvm::success();
}

// This pass is to clean up the tensor annotations in the module.
// In particular, we find two use cases:
// 1. Single device tensor op has multi device tensor argument in a single op.
// In this case, we need to insert MeshShardOp before the op and change the
// tensor to single device tensor.
// 2. Multi device tensor ops have missing multi device tensor annotation in
// case of automatic parallelism with pre-sharded inputs and outputs. In this
// case, we need to add missing TensorMeshShardingAttr.
class TTIRTensorAnnotationCleanupPass
    : public mlir::PassWrapper<TTIRTensorAnnotationCleanupPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRTensorAnnotationCleanupPass)

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = moduleOp.getContext();
    auto builder = mlir::OpBuilder(context);
    auto meshes = moduleOp->getAttrOfType<mlir::tt::ttcore::MeshesAttr>(
        mlir::tt::ttcore::MeshesAttr::name);

    // We regard a module without meshes as a module only containing single
    // device tensors. Thus, skip this pass.
    if (!meshes) {
      return;
    }

    // Visit all functions and analyze each op from backward in order to
    // determine either single or multidevcie computation context.
    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      llvm::SmallVector<mlir::Operation *> orderedOps;
      funcOp->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
          [&](mlir::Operation *op) {
            // ReturnOp is the root of this analysis and thus shouldn't be
            // touched. MeshShardOps are ones that convert single to multi
            // device tensor or vice versa, so do not need to be analyzed.
            if (mlir::isa<mlir::func::ReturnOp, mlir::tt::ttir::MeshShardOp>(
                    op)) {
              return mlir::WalkResult::skip();
            }
            orderedOps.push_back(op);
            return mlir::WalkResult::advance();
          });

      for (mlir::Operation *op : orderedOps) {
        if (failed(analyzeSingleOpAndFixWrongTensorAnnotation(meshes, op,
                                                              builder))) {
          signalPassFailure();
        }
      }
    }
  }

  llvm::StringRef getArgument() const override {
    return "shardy-tensor-annotation-cleanup";
  }

  llvm::StringRef getDescription() const override {
    return "Cleanup pass that fixes the multi-device tensor annotations.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

std::unique_ptr<Pass> createTTIRTensorAnnotationCleanupPass() {
  return std::make_unique<TTIRTensorAnnotationCleanupPass>();
}

} // namespace mlir::tt
