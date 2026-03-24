// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/WeightDtypeParser.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNKVCACHEDTYPECONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNKVCacheDtypeConversionPass
    : public impl::TTNNKVCacheDtypeConversionBase<
          TTNNKVCacheDtypeConversionPass> {
public:
  using impl::TTNNKVCacheDtypeConversionBase<
      TTNNKVCacheDtypeConversionPass>::TTNNKVCacheDtypeConversionBase;

  static bool hasDtype(mlir::Type elType, ttcore::DataType dtype) {
    if (auto t = mlir::dyn_cast<ttcore::TileType>(elType)) {
      return t.getDataType() == dtype;
    }
    return ttcore::elementTypeToDataType(elType) == dtype;
  }

  static ttcore::DataType weightDtypeToDataType(WeightDtype wd) {
    switch (wd) {
    case WeightDtype::BFP_BFloat8:
      return ttcore::DataType::BFP_BFloat8;
    case WeightDtype::BFP_BFloat4:
      return ttcore::DataType::BFP_BFloat4;
    default:
      llvm_unreachable("Invalid WeightDtype for conversion");
    }
  }

  void runOnOperation() final {
    if (targetDtype == WeightDtype::None) {
      return;
    }

    ttcore::DataType dtype = weightDtypeToDataType(targetDtype);
    mlir::OpBuilder builder(&getContext());

    getOperation()->walk([&](mlir::func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return;
      }
      processFunc(funcOp, dtype, builder);
    });
  }

private:
  // For each kv_cache function argument:
  //   1. Create a private const-eval function containing the TypecastOp.
  //   2. Replace inline uses with a LoadCachedOp at the top of the entry block.
  void castKVCacheArgs(mlir::func::FuncOp funcOp, ttcore::DataType dtype,
                       mlir::OpBuilder &builder) {
    // Collect kv_cache args that need conversion.
    struct KVCastInfo {
      mlir::BlockArgument arg;
      RankedTensorType castType;
      std::string constEvalFuncName;
    };
    
    llvm::SmallVector<KVCastInfo> infos;

    for (auto arg : funcOp.getArguments()) {
      if (!funcOp.getArgAttr(arg.getArgNumber(), ttcore::g_kvCacheAttrName)) {
        continue;
      }

      auto argType = mlir::cast<RankedTensorType>(arg.getType());
      mlir::Type elType = argType.getElementType();

      // Skip if already the target dtype or unused.
      if (hasDtype(elType, dtype) || arg.use_empty()) {
        continue;
      }

      auto castType =
          ttnn::utils::RankedTensorTypeFactory::create(argType, dtype);
      std::string constEvalFuncName =
          (funcOp.getName() + "_kv_cache_const_eval_" +
           llvm::Twine(arg.getArgNumber()))
          .str();

      builder.setInsertionPoint(funcOp);
      auto constEvalFuncType =
          builder.getFunctionType({argType}, {castType});
      auto constEvalFuncOp = builder.create<func::FuncOp>(
          funcOp.getLoc(), constEvalFuncName, constEvalFuncType);
      
      ttmlir::utils::setFunctionType(constEvalFuncOp,
                                     ttmlir::utils::FunctionType::ConstEval);
      constEvalFuncOp.setPrivate();

      // Preserve ttcore.kv_cache on the const-eval function's argument so
      // downstream passes treat it consistently.
      constEvalFuncOp.setArgAttr(0, ttcore::g_kvCacheAttrName,
                                  mlir::UnitAttr::get(builder.getContext()));

      auto *entryBlock = constEvalFuncOp.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);
      auto innerCast = builder.create<TypecastOp>(
          funcOp.getLoc(), castType, entryBlock->getArgument(0),
          ttcore::DataTypeAttr::get(builder.getContext(), dtype));
      builder.create<func::ReturnOp>(funcOp.getLoc(), innerCast.getResult());

      infos.push_back({arg, castType, constEvalFuncName});
    }

    // Insert LoadCachedOps at the top of the forward function's entry block,
    // one per kv_cache arg
    auto &entryBlock = funcOp.getBody().front();
    auto insertPt = entryBlock.begin();

    for (auto &info : infos) {
      builder.setInsertionPoint(&entryBlock, insertPt);
      auto calleeAttr =
          mlir::SymbolRefAttr::get(builder.getContext(), info.constEvalFuncName);
      auto loadOp = builder.create<ttcore::LoadCachedOp>(
          funcOp.getLoc(), mlir::TypeRange{info.castType}, calleeAttr,
          mlir::ValueRange{info.arg});

      // Replace all non-return uses of the original arg with the cached result.
      info.arg.replaceUsesWithIf(loadOp.getResult(0), [&](OpOperand &use) {
        return use.getOwner() != loadOp &&
               !mlir::isa<func::ReturnOp>(use.getOwner());
      });

      ++insertPt;
    }
  }

  // Cast the input of cache write ops to the target dtype.
  template <typename OpTy>
  void castCacheOpInput(OpTy op, ttcore::DataType dtype,
                        mlir::OpBuilder &builder) {
    auto cacheType = mlir::cast<RankedTensorType>(op.getCache().getType());
    if (!hasDtype(cacheType.getElementType(), dtype)) {
      return;
    }

    auto input = op.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    if (hasDtype(inputType.getElementType(), dtype)) {
      return;
    }

    auto dtypeAttr = ttcore::DataTypeAttr::get(builder.getContext(), dtype);
    builder.setInsertionPoint(op);
    auto typecastOp = builder.create<TypecastOp>(
        op.getLoc(),
        ttnn::utils::RankedTensorTypeFactory::create(inputType, dtype), input,
        dtypeAttr);
    op.getInputMutable().assign(typecastOp.getResult());
  }

  void processFunc(mlir::func::FuncOp funcOp, ttcore::DataType dtype,
                   mlir::OpBuilder &builder) {
    castKVCacheArgs(funcOp, dtype, builder);

    // fill_cache requires input and cache to have the same dtype — cast input.
    // update_cache accepts bf16/float32 input even with a compressed cache;
    // the kernel handles quantization internally, so no input cast is needed.
    funcOp.walk([&](Operation *op) {
      if (auto fillOp = mlir::dyn_cast<FillCacheOp>(op)) {
        castCacheOpInput(fillOp, dtype, builder);
      } else if (auto pagedFillOp = mlir::dyn_cast<PagedFillCacheOp>(op)) {
        castCacheOpInput(pagedFillOp, dtype, builder);
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttnn
