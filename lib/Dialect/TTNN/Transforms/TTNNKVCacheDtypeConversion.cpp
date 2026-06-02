// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/BFPDtypeParser.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

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

  // Ops that are safe to propagate dtype through on the kv_cache → cache_op
  // path. Extend this list when new transparent ops appear on that path.
  // Currently, the list is empty, but we may want to allow certain TMs in the
  // future.
  static bool isAllowedOnCachePath(Operation *op) { return false; }

  // Walks the linear tensor chain from `value` back through allowed ops,
  // collecting intermediate tensor values into `chain`. Returns the
  // terminating block argument, or nullptr if the chain is invalid
  static BlockArgument collectChainToRoot(Value value,
                                          llvm::SmallVectorImpl<Value> &chain) {
    while (!mlir::isa<BlockArgument>(value)) {
      chain.push_back(value);
      Operation *defOp = value.getDefiningOp();
      if (!isAllowedOnCachePath(defOp)) {
        return nullptr;
      }
      Value next;
      for (Value operand : defOp->getOperands()) {
        if (mlir::isa<RankedTensorType>(operand.getType())) {
          if (next) {
            return nullptr;
          }
          next = operand;
        }
      }
      if (!next) {
        return nullptr;
      }
      value = next;
    }
    return mlir::cast<BlockArgument>(value);
  }

  // Returns true if the chain from `value` back through allowed ops terminates
  // at a kv_cache-labeled block argument. Does not modify any types.
  static bool canPropagateDtype(func::FuncOp funcOp, Value value) {
    llvm::SmallVector<Value> chain;
    BlockArgument blockArg = collectChainToRoot(value, chain);
    return blockArg && funcOp.getArgAttr(blockArg.getArgNumber(),
                                         ttcore::g_kvCacheAttrName);
  }

  static ttcore::LocalShapeAttr
  convertLocalShapeAttr(MLIRContext *ctx, ttcore::LocalShapeAttr attr,
                        ttcore::DataType targetDtype) {
    auto localShapeType = mlir::cast<RankedTensorType>(attr.getLocalShape());
    Type newElementType = ttcore::dataTypeToElementType(ctx, targetDtype);
    auto newLocalShapeType =
        RankedTensorType::get(localShapeType.getShape(), newElementType,
                              localShapeType.getEncoding());
    return ttcore::LocalShapeAttr::get(ctx, newLocalShapeType);
  }

  // Updates all kv_cache block arg types and the function signature to
  // targetDtype. Only call this after all cache op paths have been validated.
  static void convertCacheArgTypes(func::FuncOp funcOp,
                                   ttcore::DataType targetDtype) {
    auto *ctx = funcOp.getContext();
    auto funcType = funcOp.getFunctionType();
    llvm::SmallVector<Type> newArgTypes(funcType.getInputs());
    llvm::SmallVector<Type> newResultTypes(funcType.getResults());

    for (auto &arg : funcOp.getBody().getArguments()) {
      unsigned idx = arg.getArgNumber();
      if (!funcOp.getArgAttr(idx, ttcore::g_kvCacheAttrName)) {
        continue;
      }

      auto oldType = mlir::cast<RankedTensorType>(arg.getType());
      auto newType =
          ttnn::utils::RankedTensorTypeFactory::create(oldType, targetDtype);

      // Keep local_shape metadata aligned with the rewritten argument dtype.
      if (auto localShapeAttr = funcOp.getArgAttrOfType<ttcore::LocalShapeAttr>(
              idx, ttcore::LocalShapeAttr::name)) {
        funcOp.setArgAttr(
            idx, ttcore::LocalShapeAttr::name,
            convertLocalShapeAttr(ctx, localShapeAttr, targetDtype));
      }

      arg.setType(newType);
      newArgTypes[idx] = newType;
    }

    // Update result types to reflect KV cache args passed as return values.
    Block &block = funcOp.getBody().back();
    Operation *terminator = block.getTerminator();
    if (auto returnOp = mlir::dyn_cast<func::ReturnOp>(terminator)) {
      for (auto [i, operand] : llvm::enumerate(returnOp.getOperands())) {
        newResultTypes[i] = operand.getType();

        if (!mlir::isa<BlockArgument>(operand)) {
          continue;
        }

        auto blockArg = mlir::cast<BlockArgument>(operand);
        if (!funcOp.getArgAttr(blockArg.getArgNumber(),
                               ttcore::g_kvCacheAttrName)) {
          continue;
        }

        // Update local_shape of returned KV cache args to match the new dtype.
        if (auto localShapeAttr =
                funcOp.getResultAttrOfType<ttcore::LocalShapeAttr>(
                    i, ttcore::LocalShapeAttr::name)) {
          funcOp.setResultAttr(
              i, ttcore::LocalShapeAttr::name,
              convertLocalShapeAttr(ctx, localShapeAttr, targetDtype));
        }
      }
    }

    funcOp.setFunctionType(
        mlir::FunctionType::get(ctx, newArgTypes, newResultTypes));
  }

  // Updates the element dtype of each value on the chain from `value` back to
  // its kv_cache block argument. Only call this if the chain is valid
  static void propagateDtypeTowardRoot(Value value,
                                       ttcore::DataType targetDtype) {
    llvm::SmallVector<Value> chain;
    collectChainToRoot(value, chain);
    for (Value v : chain) {
      auto tensorType = mlir::cast<RankedTensorType>(v.getType());
      v.setType(ttnn::utils::RankedTensorTypeFactory::create(tensorType,
                                                             targetDtype));
    }
  }

  // Inserts a ttnn.typecast before `op`'s input operand if it is not already
  // the target dtype.
  template <typename OpTy>
  static void insertTypecastIfNeeded(mlir::OpBuilder &builder, OpTy op,
                                     ttcore::DataType dtype) {
    auto input = op.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    mlir::Type elType = inputType.getElementType();

    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == dtype) {
        return;
      }
    } else if (ttcore::elementTypeToDataType(elType) == dtype) {
      return;
    }

    auto newInputType =
        ttnn::utils::RankedTensorTypeFactory::create(inputType, dtype);
    builder.setInsertionPoint(op);
    auto typecastOp =
        builder.create<TypecastOp>(op.getLoc(), newInputType, input);
    op.getInputMutable().assign(typecastOp.getResult());
  }

  void runOnOperation() final {
    if (targetDtype == BFPDtype::None) {
      return;
    }

    ttcore::DataType dtype = bfpDtypeToDataType(targetDtype);

    mlir::OpBuilder builder(&getContext());
    getOperation().walk([&](func::FuncOp funcOp) {
      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return;
      }

      // Check that every cache op's path back to a block arg consists only of
      // arg consists only of isAllowedOnCachePath ops. If any path contains
      // an unsupported op, skip the entire function to avoid partial
      // conversion.
      bool canConvert = true;
      funcOp.walk([&](Operation *op) {
        llvm::TypeSwitch<Operation *>(op)
            .Case<FillCacheOp, UpdateCacheOp, PagedFillCacheOp,
                  PagedUpdateCacheOp>([&](auto cacheOp) {
              if (!canPropagateDtype(funcOp, cacheOp.getCache())) {
                canConvert = false;
              }
            });
      });

      if (!canConvert) {
        return;
      }

      // Update all kv_cache block arg types, the function
      // signature, and propagate the new dtype through the chain.
      convertCacheArgTypes(funcOp, dtype);
      funcOp.walk([&](Operation *op) {
        llvm::TypeSwitch<Operation *>(op)
            .Case<FillCacheOp, UpdateCacheOp, PagedFillCacheOp,
                  PagedUpdateCacheOp>([&](auto cacheOp) {
              propagateDtypeTowardRoot(cacheOp.getCache(), dtype);
            });
      });

      // Insert typecasts on the input (fill value) of each cache op
      // so the written data matches the new cache dtype.
      // PagedUpdateCacheOp is excluded since tt-metal enforces
      // FLOAT32/BFLOAT16 for input dtype and handles dtype mismatch
      // internally.
      funcOp.walk([&](Operation *op) {
        llvm::TypeSwitch<Operation *>(op)
            .Case<FillCacheOp, UpdateCacheOp, PagedFillCacheOp>(
                [&](auto cacheOp) {
                  insertTypecastIfNeeded(builder, cacheOp, dtype);
                });
      });
    });
  }
};

} // namespace

} // namespace mlir::tt::ttnn
