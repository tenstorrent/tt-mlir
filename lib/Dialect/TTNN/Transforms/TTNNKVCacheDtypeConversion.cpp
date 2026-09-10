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

#include "llvm/ADT/DenseSet.h"
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

  // Ops that are transparent on the read path — their output dtype mirrors
  // their cache-path input dtype, so bfp_bf8 propagates through without
  // any downcast typecast.
  //
  // RMSNormOp: output dtype equals input dtype; gamma/beta stay bf16 which
  // tt-metal layernorm accepts alongside a bfp_bf8 activation.
  //
  // RepeatInterleaveOp: feeds K/V into the GLM-4 attention matmul; bfp_bf8
  // propagates through so the matmul receives bfp_bf8 K natively.
  //
  // SliceStaticOp, PermuteOp, ReshapeOp, RepeatOp: pure shape TMs.
  //
  // MatmulOp is handled separately in walkReadPath: transparent when all
  // non-cache operands are already at targetDtype (MLA up-projection), stop
  // otherwise (attention matmul with bf16 Q — consumes bfp_bf8 K natively,
  // emits bf16).
  //
  // ConcatOp is handled by the fixpoint in collectReadPathInfo: promoted
  // from boundary to transparent once all tensor inputs are in
  // propagatedValues (MLA: both c_kv K and kv_pe are bfp_bf8).
  // Mixed-input concats stay as boundaries (no typecast is inserted for
  // boundary ops; they receive bfp_bf8 directly).
  static bool isTransparentOnReadPath(Operation *op) {
    return mlir::isa<PermuteOp, ReshapeOp, RepeatOp, RepeatInterleaveOp,
                     RMSNormOp, SliceStaticOp>(op);
  }

  // ConcatOp is promoted from boundary to transparent in collectReadPathInfo
  // when all its tensor inputs are in propagatedValues.
  static bool isPromotableAtBoundary(Operation *op) {
    return mlir::isa<ConcatOp>(op);
  }

  // Returns the DataType encoded in a ranked tensor value's element type.
  static ttcore::DataType getValueDtype(Value v) {
    mlir::Type elType =
        mlir::cast<RankedTensorType>(v.getType()).getElementType();
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      return tileType.getDataType();
    }
    return ttcore::elementTypeToDataType(elType);
  }

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

  // Walks the read-path chain forward from `v`, collecting transparent op
  // results into propagatedValues and non-transparent consumers into
  // boundaryOps.
  static void walkReadPath(Value v, llvm::DenseSet<Value> &propagatedValues,
                           llvm::DenseSet<Operation *> &boundaryOps,
                           ttcore::DataType targetDtype) {
    for (OpOperand &use : v.getUses()) {
      Operation *op = use.getOwner();

      // Cache write ops are handled separately; skip here to avoid
      // double-processing.
      // SDPA variants consume bfp_bf8 K/V cache operands natively and produce
      // bf16 output; no downstream type updates are needed.
      // TypecastOp is an explicit dtype boundary — stop so we don't corrupt its
      // output type.
      // ReturnOp is not a compute op; skip it.
      if (mlir::isa<FillCacheOp, UpdateCacheOp, PagedFillCacheOp,
                    PagedUpdateCacheOp, ScaledDotProductAttentionOp,
                    ScaledDotProductAttentionDecodeOp,
                    PagedScaledDotProductAttentionDecodeOp, TypecastOp,
                    func::ReturnOp>(op) ||
          isAllowedOnCachePath(op)) {
        continue;
      }

      // MatmulOp: transparent when all non-cache operands are already at
      // targetDtype (MLA up-projection: c_kv_norm × W_UK_bfp8 → bfp_bf8
      // result). When any non-cache operand is at a different dtype (e.g.,
      // bf16 Q in the attention matmul), stop: the matmul consumes bfp_bf8
      // K natively via getOutputDtype and emits bf16.
      if (mlir::isa<MatmulOp>(op)) {
        bool allNonCacheAtDtype = true;
        for (Value operand : op->getOperands()) {
          if (!mlir::isa<RankedTensorType>(operand.getType())) {
            continue;
          }
          if (propagatedValues.count(operand)) {
            continue;
          }
          if (getValueDtype(operand) != targetDtype) {
            allNonCacheAtDtype = false;
            break;
          }
        }
        if (!allNonCacheAtDtype) {
          continue; // stop
        }
        // Transparent: propagate result types and recurse, then skip the
        // isTransparentOnReadPath check below which would wrongly classify
        // MatmulOp as a boundary op.
        for (Value result : op->getResults()) {
          if (!mlir::isa<RankedTensorType>(result.getType()) ||
              propagatedValues.count(result)) {
            continue;
          }
          propagatedValues.insert(result);
          walkReadPath(result, propagatedValues, boundaryOps, targetDtype);
        }
        continue;
      }

      if (!isTransparentOnReadPath(op)) {
        boundaryOps.insert(op);
        continue;
      }

      for (Value result : op->getResults()) {
        if (!mlir::isa<RankedTensorType>(result.getType()) ||
            propagatedValues.count(result)) {
          continue;
        }
        propagatedValues.insert(result);
        walkReadPath(result, propagatedValues, boundaryOps, targetDtype);
      }
    }
  }

  // Collects read-path values and their non-transparent consumers without
  // modifying any types.
  static void collectReadPathInfo(func::FuncOp funcOp,
                                  llvm::DenseSet<Value> &propagatedValues,
                                  llvm::DenseSet<Operation *> &boundaryOps,
                                  ttcore::DataType targetDtype) {
    for (BlockArgument arg : funcOp.getBody().getArguments()) {
      if (!funcOp.getArgAttr(arg.getArgNumber(), ttcore::g_kvCacheAttrName)) {
        continue;
      }
      propagatedValues.insert(arg);
      walkReadPath(arg, propagatedValues, boundaryOps, targetDtype);
    }

    // Promote ConcatOps from boundary to transparent when all their tensor
    // inputs are in propagatedValues (MLA: c_kv K and kv_pe both bfp_bf8).
    // Repeat until stable to handle chains of such ops.
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<Operation *> toPromote;
      for (Operation *op : boundaryOps) {
        if (!isPromotableAtBoundary(op)) {
          continue;
        }
        bool allPropagated =
            llvm::all_of(op->getOpOperands(), [&](OpOperand &oper) {
              Value v = oper.get();
              return !mlir::isa<RankedTensorType>(v.getType()) ||
                     propagatedValues.count(v);
            });
        if (allPropagated) {
          toPromote.push_back(op);
        }
      }
      for (Operation *op : toPromote) {
        boundaryOps.erase(op);
        for (Value result : op->getResults()) {
          if (!mlir::isa<RankedTensorType>(result.getType())) {
            continue;
          }
          propagatedValues.insert(result);
          walkReadPath(result, propagatedValues, boundaryOps, targetDtype);
        }
        changed = true;
      }
    }
  }

  // Updates result types of transparent read-path ops in propagatedValues to
  // targetDtype. Covers permute/reshape/repeat/rms_norm/slice_static/matmul
  // (up-proj) and promoted concat. Block args are skipped — already retyped by
  // convertCacheArgTypes. Boundary op results are not in propagatedValues and
  // are intentionally left at BFloat16.
  static void
  propagateDtypeForward(const llvm::DenseSet<Value> &propagatedValues,
                        ttcore::DataType targetDtype) {
    for (Value v : propagatedValues) {
      if (mlir::isa<BlockArgument>(v)) {
        continue;
      }
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
      // isAllowedOnCachePath ops. If any path contains an unsupported op, skip
      // the entire function to avoid partial conversion.
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

      llvm::DenseSet<Value> propagatedValues;
      llvm::DenseSet<Operation *> boundaryOps;
      collectReadPathInfo(funcOp, propagatedValues, boundaryOps, dtype);

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

      // Forward-propagate the new dtype through all transparent read-path ops.
      // No downcast typecasts are inserted: bfp_bf8 values reach their
      // consumers (attention matmuls, SDPA, etc.) directly.
      propagateDtypeForward(propagatedValues, dtype);

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
