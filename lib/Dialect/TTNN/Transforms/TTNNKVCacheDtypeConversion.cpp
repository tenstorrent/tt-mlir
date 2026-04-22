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

  static void changeCacheArgTypes(func::FuncOp funcOp,
                                  ttcore::DataType targetDtype) {
    if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
      return;
    }

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
    auto typecastOp = builder.create<TypecastOp>(
        op.getLoc(), newInputType, input,
        ttcore::DataTypeAttr::get(builder.getContext(), dtype));
    op.getInputMutable().assign(typecastOp.getResult());
  }

  void runOnOperation() final {
    if (targetDtype == BFPDtype::None) {
      return;
    }

    ttcore::DataType dtype = bfpDtypeToDataType(targetDtype);

    // Change kv_cache argument types to the target dtype and update the
    // function signature. Insert typecast operations on the input operands of
    // all cache ops so written data matches the new cache dtype.
    // PagedUpdateCacheOp is excluded: tt-metal enforces FLOAT32/BFLOAT16 for
    // that op and handles any dtype mismatch internally.
    mlir::OpBuilder builder(&getContext());
    getOperation().walk([&](func::FuncOp funcOp) {
      changeCacheArgTypes(funcOp, dtype);

      if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        return;
      }
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
