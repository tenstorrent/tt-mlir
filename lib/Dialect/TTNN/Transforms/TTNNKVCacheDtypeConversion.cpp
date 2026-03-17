#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/WeightDtypeParser.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNKVCACHEDTYPECONVERSION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Template pattern that inserts a typecast on the input operand of kv cache
// ops where it requires input and cache dtypes to match (e.g. fill_cache).
template <typename OpTy>
class KVCacheDtypePattern : public mlir::OpRewritePattern<OpTy> {
public:
  KVCacheDtypePattern(mlir::MLIRContext *ctx, ttcore::DataType targetDtype)
      : mlir::OpRewritePattern<OpTy>(ctx), targetDtype(targetDtype) {}

  mlir::LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    mlir::Type elType = inputType.getElementType();

    // Skip if input is already the target dtype.
    if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
      if (tileType.getDataType() == targetDtype) {
        return mlir::failure();
      }
    } else if (ttcore::elementTypeToDataType(elType) == targetDtype) {
      return mlir::failure();
    }

    auto newInputType =
        ttnn::utils::RankedTensorTypeFactory::create(inputType, targetDtype);
    auto typecastOp = rewriter.create<TypecastOp>(
        op.getLoc(), newInputType, input,
        ttcore::DataTypeAttr::get(rewriter.getContext(), targetDtype));

    rewriter.modifyOpInPlace(op, [&]() {
      op.getInputMutable().assign(typecastOp.getResult());
    });

    return mlir::success();
  }

private:
  ttcore::DataType targetDtype;
};

class TTNNKVCacheDtypeConversionPass
    : public impl::TTNNKVCacheDtypeConversionBase<
          TTNNKVCacheDtypeConversionPass> {
public:
  using impl::TTNNKVCacheDtypeConversionBase<
      TTNNKVCacheDtypeConversionPass>::TTNNKVCacheDtypeConversionBase;

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

  static void changeCacheArgTypes(func::FuncOp funcOp, ttcore::DataType targetDtype) {
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

      arg.setType(newType);
      newArgTypes[idx] = newType;
    }

    // Update result types to reflect KV cache args passed as return values.
    Block &block = funcOp.getBody().back();
    Operation *terminator = block.getTerminator();
    if (auto returnOp = mlir::dyn_cast<func::ReturnOp>(terminator)) {
      for (auto [i, operand] : llvm::enumerate(returnOp.getOperands())) {
        newResultTypes[i] = operand.getType();
      }
    }

    funcOp.setFunctionType(
        mlir::FunctionType::get(ctx, newArgTypes, newResultTypes));
  }

  void runOnOperation() final {
    if (targetDtype == WeightDtype::None) {
      return;
    }

    ttcore::DataType dtype = weightDtypeToDataType(targetDtype);

    // Change kv_cache argument types to the target dtype and update the
    // function signature.
    getOperation().walk([&](func::FuncOp funcOp) {
      changeCacheArgTypes(funcOp, dtype);
    });

    // Insert typecast operations on the input operands of fill_cache op.
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<KVCacheDtypePattern<FillCacheOp>>(&getContext(), dtype);

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn