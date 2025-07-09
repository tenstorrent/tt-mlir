// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"
#include "ttmlir/Dialect/StableHLO/Utils/MeshShardingUtils.h"
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

class ShardyToTTIRManualComputationOpConversionPattern
    : public mlir::OpConversionPattern<mlir::sdy::ManualComputationOp> {
  using mlir::OpConversionPattern<
      mlir::sdy::ManualComputationOp>::OpConversionPattern;

public:
  llvm::LogicalResult
  matchAndRewrite(mlir::sdy::ManualComputationOp srcOp,
                  mlir::sdy::ManualComputationOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
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
