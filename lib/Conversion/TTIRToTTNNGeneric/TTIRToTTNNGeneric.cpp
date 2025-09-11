// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNNGeneric/TTIRToTTNNGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <array>

namespace mlir::tt {
void populateTTIRToTTNNGenericPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttcore::MemorySpace defaultInputMemSpace,
    ttcore::MemorySpace defaultOutputMemSpace,
    const llvm::SmallVector<int64_t> &targetGridShape) {
  // clang-format off
//   patterns.add<
//     // Elementwise.
//     TTIRNamedElementwiseRewriter<ttir::AbsOp,        ttir::TileAbsOp>,
//     TTIRNamedElementwiseRewriter<ttir::AddOp,        ttir::TileAddOp>,
//     TTIRNamedElementwiseRewriter<ttir::CeilOp,       ttir::TileCeilOp>,
//     TTIRNamedElementwiseRewriter<ttir::CosOp,        ttir::TileCosOp>,
//     TTIRNamedElementwiseRewriter<ttir::DivOp,        ttir::TileDivOp>,
//     TTIRNamedElementwiseRewriter<ttir::ExpOp,        ttir::TileExpOp>,
//     TTIRNamedElementwiseRewriter<ttir::FloorOp,      ttir::TileFloorOp>,
//     TTIRNamedElementwiseRewriter<ttir::LogOp,        ttir::TileLogOp>,
//     TTIRNamedElementwiseRewriter<ttir::LogicalNotOp, ttir::TileLogicalNotOp>,
//     TTIRNamedElementwiseRewriter<ttir::MultiplyOp,   ttir::TileMulOp>,
//     TTIRNamedElementwiseRewriter<ttir::MaximumOp,    ttir::TileMaximumOp>,
//     TTIRNamedElementwiseRewriter<ttir::NegOp,        ttir::TileNegativeOp>,
//     TTIRNamedElementwiseRewriter<ttir::PowOp,        ttir::TilePowOp>,
//     TTIRNamedElementwiseRewriter<ttir::ReciprocalOp, ttir::TileRecipOp>,
//     TTIRNamedElementwiseRewriter<ttir::RsqrtOp,      ttir::TileRsqrtOp>,
//     TTIRNamedElementwiseRewriter<ttir::SigmoidOp,    ttir::TileSigmoidOp>,
//     TTIRNamedElementwiseRewriter<ttir::SinOp,        ttir::TileSinOp>,
//     TTIRNamedElementwiseRewriter<ttir::SqrtOp,       ttir::TileSqrtOp>,
//     TTIRNamedElementwiseRewriter<ttir::SubtractOp,   ttir::TileSubOp>,
//     TTIRNamedElementwiseRewriter<ttir::TanOp,        ttir::TileTanOp>,
//     // Reduction.
//     TTIRNamedReductionRewriter<ttir::MaxOp,          ttir::TileReduceMaxOp>,
//     TTIRNamedReductionRewriter<ttir::SumOp,          ttir::TileReduceSumOp>,
//     // Data movement.
//     TTIRNamedElementwiseRewriter<ttir::TypecastOp,   ttir::TileTypecastOp>,
//     // Permute (handles tranpose ops, since they're canonicalized into permutes).
//     TTIRPermuteRewriter
//   >(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, targetGridShape);


  // Matmul.
//   patterns.add<TTIRMatmulRewriter<ttir::TileMatmulOp>>(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, targetGridShape);
  // clang-format on
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
