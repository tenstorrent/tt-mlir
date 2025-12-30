// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTLOADSTOREOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper function to check if an operand is remote (i.e., is a stream op)
static bool isRemote(Value operand) {
  // Remote operands are those that come from stream_layout ops
  return mlir::isa_and_nonnull<StreamLayoutOp>(operand.getDefiningOp());
}

// Helper function to build grid dimension indices from indexing map
static SmallVector<Value> buildGridIndices(OpBuilder &builder, Location loc,
                                           AffineMap indexingMap) {
  SmallVector<Value> indices;
  for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
    AffineExpr expr = indexingMap.getResult(i);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      // Create IterIndexOp for this dimension
      indices.push_back(builder.create<IterIndexOp>(
          loc, static_cast<int64_t>(dimExpr.getPosition())));
    } else if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
      // Constant expression - create constant index
      indices.push_back(
          builder.create<arith::ConstantIndexOp>(loc, constExpr.getValue()));
    }
  }
  return indices;
}

// Helper function to get generic operand and indexing map from CB block
// argument
static std::optional<std::pair<Value, AffineMap>>
getGenericOperandAndIndexingMap(Operation *op, Value cbValue) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return std::nullopt;
  }

  // Skip if generic op is in explicit datamovement form (no indexing maps)
  if (generic.isExplicitDatamovementForm()) {
    return std::nullopt;
  }

  BlockArgument cbArg = dyn_cast<BlockArgument>(cbValue);
  if (!cbArg) {
    return std::nullopt;
  }

  unsigned operandIndex = cbArg.getArgNumber();
  Value genericOperand = generic->getOperand(operandIndex);

  // Verify operand is a memref (post-bufferization)
  if (!isa<MemRefType>(genericOperand.getType())) {
    return std::nullopt;
  }

  // Get the indexing map
  AffineMap indexingMap = generic.getIndexingMap(operandIndex);
  return std::make_pair(genericOperand, indexingMap);
}

class D2MInsertLoadStoreOps
    : public impl::D2MInsertLoadStoreOpsBase<D2MInsertLoadStoreOps> {
public:
  using impl::D2MInsertLoadStoreOpsBase<
      D2MInsertLoadStoreOps>::D2MInsertLoadStoreOpsBase;

  void runOnOperation() final {
    OpBuilder builder(&getContext());

    // Process WaitOp operations
    getOperation()->walk([&](WaitOp waitOp) {
      // Only process memref types (skip tensor types)
      auto memrefType = waitOp.getCbType().getUnderlyingAs<MemRefType>();
      if (!memrefType) {
        return;
      }

      Value cbValue = waitOp.getCb();
      auto genericInfo = getGenericOperandAndIndexingMap(waitOp, cbValue);
      if (!genericInfo) {
        return;
      }

      auto [genericOperand, indexingMap] = *genericInfo;

      builder.setInsertionPoint(waitOp);
      Location loc = waitOp.getLoc();

      if (isRemote(genericOperand)) {
        // Remote case: insert remote_load before wait
        // Get the input from stream_layout (the physical memref with device
        // layout)
        auto streamOp = cast<StreamLayoutOp>(genericOperand.getDefiningOp());
        Value remoteMemref = streamOp.getInput();
        SmallVector<Value> indices =
            buildGridIndices(builder, loc, indexingMap);
        builder.create<RemoteLoadOp>(loc, cbValue, remoteMemref, indices);
      } else {
        // Local case: insert reserve immediately before wait
        builder.create<ReserveOp>(loc, cbValue);
      }
    });

    // Process ReserveOp operations
    getOperation()->walk([&](ReserveOp reserveOp) {
      // Only process memref types (skip tensor types)
      auto memrefType = reserveOp.getCbType().getUnderlyingAs<MemRefType>();
      if (!memrefType) {
        return;
      }

      Value cbValue = reserveOp.getCb();
      auto genericInfo = getGenericOperandAndIndexingMap(reserveOp, cbValue);
      if (!genericInfo) {
        return;
      }

      auto [genericOperand, indexingMap] = *genericInfo;

      builder.setInsertionPointAfter(reserveOp);
      Location loc = reserveOp.getLoc();

      if (isRemote(genericOperand)) {
        // Remote case: insert remote_store after reserve
        // Get the input from stream_layout (the physical memref with device
        // layout)
        auto streamOp = cast<StreamLayoutOp>(genericOperand.getDefiningOp());
        Value remoteMemref = streamOp.getInput();
        SmallVector<Value> indices =
            buildGridIndices(builder, loc, indexingMap);
        builder.create<RemoteStoreOp>(loc, remoteMemref, indices, cbValue);
      } else {
        // Local case: insert wait immediately after reserve
        builder.create<WaitOp>(loc, cbValue);
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
