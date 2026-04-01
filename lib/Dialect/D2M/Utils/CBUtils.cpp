// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::d2m {

unsigned getNextAvailablePort(Region &region, PortCounter &portCounters,
                              Operation *genericOp) {
  auto it = portCounters.find(genericOp);
  if (it != portCounters.end()) {
    return it->second;
  }

  // Scan existing get_cb ops to find the max port in use.
  std::optional<unsigned> maxPort;
  if (!region.empty()) {
    region.front().walk([&](GetCBOp getCbOp) {
      unsigned port = static_cast<unsigned>(getCbOp.getPort());
      maxPort = std::max(maxPort.value_or(0), port + 1);
    });
  }
  unsigned next = maxPort.value_or(0);
  portCounters[genericOp] = next;
  return next;
}

Value getOrCreateCB(GenericOp generic, Region &region, unsigned operandIndex,
                    IRRewriter &rewriter, CBCache &cache,
                    PortCounter &portCounters) {
  auto key = std::make_pair(generic.getOperation(), operandIndex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  auto L1Attr = mlir::tt::ttcore::MemorySpaceAttr::get(
      generic.getContext(), mlir::tt::ttcore::MemorySpace::DeviceL1);

  // Derive the CB underlying type for this operand.  Try:
  //   1. In-generic alloc with CBLayoutAttr (set by
  //   Allocate/insertStream)
  //   2. The generic operand's device layout (shard shape)
  MemRefType cbUnderlyingType;
  Value operandAlloc = GenericOp::getOperandAlloc(region, operandIndex);
  if (operandAlloc) {
    if (auto allocOp = operandAlloc.getDefiningOp<memref::AllocOp>()) {
      auto allocType = allocOp.getType();
      if (mlir::isa<ttcore::CBLayoutAttr>(allocType.getLayout())) {
        cbUnderlyingType = allocType;
      }
    }
  }

  if (!cbUnderlyingType) {
    // Derive from the generic operand's device layout.
    auto operandType =
        mlir::cast<ShapedType>(generic->getOperand(operandIndex).getType());
    auto layout = mlir::tt::ttcore::getDeviceLayout(operandType);
    if (!layout) {
      return Value();
    }
    auto shardShape = layout.getShardShape(operandType);
    cbUnderlyingType = MemRefType::get(shardShape, operandType.getElementType(),
                                       nullptr, L1Attr);
  }

  auto cbType = CBType::get(cbUnderlyingType);

  // For IO operands, use operandIndex as the port so the hardware CB port
  // matches the CB descriptor array position in D2MToTTNN (indexed by
  // operand).  The sequential counter is for non-IO CBs (scratch,
  // intermediates) that don't map 1:1 to operands.
  // When fusion with intermediates requires decoupled port
  // assignment, introduce a port remapping in D2MToTTNN instead of relying on
  // port == operand index for IO operands.
  unsigned ioSize = generic.getInputsAndOutputs().size();
  unsigned port;
  if (operandIndex < ioSize) {
    port = operandIndex;
  } else {
    port = getNextAvailablePort(region, portCounters, generic.getOperation());
    portCounters[generic.getOperation()] = port + 1;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&region.front());
  auto getCBOp = rewriter.create<GetCBOp>(
      generic.getLoc(), cbType, port,
      rewriter.getI64IntegerAttr(static_cast<int64_t>(operandIndex)),
      ResolutionStageAttr::get(rewriter.getContext(),
                               ResolutionStage::Compile));

  Value result = getCBOp.getResult();
  cache[key] = result;
  return result;
}

Value findAssociatedCB(Operation *op, Value memrefOperand, IRRewriter &rewriter,
                       CBCache &cache, PortCounter &portCounters) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return Value();
  }

  // Find which operand index this memref corresponds to.
  unsigned operandIndex = UINT_MAX;
  for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
    if (generic->getOperand(i) == memrefOperand) {
      operandIndex = i;
      break;
    }
  }
  if (operandIndex == UINT_MAX) {
    return Value();
  }

  // Find the generic op's thread region that contains this operation.
  Region *genericRegion = nullptr;
  if (generic.getNumRegions() == 1) {
    genericRegion = &generic.getRegion(0);
  } else {
    genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  }
  if (!genericRegion || genericRegion->empty()) {
    return Value();
  }

  return getOrCreateCB(generic, *genericRegion, operandIndex, rewriter, cache,
                       portCounters);
}

} // namespace mlir::tt::d2m
