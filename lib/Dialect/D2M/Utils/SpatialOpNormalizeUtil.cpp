// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/SpatialOpNormalizeUtil.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {
namespace detail {

// Operand used by a nested generic, traced through view_layout chains that stay
// inside the spatial op until a value defined outside the spatial regions.
static Value resolveToRegionBorderValue(Value operand,
                                        d2m::SpatialOp spatialOp) {
  auto inSpatialRegion = [&](Value val) {
    Operation *def = val.getDefiningOp();
    if (!def) {
      return false;
    }
    Region *parent = def->getBlock()->getParent();
    return llvm::any_of(
        spatialOp->getRegions(),
        [parent](Region &spatialRegion) { return &spatialRegion == parent; });
  };
  Value current = operand;
  while (inSpatialRegion(current)) {
    if (auto viewOp = current.getDefiningOp<d2m::ViewLayoutOp>()) {
      current = viewOp.getInput();
    } else {
      break;
    }
  }
  return current;
}

// Hoist every op except d2m.generic out of each spatial region: prefix ops
// move before the spatial (forward moveBefore order), suffix ops (after generic
// until an optional d2m.spatial_yield or block end) move after the spatial
// with generic-result uses rewired to spatial results. Any existing
// d2m.spatial_yield ops in the region are erased, then a fresh yield of the
// generic's results is created at the end of the region block.
static void hoistNonGenericOpsAroundSpatial(d2m::SpatialOp spatialOp) {
  mlir::OpBuilder builder(spatialOp.getContext());
  Operation *suffixInsertTail = spatialOp.getOperation();
  unsigned cumulativeGenericResults = 0;

  for (Region &spatialRegion : spatialOp->getRegions()) {
    TT_assertv(!spatialRegion.empty(),
               "each spatial region must not be empty.");
    Block &regionBlock = spatialRegion.front();
    auto genericOps = llvm::to_vector(regionBlock.getOps<d2m::GenericOp>());
    TT_assertv(genericOps.size() == 1u,
               "each spatial region must contain exactly one d2m.generic op.");
    d2m::GenericOp genericOp = genericOps.front();

    llvm::SmallVector<Operation *> beforeOps;
    llvm::SmallVector<Operation *> afterOps;
    bool seenGeneric = false;
    for (Operation &regionBodyOp : regionBlock) {
      if (&regionBodyOp == genericOp.getOperation()) {
        seenGeneric = true;
        continue;
      }
      if (!seenGeneric) {
        beforeOps.push_back(&regionBodyOp);
      } else {
        if (mlir::isa<d2m::SpatialYieldOp>(&regionBodyOp)) {
          break;
        }
        afterOps.push_back(&regionBodyOp);
      }
    }

    for (d2m::SpatialYieldOp yieldOp :
         llvm::to_vector(regionBlock.getOps<d2m::SpatialYieldOp>())) {
      yieldOp.erase();
    }

    for (Operation *prefixOp : beforeOps) {
      prefixOp->moveBefore(spatialOp);
    }

    if (!afterOps.empty()) {
      const unsigned resultBase = cumulativeGenericResults;
      const unsigned numGenResults = genericOp->getNumResults();
      const unsigned numSpatialResults = spatialOp->getNumResults();
      TT_assertv(((numGenResults == 0u) ||
                  (resultBase + numGenResults <= numSpatialResults)),
                 "spatial op result count must cover generic results for "
                 "suffix hoist remapping");

      for (unsigned resultIndex = 0; resultIndex < numGenResults;
           ++resultIndex) {
        Value genericResult = genericOp.getResult(resultIndex);
        Value spatialResult = spatialOp.getResult(resultBase + resultIndex);
        genericResult.replaceUsesWithIf(spatialResult, [&](OpOperand &operand) {
          return llvm::is_contained(afterOps, operand.getOwner());
        });
      }

      for (Operation *suffixOp : afterOps) {
        for (Value operand : suffixOp->getOperands()) {
          TT_assertv(!llvm::is_contained(genericOp.getResults(), operand),
                     "suffix op operand should use spatial op results after "
                     "remapping");
        }
      }

      for (Operation *suffixOp : afterOps) {
        suffixOp->moveAfter(suffixInsertTail);
        suffixInsertTail = suffixOp;
      }
    }

    cumulativeGenericResults += genericOp->getNumResults();

    builder.setInsertionPointToEnd(&regionBlock);
    builder.create<d2m::SpatialYieldOp>(genericOp.getLoc(),
                                        genericOp.getResults());
  }
}

// Reassign d2m.spatial ins/outs from generics' operands; when result count
// matches outs, update each spatial result's type from the corresponding out
// value only when that out is a ranked tensor (spatial results are tensors;
// outs may be memref).
static void rebuildSpatialOpInsOutsAndResultTypes(d2m::SpatialOp spatialOp) {
  llvm::SmallVector<Value> inputs;
  llvm::SmallVector<Value> outputs;
  for (Region &region : spatialOp->getRegions()) {
    if (region.empty()) {
      continue;
    }
    for (d2m::GenericOp genericOp : region.front().getOps<d2m::GenericOp>()) {
      for (Value input : genericOp.getInputs()) {
        inputs.push_back(resolveToRegionBorderValue(input, spatialOp));
      }
      for (Value output : genericOp.getOutputs()) {
        outputs.push_back(resolveToRegionBorderValue(output, spatialOp));
      }
    }
  }
  spatialOp.getInputsMutable().assign(inputs);
  spatialOp.getOutputsMutable().assign(outputs);
  if (spatialOp->getNumResults() == outputs.size()) {
    for (auto [result, outVal] : llvm::zip(spatialOp->getResults(), outputs)) {
      if (mlir::isa<RankedTensorType>(outVal.getType())) {
        result.setType(outVal.getType());
      }
    }
  }
}

void normalizeSingleSpatialOp(d2m::SpatialOp spatialOp) {
  hoistNonGenericOpsAroundSpatial(spatialOp);
  rebuildSpatialOpInsOutsAndResultTypes(spatialOp);
}

} // namespace detail

void normalizeSpatialOpsInModule(ModuleOp module) {
  SmallVector<d2m::SpatialOp> spatials;
  module.walk([&](d2m::SpatialOp spatialOp) { spatials.push_back(spatialOp); });

  for (d2m::SpatialOp spatialOp : spatials) {
    detail::normalizeSingleSpatialOp(spatialOp);
  }
}

void normalizeSpatialOpContainingGeneric(GenericOp genericOp) {
  for (Region *r = genericOp->getParentRegion(); r;) {
    Operation *parent = r->getParentOp();
    if (!parent) {
      break;
    }
    if (auto spatial = mlir::dyn_cast<d2m::SpatialOp>(parent)) {
      detail::normalizeSingleSpatialOp(spatial);
      return;
    }
    r = parent->getParentRegion();
  }
}

} // namespace mlir::tt::d2m
