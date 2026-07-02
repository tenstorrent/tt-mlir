// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFORCEFINALDEALLOCS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Returns the input value whose device buffer op's result aliases, or a null
// Value if op allocates its own buffer (i.e. is not acting as a view).
//
// Today, only view-eligible reshapes alias their input. This helper is the
// single place that knowledge lives so that the rest of the pass is decoupled
// from it. A future ViewOpInterface can replace the body without touching the
// pass logic.
Value getViewSource(Operation *op) {
  if (canReshapeBeView(op)) {
    return op->getOperand(0);
  }
  return Value();
}

// Returns the activation of the given conv op if the conv deallocates it
// itself, or a null value otherwise.
template <typename ConvOpTy>
Value getConvDeallocatedActivation(ConvOpTy conv) {
  auto config = conv.getConv2dConfigAttr();
  Value input = conv.getInput();
  auto inputType = mlir::cast<RankedTensorType>(input.getType());

  // The conv deallocates its activation when the flag is set and the
  // activation is in L1 memory.
  bool deallocatesActivation =
      config && config.getDeallocateActivation() &&
      config.getDeallocateActivation().getValue() &&
      utils::getBufferTypeFromTensor(inputType) == BufferType::L1;

  return deallocatesActivation ? input : Value();
}

// Walks up the chain of view ops until reaching the value that produces the
// underlying buffer, caching the result for every value on the path.
Value canonicalRoot(Value value, llvm::DenseMap<Value, Value> &valueToRoot) {
  llvm::SmallVector<Value, 4> path;
  Value current = value;
  while (true) {
    auto it = valueToRoot.find(current);
    if (it != valueToRoot.end()) {
      current = it->second;
      break;
    }
    Operation *defOp = current.getDefiningOp();
    Value source = defOp ? getViewSource(defOp) : Value();
    if (!source) {
      break;
    }
    path.push_back(current);
    current = source;
  }
  Value root = current;
  for (Value aliased : path) {
    valueToRoot[aliased] = root;
  }
  valueToRoot[root] = root;
  return root;
}

} // namespace

// A `ttnn.deallocate` with the force flag set to false frees the buffer only
// when its input variable is the last one referencing that buffer. This
// becomes a problem when several handles alias one buffer: e.g. a
// view-eligible reshape op returns a tensor that points to its input's device
// buffer. Deallocate ops are inserted in the IR per SSA value by the
// `TTNNDeallocate` pass. However, in the mentioned case, they act as no-ops,
// so the buffer is never freed. This can result in L1 allocation failure.
//
// For each underlying buffer, this pass walks that buffer's deallocate ops from
// bottom to top and sets the force flag to true on the last one in program
// order (the true final use of that buffer), so the buffer is properly freed.
// The remaining deallocates of that buffer are left as force=false no-ops.
// Buffers that escape the function (returned variables) or activations that a
// conv op deallocates itself are left untouched.
class TTNNForceFinalDeallocs
    : public impl::TTNNForceFinalDeallocsBase<TTNNForceFinalDeallocs> {
public:
  using impl::TTNNForceFinalDeallocsBase<
      TTNNForceFinalDeallocs>::TTNNForceFinalDeallocsBase;

  void runOnOperation() final {
    getOperation()->walk([&](func::FuncOp funcOp) { processFunc(funcOp); });
  }

private:
  // Collects the roots that must never be force-freed. These are either
  // variables returned from the function, or conv activations that the conv op
  // deallocates itself.
  llvm::DenseSet<Value>
  collectDoNotForceRoots(func::FuncOp funcOp,
                         llvm::DenseMap<Value, Value> &valueToRoot) {
    llvm::DenseSet<Value> doNotForceRoots;
    funcOp.walk([&](Operation *op) {
      if (auto returnOp = mlir::dyn_cast<func::ReturnOp>(op)) {
        for (Value operand : returnOp.getOperands()) {
          doNotForceRoots.insert(canonicalRoot(operand, valueToRoot));
        }
        return WalkResult::advance();
      }

      Value convActivation;
      if (auto convOp = mlir::dyn_cast<Conv2dOp>(op)) {
        convActivation = getConvDeallocatedActivation(convOp);
      } else if (auto convOp = mlir::dyn_cast<ConvTranspose2dOp>(op)) {
        convActivation = getConvDeallocatedActivation(convOp);
      }
      if (convActivation) {
        doNotForceRoots.insert(canonicalRoot(convActivation, valueToRoot));
      }
      return WalkResult::advance();
    });
    return doNotForceRoots;
  }

  // Forces the last deallocation of each buffer that has more than one
  // (aliasing) deallocations.
  void processFunc(func::FuncOp funcOp) {
    // Resolves each value to the root value identifying its underlying
    // buffer.
    llvm::DenseMap<Value, Value> valueToRoot;

    // Buffers that are used outside the function (returned variables) or
    // deallocated by a conv op (conv op L1 activations) cannot be force-freed.
    llvm::DenseSet<Value> doNotForceRoots =
        collectDoNotForceRoots(funcOp, valueToRoot);

    // Count deallocations per buffer so we only touch buffers that
    // actually have multiple (aliasing) deallocations.
    llvm::SmallVector<DeallocateOp> deallocs;
    llvm::DenseMap<Value, unsigned> deallocCountByRoot;
    funcOp.walk([&](DeallocateOp deallocOp) {
      deallocs.push_back(deallocOp);
      deallocCountByRoot[canonicalRoot(deallocOp.getInput(), valueToRoot)]++;
    });

    // Walk deallocations bottom-to-top. The first deallocate in order for the
    // given buffer is the last in program order.
    llvm::DenseSet<Value> forcedRoots;
    for (auto deallocOp : llvm::reverse(deallocs)) {
      Value root = canonicalRoot(deallocOp.getInput(), valueToRoot);
      if (doNotForceRoots.contains(root)) {
        continue;
      }
      if (deallocCountByRoot.lookup(root) < 2) {
        continue;
      }
      if (!forcedRoots.insert(root).second) {
        continue;
      }
      deallocOp.setForce(true);
    }
  }
};

} // namespace mlir::tt::ttnn
