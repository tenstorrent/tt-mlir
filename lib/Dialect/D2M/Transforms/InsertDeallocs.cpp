//===- InsertDeallocs.cpp - Insert dealloc operations --------------------===//
//
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

#include <algorithm>
#include <vector>

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_D2MINSERTDEALLOCS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Liveness information for each operation.
struct LivenessInfo {
  llvm::DenseSet<Value> liveIn;
  llvm::DenseSet<Value> liveOut;
  llvm::DenseSet<Value> def;
  llvm::DenseSet<Value> use;
};

// Maps each value to the set of values it aliases.
using AliasMap = llvm::DenseMap<Value, llvm::SmallSetVector<Value, 4>>;

class D2MInsertDeallocsPass
    : public impl::D2MInsertDeallocsBase<D2MInsertDeallocsPass> {
public:
  using impl::D2MInsertDeallocsBase<
      D2MInsertDeallocsPass>::D2MInsertDeallocsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Process each function
    module.walk([&](func::FuncOp func) {
      llvm::errs() << "D2MInsertDeallocs: processing function " << func.getName() << "\n";
      processFunction(func);
    });
  }

private:
  void processFunction(func::FuncOp func) {
    // Step 1: Build alias map.
    AliasMap aliasMap = buildAliasMap(func);

    // Step 2: Collect operations in execution order.
    llvm::SmallVector<Operation *> ops;
    func.walk([&](Operation *op) {
      // Skip nested regions for now - handle at block level.
      if (op->getParentOp() == func.getOperation()) {
        ops.push_back(op);
      }
    });

    // For simplicity, work with the main block.
    if (func.getBody().getBlocks().size() != 1) {
      func.emitWarning("D2MInsertDeallocs: function has multiple blocks, "
                       "only processing first block");
    }

    Block &block = func.front();
    ops.clear();
    for (Operation &op : block.getOperations()) {
      ops.push_back(&op);
    }

    // Step 3: Run backward liveness analysis
    llvm::DenseMap<Operation *, LivenessInfo> livenessInfo =
        computeLiveness(ops, aliasMap);

    // Step 4: Insert deallocs
    insertDeallocs(ops, livenessInfo, aliasMap, func);
  }

  // Build alias map for view and stream operations.
  AliasMap buildAliasMap(func::FuncOp func) {
    AliasMap aliasMap;

    func.walk([&](Operation *op) {
      // Handle view_layout: result aliases input.
      if (auto viewOp = llvm::dyn_cast<d2m::ViewLayoutOp>(op)) {
        Value input = viewOp.getInput();
        Value result = viewOp.getResult();
        aliasMap[result].insert(input);
        aliasMap[input].insert(result);
      }

      // Handle stream_layout: result aliases storage.
      // The operation copies data from input to storage, but they don't alias
      // each other - only result and storage share the same memory.
      if (auto streamOp = llvm::dyn_cast<d2m::StreamLayoutOp>(op)) {
        Value storage = streamOp.getStorage();
        Value result = streamOp.getResult();

        // Result aliases storage (they point to the same memory).
        aliasMap[result].insert(storage);
        aliasMap[storage].insert(result);

        // Note: input is used by the stream_layout operation itself,
        // so standard liveness analysis will handle its lifetime correctly.
        // We don't add it to the alias map because it doesn't alias storage.
      }
    });

    return aliasMap;
  }

  // Compute DEF and USE sets for an operation.
  void computeDefUse(Operation *op, LivenessInfo &info, const AliasMap &aliasMap) {
    // DEF: values defined by this operation.
    for (Value result : op->getResults()) {
      if (llvm::isa<mlir::MemRefType>(result.getType())) {
        info.def.insert(result);
      }
    }

    // USE: operands used by this operation.
    for (Value operand : op->getOperands()) {
      if (llvm::isa<mlir::MemRefType>(operand.getType())) {
        info.use.insert(operand);
      }
    }

    // Special handling for stores/mutations: they "kill" the target and aliases.
    // For now, we don't have explicit store ops, but stream_layout modifies storage.
    if (auto streamOp = llvm::dyn_cast<d2m::StreamLayoutOp>(op)) {
      Value storage = streamOp.getStorage();
      // Storage is both used and defined (modified).
      info.use.insert(storage);
      info.def.insert(storage);
    }
  }

  // Standard backward liveness analysis.
  llvm::DenseMap<Operation *, LivenessInfo>
  computeLiveness(llvm::ArrayRef<Operation *> ops, const AliasMap &aliasMap) {
    llvm::DenseMap<Operation *, LivenessInfo> info;

    if (ops.empty()) {
      return info;
    }

    // Step 1: Compute DEF and USE sets for each operation.
    for (Operation *op : ops) {
      computeDefUse(op, info[op], aliasMap);
    }

    // Step 2: Backward dataflow analysis:
    // live_in[i] = use[i] ∪ (live_out[i] - def[i])
    // live_out[i-1] = live_in[i]

    // Start from the end with empty live_out.
    info[ops.back()].liveOut.clear();

    // Backward pass
    for (int i = ops.size() - 1; i >= 0; --i) {
      Operation *op = ops[i];
      LivenessInfo &opInfo = info[op];

      // live_in = use ∪ (live_out - def)
      opInfo.liveIn = opInfo.use;
      for (Value v : opInfo.liveOut) {
        if (!opInfo.def.contains(v)) {
          opInfo.liveIn.insert(v);
        }
      }

      // Propagate to previous operation.
      if (i > 0) {
        Operation *prevOp = ops[i - 1];
        info[prevOp].liveOut = opInfo.liveIn;
      }
    }

    return info;
  }

  // Find the last operation where a value is live.
  Operation *findLastLive(Value val,
                          llvm::ArrayRef<Operation *> ops,
                          const llvm::DenseMap<Operation *, LivenessInfo> &livenessInfo,
                          const AliasMap &aliasMap) {
    Operation *lastLive = nullptr;

    // Collect all aliases of this value.
    llvm::SmallSetVector<Value, 4> allValues;
    allValues.insert(val);
    if (aliasMap.count(val)) {
      for (Value alias : aliasMap.lookup(val)) {
        allValues.insert(alias);
      }
    }

    // Find the last operation where any of these values is live.
    for (Operation *op : ops) {
      const LivenessInfo &info = livenessInfo.lookup(op);
      for (Value v : allValues) {
        if (info.liveIn.contains(v) || info.liveOut.contains(v)) {
          lastLive = op;
          break;
        }
      }
    }

    return lastLive;
  }

  // Check if an operation's region allows deallocs.
  bool isValidDeallocRegion(Operation *op) {
    // Never insert deallocs inside d2m.generic.
    if (llvm::isa<d2m::GenericOp>(op)) {
      return false;
    }

    // Functions are valid.
    if (llvm::isa<func::FuncOp>(op)) {
      return true;
    }

    // Default: allow
    return true;
  }

  // Find a valid insertion point by walking up the parent chain.
  Operation *findValidInsertionPoint(Operation *lastLive) {
    Operation *candidate = lastLive;

    while (candidate) {
      Operation *parent = candidate->getParentOp();
      if (!parent || isValidDeallocRegion(parent)) {
        return candidate;
      }
      candidate = parent;
    }

    return nullptr;
  }

  // Check if a dealloc already exists for this buffer or any of its aliases.
  bool hasExistingDealloc(Value buffer, const AliasMap &aliasMap,
                          llvm::ArrayRef<Operation *> ops) {
    // Collect all values to check (buffer + all its aliases).
    llvm::SmallSetVector<Value, 4> allValues;
    allValues.insert(buffer);
    if (aliasMap.count(buffer)) {
      for (Value alias : aliasMap.lookup(buffer)) {
        allValues.insert(alias);
      }
    }

    // Check if any of these values has a dealloc already.
    for (Operation *op : ops) {
      if (auto deallocOp = llvm::dyn_cast<memref::DeallocOp>(op)) {
        Value deallocated = deallocOp.getMemref();
        if (allValues.contains(deallocated)) {
          return true;
        }
      }
    }
    return false;
  }

  // Insert deallocs for allocated buffers.
  void insertDeallocs(llvm::ArrayRef<Operation *> ops,
                      const llvm::DenseMap<Operation *, LivenessInfo> &livenessInfo,
                      const AliasMap &aliasMap,
                      func::FuncOp func) {
    OpBuilder builder(func.getContext());

    // Collect all allocated buffers.
    llvm::SmallVector<Value> allocatedBuffers;
    for (Operation *op : ops) {
      if (auto allocOp = llvm::dyn_cast<memref::AllocOp>(op)) {
        allocatedBuffers.push_back(allocOp.getResult());
      }
    }

    llvm::errs() << "D2MInsertDeallocs: found " << allocatedBuffers.size() << " allocated buffers\n";

    // For each allocated buffer, find where it dies and insert dealloc.
    for (Value buffer : allocatedBuffers) {
      Operation *defOp = buffer.getDefiningOp();

      llvm::errs() << "D2MInsertDeallocs: processing buffer defined by: ";
      defOp->print(llvm::errs());
      llvm::errs() << "\n";

      // Skip if a dealloc already exists for this buffer or its aliases.
      if (hasExistingDealloc(buffer, aliasMap, ops)) {
        llvm::errs() << "  -> already has a dealloc, skipping\n";
        continue;
      }

      // Skip if we've already marked this buffer as deallocated.
      if (defOp->hasAttr("d2m.dealloc_inserted")) {
        llvm::errs() << "  -> already has d2m.dealloc_inserted attribute, skipping\n";
        continue;
      }

      // Skip if any alias has already been deallocated.
      bool alreadyProcessed = false;
      if (aliasMap.count(buffer)) {
        llvm::errs() << "  -> has " << aliasMap.lookup(buffer).size() << " aliases\n";
        for (Value alias : aliasMap.lookup(buffer)) {
          if (Operation *aliasDefOp = alias.getDefiningOp()) {
            if (aliasDefOp->hasAttr("d2m.dealloc_inserted")) {
              llvm::errs() << "  -> alias already processed, skipping\n";
              alreadyProcessed = true;
              break;
            }
          }
        }
      }
      if (alreadyProcessed) {
        continue;
      }

      Operation *lastLive = findLastLive(buffer, ops, livenessInfo, aliasMap);
      Operation *insertPoint = nullptr;

      if (!lastLive) {
        // Buffer is never used - insert dealloc right after allocation.
        builder.setInsertionPointAfter(defOp);
        insertPoint = defOp;
      } else {
        // Find valid insertion point (never inside d2m.generic).
        insertPoint = findValidInsertionPoint(lastLive);
        if (!insertPoint) {
          buffer.getDefiningOp()->emitWarning(
              "Could not find valid insertion point for dealloc");
          continue;
        }

        // If the insertion point is a terminator, insert before it.
        // Otherwise, insert after it.
        if (insertPoint->hasTrait<OpTrait::IsTerminator>()) {
          builder.setInsertionPoint(insertPoint);
        } else {
          builder.setInsertionPointAfter(insertPoint);
        }
      }

      // Insert the dealloc.
      llvm::errs() << "  -> inserting dealloc after: ";
      insertPoint->print(llvm::errs());
      llvm::errs() << "\n";
      auto deallocOp = builder.create<memref::DeallocOp>(buffer.getLoc(), buffer);
      llvm::errs() << "  -> created dealloc operation: ";
      deallocOp->print(llvm::errs());
      llvm::errs() << "\n";

      // Mark this buffer and all its aliases as having dealloc inserted.
      defOp->setAttr("d2m.dealloc_inserted", builder.getUnitAttr());
      if (aliasMap.count(buffer)) {
        for (Value alias : aliasMap.lookup(buffer)) {
          if (Operation *aliasDefOp = alias.getDefiningOp()) {
            aliasDefOp->setAttr("d2m.dealloc_inserted", builder.getUnitAttr());
          }
        }
      }
    }

    // Clean up temporary attributes.
    for (Operation *op : ops) {
      op->removeAttr("d2m.dealloc_inserted");
    }
  }
};

} // namespace

} // namespace mlir::tt::d2m
