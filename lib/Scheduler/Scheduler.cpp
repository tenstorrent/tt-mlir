// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Scheduler/Scheduler.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::tt::scheduler {

// TTNN op is scheduleable if it is not an EmptyOp and has at least one result.
static bool isTTNNScheduleableOp(mlir::Operation *op) {
  return isa<ttnn::TTNNDialect>(op->getDialect()) && op->getNumResults() > 0 &&
         !llvm::isa<ttnn::EmptyOp>(op) && !llvm::isa<ttnn::GetDeviceOp>(op);
}

static bool isTTIRSchedulableOp(mlir::Operation *op) {
  return isa<ttir::TTIRDialect>(op->getDialect()) &&
         !llvm::isa<ttir::EmptyOp>(op);
}

bool Scheduler::isTTSchedulableOp(mlir::Operation *op) {
  return isTTNNScheduleableOp(op) || isTTIRSchedulableOp(op);
}

// Init the dependencies map of all ops which are TTIR ops
Scheduler::Scheduler(func::FuncOp *func) {
  for (auto &op : func->getOps()) {
    if (isTTSchedulableOp(&op)) {
      dependencies[&op] = {};
      opsWithinFuncOp.push_back(&op);
    }
  }

  for (auto &op : func->getOps()) {
    // Skip non TTIR operations
    // Skip operations which do not implement DestinationStyleOpInterface
    if (!isTTSchedulableOp(&op)) {
      continue;
    }

    OpResult result = op.getResult(0);

    for (mlir::Operation *use : result.getUsers()) {
      // Skip non TTIR operations
      // Skip operations which set the result
      if (isTTSchedulableOp(use) && use->getResult(0) != result) {
        dependencies[use].push_back(&op);
      }
    }
  }
}

Scheduler::Scheduler(const Scheduler &scheduler)
    : scheduledOpsMap(scheduler.scheduledOpsMap), schedule(scheduler.schedule),
      opsWithinFuncOp(scheduler.opsWithinFuncOp),
      dependencies(scheduler.dependencies) {}

llvm::SmallVector<mlir::Operation *> Scheduler::getSchedulableOps() {
  llvm::SmallVector<mlir::Operation *> schedulableOps;
  for (auto &op : opsWithinFuncOp) {
    if (!scheduledOpsMap.contains(op) && canSchedule(op)) {
      schedulableOps.push_back(op);
    }
  }

  if (schedulableOps.size() <= 1) {
    return schedulableOps;
  }

  // We will sort schedulable ops by prioritizing ops whose successors are still
  // blocked after scheduling it. This is a heuristic that lets us create longer
  // chains of ops that contain join nodes in fork-join structure.
  // This is not general solution and we want to change it in the future.
  // TODO(rpavlovicTT) https://github.com/tenstorrent/tt-mlir/issues/3744
  auto hasBlockedSuccessor = [&](mlir::Operation *op) -> bool {
    // A successor is any op for which 'op' is a dependency.
    for (Operation *succ : opsWithinFuncOp) {
      if (succ == op) {
        continue;
      }
      auto it = dependencies.find(succ);
      if (it == dependencies.end()) {
        continue;
      }
      ArrayRef<Operation *> succDeps = it->second;

      // Check if 'op' is a dependency of 'succ'.
      if (std::find(succDeps.begin(), succDeps.end(), op) != succDeps.end()) {
        // Simulate scheduling 'op' and check if 'succ' would still have
        // unscheduled deps.
        for (Operation *dep : succDeps) {
          if (dep == op) {
            continue;
          }
          if (!scheduledOpsMap.contains(dep)) {
            // Found a successor that would still be blocked.
            return true;
          }
        }
      }
    }
    return false;
  };

  std::stable_sort(schedulableOps.begin(), schedulableOps.end(),
                   [&](mlir::Operation *a, mlir::Operation *b) {
                     bool aBlocked = hasBlockedSuccessor(a);
                     bool bBlocked = hasBlockedSuccessor(b);
                     if (aBlocked != bBlocked) {
                       return aBlocked > bBlocked;
                     }
                     return false;
                   });

  return schedulableOps;
}

bool Scheduler::canSchedule(mlir::Operation *op) {
  for (mlir::Operation *dep : dependencies[op]) {
    if (!scheduledOpsMap.count(dep) || isa<ttnn::UpdateCacheOp>(op) ||
        isa<ttir::FillCacheOp>(op)) {
      return false;
    }
  }

  return true;
}

void Scheduler::scheduleOp(mlir::Operation *op) {
  assert(!scheduledOpsMap.contains(op) && "op is already scheduled");
  scheduledOpsMap.insert(op);
  schedule.push_back(op);
}

std::unique_ptr<Scheduler> Scheduler::snapshot() {
  return std::make_unique<Scheduler>(*this);
}

llvm::SmallVector<mlir::Operation *> Scheduler::getSchedule() const {
  return schedule;
}

bool Scheduler::hasUnscheduledOps() const {
  return scheduledOpsMap.size() < opsWithinFuncOp.size();
}

} // namespace mlir::tt::scheduler
