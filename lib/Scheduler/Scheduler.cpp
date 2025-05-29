// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Scheduler/Scheduler.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
      funcOps.push_back(&op);
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
      funcOps(scheduler.funcOps),
      dependencies(scheduler.dependencies) {}

llvm::SmallVector<mlir::Operation *> Scheduler::getScheduleableOps() {
  llvm::SmallVector<mlir::Operation *> scheduleableOps;
  for (auto &op : funcOps) {
    if (!scheduledOpsMap.contains(op) && canSchedule(op)) {
      scheduleableOps.push_back(op);
    }
  }

  // Only sort if more than one schedulable op
  if (scheduleableOps.size() > 1) {
    auto has_blocked_successor = [&](mlir::Operation *op) -> bool {
      // A successor is any op for which 'op' is a dependency
      for (auto &succ : funcOps) {
        if (succ == op) continue;
        auto it = dependencies.find(succ);
        if (it == dependencies.end()) continue;
        const auto &succ_deps = it->second;
        // Is 'op' a dependency of 'succ'?
        if (std::find(succ_deps.begin(), succ_deps.end(), op) != succ_deps.end()) {
          // Simulate scheduling 'op': would 'succ' still have unscheduled deps?
          bool still_blocked = false;
          for (auto *dep : succ_deps) {
            if (dep == op) continue;
            if (!scheduledOpsMap.contains(dep)) {
              still_blocked = true;
              break;
            }
          }
          if (still_blocked) {
            return true; // Found a successor that would still be blocked
          }
        }
      }
      return false;
    };
    std::stable_sort(scheduleableOps.begin(), scheduleableOps.end(),
      [&](mlir::Operation *a, mlir::Operation *b) {
        bool a_blocked = has_blocked_successor(a);
        bool b_blocked = has_blocked_successor(b);
        // Prioritize ops with blocked successors
        if (a_blocked != b_blocked) return a_blocked > b_blocked;
        return false;
      });
  }

  return scheduleableOps;
}

bool Scheduler::canSchedule(mlir::Operation *op) {
  for (mlir::Operation *dep : dependencies[op]) {
    if (!scheduledOpsMap.count(dep)) {
      return false;
    }
  }

  return true;
}

void Scheduler::scheduleOp(mlir::Operation *op) {
  assert(!scheduledOpsMap.count(op) && "op is already scheduled");
  scheduledOpsMap.insert(op);
  schedule.push_back(op);
}

std::unique_ptr<Scheduler> Scheduler::snapshot() {
  return std::make_unique<Scheduler>(*this);
}

llvm::SmallVector<mlir::Operation *> Scheduler::getSchedule() const {
  return schedule;
}

bool Scheduler::hasUnscheduledOps() const { return scheduledOpsMap.size() < funcOps.size(); }

} // namespace mlir::tt::scheduler
