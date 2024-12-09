// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Scheduler/Scheduler.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsDialect.h.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace mlir::tt::scheduler {

// TTNN op is scheduleable if it is not an EmptyOp and has at least one result.
bool isTTNNScheduleableOp(mlir::Operation *op) {
  return isa<ttnn::TTNNDialect>(op->getDialect()) && op->getNumResults() > 0 &&
         !llvm::isa<ttnn::EmptyOp>(op);
}

bool isTTIROp(mlir::Operation *op) {
  return isa<ttir::TTIRDialect>(op->getDialect());
}

bool Scheduler::isTTShedulableOp(mlir::Operation *op) {
  return isTTNNScheduleableOp(op) || isTTIROp(op);
}

// Init the dependencies map of all ops which are TTIR ops
Scheduler::Scheduler(func::FuncOp *func) {
  for (auto &op : func->getOps()) {
    if (isTTShedulableOp(&op)) {
      dependencies[&op] = {};
      unscheduledOps.insert(&op);
    }
  }

  for (auto &op : func->getOps()) {
    // Skip non TTIR operations
    // Skip operations which do not implement DestinationStyleOpInterface
    if (!isTTShedulableOp(&op)) {
      continue;
    }

    OpResult result = op.getResult(0);

    for (mlir::Operation *use : result.getUsers()) {
      // Skip non TTIR operations
      // Skip operations which set the result
      if (isTTShedulableOp(use) && use->getResult(0) != result) {
        dependencies[use].push_back(&op);
      }
    }
  }
}

Scheduler::Scheduler(const Scheduler &scheduler)
    : scheduledOpsMap(scheduler.scheduledOpsMap), schedule(scheduler.schedule),
      unscheduledOps(scheduler.unscheduledOps),
      dependencies(scheduler.dependencies) {}

llvm::SmallVector<mlir::Operation *> Scheduler::getScheduleableOps() {
  llvm::SmallVector<mlir::Operation *> scheduleableOps;
  for (auto &op : unscheduledOps) {
    if (canSchedule(op)) {
      scheduleableOps.push_back(op);
    }
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
  scheduledOpsMap.insert(op);
  unscheduledOps.erase(op);
  schedule.push_back(op);
}

std::unique_ptr<Scheduler> Scheduler::snapshot() {
  return std::make_unique<Scheduler>(*this);
}

llvm::SmallVector<mlir::Operation *> Scheduler::getSchedule() const {
  return schedule;
}

bool Scheduler::hasUnscheduledOps() const { return !unscheduledOps.empty(); }
} // namespace mlir::tt::scheduler
