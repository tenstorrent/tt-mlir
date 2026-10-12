// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TTNNL1ProducerFirst
// ===================
//
// For an op with several tensor operands, order its independent producers so
// that the one reading an L1 tensor runs before the one reading only DRAM /
// host tensors. See the pass description in Passes.td.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNL1PRODUCERFIRST
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static bool ruleDisabled() {
  static const bool disabled = [] {
    const char *env = std::getenv("TTMLIR_OPT_DISABLE");
    return env && llvm::StringRef(env).contains("l1_producer_first");
  }();
  return disabled;
}

static bool probeEnabled() {
  static const bool enabled = std::getenv("TTMLIR_SPILL_PROBE") != nullptr;
  return enabled;
}

// True if `v` is a device tensor in L1.
static bool isL1Tensor(Value v) {
  auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!type) {
    return false;
  }
  auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(type.getEncoding());
  return layout && layout.hasL1BufferType();
}

// Position (in block order) of the most recently defined L1 operand of `op`,
// or -1 when it reads no L1 tensor. Spills have not been inserted yet when
// this pass runs, so "DRAM-fed" producers are not visible as such: what
// distinguishes them is that their L1 input was produced long ago (and will
// be spilled), while the sibling reads a value produced just before.
static int64_t newestL1OperandPos(Operation *op,
                                  const llvm::DenseMap<Operation *, int64_t> &pos) {
  int64_t newest = -1;
  for (Value operand : op->getOperands()) {
    if (!isL1Tensor(operand)) {
      continue;
    }
    Operation *def = operand.getDefiningOp();
    auto it = def ? pos.find(def) : pos.end();
    if (it != pos.end()) {
      newest = std::max(newest, it->second);
    }
  }
  return newest;
}

// Every operand of `op` is a block argument or defined before `point`.
static bool operandsDominate(Operation *op, Operation *point) {
  for (Value operand : op->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def) {
      continue;
    }
    if (def->getBlock() != point->getBlock() || !def->isBeforeInBlock(point)) {
      return false;
    }
  }
  return true;
}

class TTNNL1ProducerFirstPass
    : public impl::TTNNL1ProducerFirstBase<TTNNL1ProducerFirstPass> {
public:
  using impl::TTNNL1ProducerFirstBase<
      TTNNL1ProducerFirstPass>::TTNNL1ProducerFirstBase;

  void runOnOperation() final {
    if (ruleDisabled()) {
      return;
    }
    getOperation()->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }
      llvm::DenseMap<Operation *, int64_t> position;
      int64_t idx = 0;
      func.walk([&](Operation *op) { position[op] = idx++; });
      // Collect first, then move: moving while walking the block is unsafe.
      llvm::SmallVector<std::pair<Operation *, Operation *>> moves;
      func.walk([&](Operation *consumer) {
        if (consumer->getNumOperands() < 2 ||
            !mlir::isa<TTNNDialect>(consumer->getDialect())) {
          return;
        }
        llvm::SmallVector<Operation *> producers;
        for (Value operand : consumer->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (def && def->getBlock() == consumer->getBlock() &&
              def->getNumResults() == 1 &&
              mlir::isa<TTNNDialect>(def->getDialect()) &&
              !llvm::is_contained(producers, def)) {
            producers.push_back(def);
          }
        }
        for (Operation *dramFed : producers) {
          int64_t oldInput = newestL1OperandPos(dramFed, position);
          for (Operation *l1Fed : producers) {
            if (l1Fed == dramFed || !dramFed->isBeforeInBlock(l1Fed)) {
              continue;
            }
            // Only compute producers are worth moving: they are the ops
            // whose circular buffers evict their own just-produced input.
            // A layout op (reshard, layout or dtype change, view) is short
            // and placed right before its consumer on purpose; moving one
            // earlier only lengthens the live range of its output.
            if (mlir::isa<ToMemoryConfigOp, ToTensorSpecOp, ToLayoutOp,
                          ReshapeOp, TypecastOp, ReallocateOp>(l1Fed)) {
              continue;
            }
            // Move l1Fed first only when its L1 input is more recent than
            // every L1 input of dramFed (recent first: LIFO frees L1 early).
            int64_t recentInput = newestL1OperandPos(l1Fed, position);
            if (recentInput < 0 || recentInput <= oldInput) {
              continue;
            }
            // l1Fed must not depend on dramFed (directly or through ops in
            // between): all its operands must already be defined before
            // dramFed.
            if (!operandsDominate(l1Fed, dramFed)) {
              continue;
            }
            moves.emplace_back(l1Fed, dramFed);
          }
        }
      });
      // Debug filter: TTMLIR_PRODUCER_FIRST_ONLY=<substr>[,<substr>...] keeps
      // only the moves whose moved op's location contains one of the
      // substrings; TTMLIR_PRODUCER_FIRST_SKIP=<substr>[,...] drops those.
      static const char *onlyEnv = std::getenv("TTMLIR_PRODUCER_FIRST_ONLY");
      static const char *skipEnv = std::getenv("TTMLIR_PRODUCER_FIRST_SKIP");
      auto locMatches = [](Operation *op, const char *list) {
        std::string locStr;
        llvm::raw_string_ostream os(locStr);
        op->getLoc().print(os);
        llvm::SmallVector<llvm::StringRef> parts;
        llvm::StringRef(list).split(parts, ',', -1, false);
        for (llvm::StringRef part : parts) {
          if (!part.empty() && llvm::StringRef(locStr).contains(part)) {
            return true;
          }
        }
        return false;
      };
      for (auto &[l1Fed, dramFed] : moves) {
        if (!dramFed->isBeforeInBlock(l1Fed) ||
            !operandsDominate(l1Fed, dramFed)) {
          continue; // an earlier move already changed the order
        }
        if ((onlyEnv && !locMatches(l1Fed, onlyEnv)) ||
            (skipEnv && locMatches(l1Fed, skipEnv))) {
          if (probeEnabled()) {
            llvm::errs() << "[L1_PRODUCER_FIRST] filtered out move of "
                         << l1Fed->getName() << " @" << l1Fed->getLoc() << "\n";
          }
          continue;
        }
        if (probeEnabled()) {
          llvm::errs() << "[L1_PRODUCER_FIRST] moving "
                       << l1Fed->getName() << " @" << l1Fed->getLoc()
                       << " before " << dramFed->getName() << " @"
                       << dramFed->getLoc() << "\n";
        }
        l1Fed->moveBefore(dramFed);
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
