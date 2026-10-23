// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TTNNBlockFloatIntermediates
// ===========================
//
// Writes the tiled intermediate activations of an image backbone as
// bfp_bf8. See the pass description in Passes.td.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNBLOCKFLOATINTERMEDIATES
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Kill switch (comma list in TTMLIR_OPT_DISABLE).
static bool blockFloatDisabled() {
  static const bool disabled = [] {
    const char *env = std::getenv("TTMLIR_OPT_DISABLE");
    return env && llvm::StringRef(env).contains("bfp8_intermediates");
  }();
  return disabled;
}

static TTNNLayoutAttr layoutOf(Value v) {
  auto rt = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!rt) {
    return nullptr;
  }
  return mlir::dyn_cast_or_null<TTNNLayoutAttr>(rt.getEncoding());
}

static bool isTiledBf16(Value v) {
  TTNNLayoutAttr l = layoutOf(v);
  return l && l.isTiled() && l.getDataType() == ttcore::DataType::BFloat16;
}

static bool isBfp8(Value v) {
  TTNNLayoutAttr l = layoutOf(v);
  return l && l.getDataType() == ttcore::DataType::BFP_BFloat8;
}

// Ops that only move or view data: the result carries its input's dtype.
static bool isPassThrough(Operation *op) {
  return mlir::isa<ReshapeOp, SliceStaticOp, ToMemoryConfigOp, ReallocateOp>(
      op);
}

// A layout op that keeps the tensor tiled (a memory move expressed as
// to_tensor_spec, or a tile -> tile to_layout) passes the dtype through; one
// that untilizes does not.
static bool isTiledLayoutOp(Operation *op) {
  if (!mlir::isa<ToLayoutOp, ToTensorSpecOp>(op) || op->getNumResults() != 1) {
    return false;
  }
  TTNNLayoutAttr in = layoutOf(op->getOperand(0));
  TTNNLayoutAttr out = layoutOf(op->getResult(0));
  return in && out && in.isTiled() && out.isTiled();
}

// A tilize: row-major in, tiled out. Its result may take the block-float
// dtype (the decomposition fuses the cast into the tilize).
static bool isTilize(Operation *op) {
  if (!mlir::isa<ToLayoutOp, ToTensorSpecOp>(op) || op->getNumResults() != 1) {
    return false;
  }
  TTNNLayoutAttr in = layoutOf(op->getOperand(0));
  TTNNLayoutAttr out = layoutOf(op->getResult(0));
  return in && out && !in.isTiled() && out.isTiled();
}

// The highest-resolution stage keeps bf16: its rounding reaches every
// downstream layer, and on an image backbone bfp8 on those outputs alone
// cost several hundredths of PCC. Same threshold as the fidelity pass.
// TTMLIR_BFP8_LARGE_SPATIAL_PIXELS overrides it (0 lowers every op).
static int64_t largeSpatialPixels() {
  static const int64_t v = [] {
    const char *env = std::getenv("TTMLIR_BFP8_LARGE_SPATIAL_PIXELS");
    if (env) {
      return static_cast<int64_t>(std::strtoll(env, nullptr, 10));
    }
    return static_cast<int64_t>(100000);
  }();
  return v;
}
static bool isLargeSpatial(Operation *op) {
  int64_t pixels = 0;
  if (auto conv = mlir::dyn_cast<Conv2dOp>(op)) {
    pixels = static_cast<int64_t>(conv.getInputHeight()) * conv.getInputWidth();
  } else if (auto convT = mlir::dyn_cast<ConvTranspose2dOp>(op)) {
    pixels = static_cast<int64_t>(convT.getInputHeight()) * convT.getInputWidth();
  } else {
    return false;
  }
  return pixels > largeSpatialPixels();
}

// Release-build experiment hook: TTMLIR_BFP8_KEEP_PRECISE="<loc substring>
// [,...]" leaves the matching producers in bf16.
static bool keepPrecise(Operation *op) {
  static const std::vector<std::string> pats = [] {
    std::vector<std::string> v;
    const char *env = std::getenv("TTMLIR_BFP8_KEEP_PRECISE");
    llvm::SmallVector<llvm::StringRef> items;
    llvm::StringRef(env ? env : "").split(items, ',', -1, false);
    for (llvm::StringRef it : items) {
      v.push_back(it.trim().str());
    }
    return v;
  }();
  if (pats.empty()) {
    return false;
  }
  std::string locName;
  if (auto nameLoc = mlir::dyn_cast<NameLoc>(op->getLoc())) {
    locName = nameLoc.getName().str();
  }
  for (const std::string &p : pats) {
    if (!p.empty() && llvm::StringRef(locName).contains(p)) {
      return true;
    }
  }
  return false;
}

// Only tensors worth the bandwidth take the block-float dtype: the precision
// cost of bfp8 is the same for every tensor, the gain is proportional to the
// bytes moved, and on an image backbone the small deep-stage tensors cost
// 0.01 of PCC between them for a few tens of microseconds. The threshold is
// a fraction of the largest tiled activation of the function (so it scales
// with the model), TTMLIR_BFP8_MIN_FRACTION (default 0.15); 0 lowers every
// eligible op.
static double minFraction() {
  static const double v = [] {
    const char *env = std::getenv("TTMLIR_BFP8_MIN_FRACTION");
    return env ? std::strtod(env, nullptr) : 0.15;
  }();
  return v;
}
static uint64_t tensorBytes(Value v) {
  auto rt = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!rt) {
    return 0;
  }
  Type scalar = rt.getElementType();
  if (!scalar.isIntOrFloat()) {
    return 0;
  }
  return static_cast<uint64_t>(rt.getNumElements()) *
         llvm::divideCeil(scalar.getIntOrFloatBitWidth(), 8);
}

// Ops downstream of a sampling op (ttnn.grid_sample) keep bf16 results
// unless TTMLIR_BFP8_AFTER_SAMPLING=1: the resampled feature map feeds the
// regression heads, whose outputs are the most sensitive to the 8-bit
// mantissa, while the image-space backbone's lowering was validated at block
// level.
static bool bfp8AfterSampling() {
  static const bool v = [] {
    const char *env = std::getenv("TTMLIR_BFP8_AFTER_SAMPLING");
    return env && llvm::StringRef(env) == "1";
  }();
  return v;
}

static bool isProducerSeed(Operation *op,
                           const llvm::DenseSet<Operation *> &afterSampling) {
  return mlir::isa<Conv2dOp, ConvTranspose2dOp, MatmulOp, LinearOp>(op) &&
         !isLargeSpatial(op) && !keepPrecise(op) &&
         !afterSampling.contains(op);
}

// True when some use of `v`, followed through pass-through ops, leaves the
// tiled world: an untilize, the function return, or an op that is neither a
// consumer of tiles we know nor a pass-through.
static bool reachesRowMajorOrReturn(Value v, llvm::DenseSet<Value> &seen) {
  if (!seen.insert(v).second) {
    return false;
  }
  for (Operation *user : v.getUsers()) {
    if (mlir::isa<func::ReturnOp>(user)) {
      return true;
    }
    if (mlir::isa<DeallocateOp>(user)) {
      continue;
    }
    if (isPassThrough(user) || isTiledLayoutOp(user)) {
      if (reachesRowMajorOrReturn(user->getResult(0), seen)) {
        return true;
      }
      continue;
    }
    if (mlir::isa<ToLayoutOp, ToTensorSpecOp>(user)) {
      // Untilize (tiled in, row-major out).
      return true;
    }
    if (mlir::isa<Conv2dOp, ConvTranspose2dOp, MatmulOp, LinearOp, ConcatOp,
                  AddOp, MultiplyOp, SubtractOp, MaxPool2dOp>(user)) {
      // Tile consumers. A pool reads row-major, but the layout propagation
      // already placed the untilize in front of it, so a direct pool user
      // means the pool consumes tiles via its own halo (not the case today).
      continue;
    }
    return true;
  }
  return false;
}

static bool setDtype(Value v, ttcore::DataType dtype) {
  auto rt = mlir::dyn_cast<RankedTensorType>(v.getType());
  TTNNLayoutAttr l = layoutOf(v);
  if (!rt || !l || l.getDataType() == dtype) {
    return false;
  }
  v.setType(utils::RankedTensorTypeFactory::create(rt, dtype));
  return true;
}

class TTNNBlockFloatIntermediates
    : public impl::TTNNBlockFloatIntermediatesBase<
          TTNNBlockFloatIntermediates> {
public:
  using impl::TTNNBlockFloatIntermediatesBase<
      TTNNBlockFloatIntermediates>::TTNNBlockFloatIntermediatesBase;

  void runOnOperation() final {
    if (blockFloatDisabled()) {
      return;
    }
    ModuleOp module = getOperation();
    const ttcore::DataType bfp8 = ttcore::DataType::BFP_BFloat8;

    module->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }
      // 1. Seeds: tiled bf16 results of convolutions and matmuls that never
      //    leave the tiled world and are large enough to be worth it.
      llvm::DenseSet<Operation *> afterSampling;
      if (!bfp8AfterSampling()) {
        afterSampling = utils::collectOpsDownstreamOfSampling(func);
      }
      uint64_t largest = 0;
      func.walk([&](Operation *op) {
        if (isProducerSeed(op, afterSampling) && op->getNumResults() == 1 &&
            isTiledBf16(op->getResult(0))) {
          largest = std::max(largest, tensorBytes(op->getResult(0)));
        }
      });
      const uint64_t minBytes =
          static_cast<uint64_t>(minFraction() * static_cast<double>(largest));
      llvm::SmallVector<Value> worklist;
      func.walk([&](Operation *op) {
        if (!isProducerSeed(op, afterSampling) || op->getNumResults() != 1) {
          return;
        }
        Value r = op->getResult(0);
        if (!isTiledBf16(r) || tensorBytes(r) < minBytes) {
          return;
        }
        llvm::DenseSet<Value> seen;
        if (reachesRowMajorOrReturn(r, seen)) {
          return;
        }
        if (setDtype(r, bfp8)) {
          worklist.push_back(r);
        }
      });

      // 2. Forward propagation through pass-through ops; concat / binary ops
      //    whose inputs all carry bfp8 propagate too.
      auto propagate = [&]() {
        while (!worklist.empty()) {
          Value v = worklist.pop_back_val();
          for (Operation *user : v.getUsers()) {
            if (user->getNumResults() != 1) {
              continue;
            }
            Value r = user->getResult(0);
            if (isPassThrough(user) || isTiledLayoutOp(user)) {
              if (setDtype(r, bfp8)) {
                worklist.push_back(r);
              }
              continue;
            }
            if (mlir::isa<ConcatOp, AddOp, MultiplyOp, SubtractOp>(user)) {
              bool all = llvm::all_of(user->getOperands(), [](Value o) {
                return !mlir::isa<RankedTensorType>(o.getType()) || isBfp8(o);
              });
              if (all && isTiledBf16(r)) {
                llvm::DenseSet<Value> seen;
                if (!reachesRowMajorOrReturn(r, seen) && setDtype(r, bfp8)) {
                  worklist.push_back(r);
                }
              }
            }
          }
        }
      };
      propagate();

      // 3. Mixed inputs: a concat or binary op with one bfp8 input and one
      //    bf16 input. A bf16 input that comes from a tilize (through
      //    pass-through ops) is tilized straight to bfp8 (fused cast); any
      //    other gets an explicit typecast in front of the consumer.
      OpBuilder builder(func.getContext());
      bool changed = true;
      while (changed) {
        changed = false;
        func.walk([&](Operation *op) {
          if (!mlir::isa<ConcatOp, AddOp, MultiplyOp, SubtractOp>(op)) {
            return;
          }
          bool anyBfp8 = false;
          for (Value o : op->getOperands()) {
            anyBfp8 |= isBfp8(o);
          }
          if (!anyBfp8) {
            return;
          }
          for (OpOperand &use : op->getOpOperands()) {
            Value o = use.get();
            if (!isTiledBf16(o)) {
              continue;
            }
            // Walk back through pass-through ops to a tilize.
            Value cur = o;
            Operation *tilize = nullptr;
            while (Operation *def = cur.getDefiningOp()) {
              if (isTilize(def)) {
                // A tilize shared by several consumers may still take the
                // block-float dtype when none of its uses leaves the tiled
                // world (they all accept bfp8 tiles).
                llvm::DenseSet<Value> seen;
                if (!reachesRowMajorOrReturn(def->getResult(0), seen)) {
                  tilize = def;
                }
                break;
              }
              if (isPassThrough(def) || isTiledLayoutOp(def)) {
                cur = def->getOperand(0);
                continue;
              }
              break;
            }
            if (tilize) {
              setDtype(tilize->getResult(0), bfp8);
              worklist.push_back(tilize->getResult(0));
              propagate();
              changed = true;
              return;
            }
            // A binary op producer (a scalar multiply of a pooled tensor, say)
            // writes the block-float dtype itself: binary_ng fuses the cast
            // into its pack, whatever its inputs' dtype.
            if (Operation *def = cur.getDefiningOp()) {
              if (mlir::isa<AddOp, MultiplyOp, SubtractOp>(def) &&
                  def->getNumResults() == 1 && def->getResult(0) == cur) {
                llvm::DenseSet<Value> seen;
                if (!reachesRowMajorOrReturn(cur, seen) && setDtype(cur, bfp8)) {
                  worklist.push_back(cur);
                  propagate();
                  changed = true;
                  return;
                }
              }
            }
            builder.setInsertionPoint(op);
            auto rt = mlir::cast<RankedTensorType>(o.getType());
            auto castType = utils::RankedTensorTypeFactory::create(rt, bfp8);
            auto cast = builder.create<TypecastOp>(op->getLoc(), castType, o);
            use.set(cast.getResult());
            // With every input now bfp8 the op's own result follows (and
            // propagates), unless its uses leave the tiled world.
            bool all = llvm::all_of(op->getOperands(), [](Value x) {
              return !mlir::isa<RankedTensorType>(x.getType()) || isBfp8(x);
            });
            if (all && op->getNumResults() == 1 &&
                isTiledBf16(op->getResult(0))) {
              llvm::DenseSet<Value> seen;
              if (!reachesRowMajorOrReturn(op->getResult(0), seen) &&
                  setDtype(op->getResult(0), bfp8)) {
                worklist.push_back(op->getResult(0));
                propagate();
              }
            }
            changed = true;
            return;
          }
        });
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
