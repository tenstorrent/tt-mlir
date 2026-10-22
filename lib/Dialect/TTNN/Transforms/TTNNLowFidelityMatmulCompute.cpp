// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TTNNLowFidelityMatmulCompute
// ============================
//
// Runs the convolutions and matmuls at LoFi with bf16 accumulation. See the
// pass description in Passes.td.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNLOWFIDELITYMATMULCOMPUTE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Kill switch (comma list in TTMLIR_OPT_DISABLE).
static bool lowFidelityDisabled() {
  static const bool disabled = [] {
    const char *env = std::getenv("TTMLIR_OPT_DISABLE");
    return env && llvm::StringRef(env).contains("lofi");
  }();
  return disabled;
}

// Release-build experiment hook: TTMLIR_MATMUL_FIDELITY=lofi|hifi2|hifi3|hifi4
// and TTMLIR_MATMUL_FP32_ACC=0|1 override the pass's choice (HiFi2, fp32
// accumulation).
static MathFidelity fidelity() {
  static const MathFidelity f = [] {
    const char *env = std::getenv("TTMLIR_MATMUL_FIDELITY");
    llvm::StringRef s(env ? env : "");
    if (s == "hifi2") {
      return MathFidelity::HiFi2;
    }
    if (s == "hifi3") {
      return MathFidelity::HiFi3;
    }
    if (s == "hifi4") {
      return MathFidelity::HiFi4;
    }
    if (s == "lofi") {
      return MathFidelity::LoFi;
    }
    return MathFidelity::HiFi2;
  }();
  return f;
}
// Release-build experiment hook: TTMLIR_MATMUL_KEEP_PRECISE="<loc substring>
// [,<loc substring>...]" leaves the matching ops at the pipeline defaults.
static bool keepPrecise(Operation *op) {
  static const std::vector<std::string> pats = [] {
    std::vector<std::string> v;
    const char *env = std::getenv("TTMLIR_MATMUL_KEEP_PRECISE");
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
// The highest-resolution stage keeps the pipeline fidelity: every downstream
// layer inherits its rounding, and on an image backbone HiFi2 on the convs
// above this spatial size alone cost 0.03 of PCC while HiFi2 everywhere
// else cost nothing measurable. Same threshold as the large-spatial conv
// config search. TTMLIR_MATMUL_LARGE_SPATIAL_PIXELS overrides it (0 lowers
// every op).
static int64_t largeSpatialPixels() {
  static const int64_t v = [] {
    const char *env = std::getenv("TTMLIR_MATMUL_LARGE_SPATIAL_PIXELS");
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
static bool fp32Acc() {
  static const bool v = [] {
    const char *env = std::getenv("TTMLIR_MATMUL_FP32_ACC");
    return !env || llvm::StringRef(env) != "0";
  }();
  return v;
}

class TTNNLowFidelityMatmulCompute
    : public impl::TTNNLowFidelityMatmulComputeBase<
          TTNNLowFidelityMatmulCompute> {
public:
  using impl::TTNNLowFidelityMatmulComputeBase<
      TTNNLowFidelityMatmulCompute>::TTNNLowFidelityMatmulComputeBase;

  void runOnOperation() final {
    if (lowFidelityDisabled()) {
      return;
    }
    MLIRContext *ctx = &getContext();
    auto lower = [&](Operation *op) {
      auto cfgOp = dyn_cast<TTNNComputeKernelConfigOpInterface>(op);
      if (!cfgOp) {
        return;
      }
      DeviceComputeKernelConfigAttr cfg = cfgOp.getComputeConfigAttr();
      if (!cfg) {
        cfg = DeviceComputeKernelConfigAttr::get(ctx);
      }
      cfg = cfg.withMathFidelity(fidelity()).withFp32DestAccEn(fp32Acc());
      cfgOp.setComputeConfigAttr(cfg);
    };
    getOperation()->walk([&](Operation *op) {
      if (isa<Conv2dOp, ConvTranspose2dOp, MatmulOp, LinearOp>(op) &&
          !keepPrecise(op) && !isLargeSpatial(op)) {
        lower(op);
      }
    });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
