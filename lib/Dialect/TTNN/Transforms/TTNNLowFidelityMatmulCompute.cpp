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
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
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
// Ops downstream of a sampling op (ttnn.grid_sample) keep the pipeline
// fidelity: a resampled feature map feeds regression heads whose outputs lost
// more correlation under HiFi2 than the image-space backbone did, while the
// backbone's lowering was validated at block level.
// TTMLIR_MATMUL_LOWER_AFTER_SAMPLING=1 lowers them too.
static bool lowerAfterSampling() {
  static const bool v = [] {
    const char *env = std::getenv("TTMLIR_MATMUL_LOWER_AFTER_SAMPLING");
    return env && llvm::StringRef(env) == "1";
  }();
  return v;
}
// HiFi2 keeps every mantissa bit of a block-float (bfp_bf8) operand, so on an
// op whose activation and weight are both block-float it changes nothing but
// the time; on a bf16 operand it drops the low mantissa bits. Only ops with
// block-float inputs are lowered (a bias operand does not count).
// TTMLIR_MATMUL_LOWER_BF16_INPUTS=1 lowers ops with bf16 inputs too.
static bool lowerBf16Inputs() {
  static const bool v = [] {
    const char *env = std::getenv("TTMLIR_MATMUL_LOWER_BF16_INPUTS");
    return env && llvm::StringRef(env) == "1";
  }();
  return v;
}
static bool isBlockFloat(Value v) {
  auto rt = mlir::dyn_cast<RankedTensorType>(v.getType());
  auto layout = rt ? mlir::dyn_cast_or_null<TTNNLayoutAttr>(rt.getEncoding())
                   : TTNNLayoutAttr();
  return layout && layout.getDataType() == ttcore::DataType::BFP_BFloat8;
}
// A convolution's weight operand is the raw weight at this stage; the dtype
// it is prepared in is the conv config's `weights_dtype`.
static bool weightIsBlockFloat(Operation *op) {
  Conv2dConfigAttr cfg;
  if (auto conv = mlir::dyn_cast<Conv2dOp>(op)) {
    cfg = conv.getConv2dConfigAttr();
  } else if (auto convT = mlir::dyn_cast<ConvTranspose2dOp>(op)) {
    cfg = convT.getConv2dConfigAttr();
  }
  if (cfg && cfg.getWeightsDtype().has_value() &&
      *cfg.getWeightsDtype() == ttcore::DataType::BFP_BFloat8) {
    return true;
  }
  return isBlockFloat(op->getOperand(1));
}
static bool inputsAreBlockFloat(Operation *op) {
  if (lowerBf16Inputs()) {
    return true;
  }
  return isBlockFloat(op->getOperand(0)) && weightIsBlockFloat(op);
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
    getOperation()->walk([&](func::FuncOp func) {
      llvm::DenseSet<Operation *> afterSampling;
      if (!lowerAfterSampling()) {
        afterSampling = utils::collectOpsDownstreamOfSampling(func);
      }
      func.walk([&](Operation *op) {
        // An op that carries `ttnn.keep_compute_config` chose its own
        // compute config (e.g. a matmul built with bf16 accumulation for
        // its sub-block width).
        if (isa<Conv2dOp, ConvTranspose2dOp, MatmulOp, LinearOp>(op) &&
            !keepPrecise(op) && !isLargeSpatial(op) &&
            !afterSampling.contains(op) && inputsAreBlockFloat(op) &&
            !op->hasAttr("ttnn.keep_compute_config")) {
          lower(op);
        }
      });
    });
  }
};

} // namespace
} // namespace mlir::tt::ttnn
