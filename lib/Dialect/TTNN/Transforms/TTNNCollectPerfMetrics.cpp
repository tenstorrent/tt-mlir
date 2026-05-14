// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <string>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNCOLLECTPERFMETRICS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

struct OperationMetrics {
  std::string opName;
  std::string location;
  bool isSharded = false;
  bool isSpilledToDRAM = false;
  bool hasSystemMemory = false;
  std::string layoutInfo;
};

// First-cut single-chip perf ceiling estimate. See
// docs/superpowers/specs/2026-05-14-perf-target-estimation-design.md.
struct PerfTargets {
  // Hardware description.
  std::string arch;
  unsigned chipCountInSystemDesc = 0;
  uint64_t workerGridCores = 0;
  uint64_t dramBandwidthBytesPerSec = 0;
  uint64_t peakFlopsLofi = 0;
  uint64_t peakFlopsHifi2 = 0;
  uint64_t peakFlopsHifi3 = 0;
  uint64_t peakFlopsHifi4 = 0;

  // Graph statistics.
  uint64_t paramCount = 0;
  uint64_t paramMemoryBytes = 0;
  uint64_t kvCacheCount = 0;
  uint64_t kvCacheMemoryBytes = 0;
  uint64_t inputCount = 0;
  uint64_t inputMemoryBytes = 0;
  // Embedding-feeding params that are double-counted in DRAM-bound rooflines
  // when tt-xla compiles tied embeddings as untied (input embedding lookup
  // accesses one row per token — effectively negligible — while the LM head
  // matmul reads the full V*H weight; if both tensors are present they show
  // up as two separate args here). For the "effective" DRAM-bound figure we
  // subtract these out, matching what a properly-tied compile would read.
  uint64_t embeddingParamCount = 0;
  uint64_t embeddingParamMemoryBytes = 0;
  uint64_t effectiveParamCount = 0;
  uint64_t effectiveParamMemoryBytes = 0;

  uint64_t matmulFlops = 0;
  uint64_t linearFlops = 0;
  uint64_t conv2dFlops = 0;
  uint64_t sparseMatmulFlops = 0;
  uint64_t totalFlops = 0;

  // Derived quantities.
  double dramRooflineTimeSec = 0.0;
  double computeRooflineTimeSecLofi = 0.0;
  double computeRooflineTimeSecHifi2 = 0.0;
  double computeRooflineTimeSecHifi3 = 0.0;
  double computeRooflineTimeSecHifi4 = 0.0;
  std::string bound;
  double topPerfTimeSec = 0.0;
  double topPerfSamplesPerSec = 0.0;
};

// Single place to encode the empirical utilization assumption. Real-world
// kernels rarely hit theoretical roofline; we approximate by 2x.
inline double topPerfTimeFromRoofline(double dramTime, double computeTime) {
  return 2.0 * std::max(dramTime, computeTime);
}

// Logical scalar shape of a tensor. In TTNN MLIR, the outer tensor shape is
// always in scalar (logical) units even when the element type is a TileType
// (the tile element is a storage hint; the memref inside the layout encoding
// is what carries the per-tile dimensions). So this is just tt.getShape().
inline llvm::SmallVector<int64_t>
getScalarTensorShape(mlir::RankedTensorType tt) {
  return llvm::SmallVector<int64_t>(tt.getShape());
}

inline uint64_t getScalarVolume(mlir::RankedTensorType tt) {
  uint64_t v = 1;
  for (int64_t d : tt.getShape()) {
    if (d <= 0) {
      return 0;
    }
    v *= static_cast<uint64_t>(d);
  }
  return v;
}

// Bytes per scalar element for a TileType. tile.getSizeBytes() gives bytes
// per tile (e.g. ~1056 for a 32x32 BFP8 tile); divide by tile element count
// to get bytes per scalar element (~1.03 for BFP8 BF8).
inline double getBytesPerScalarElement(mlir::Type elementType) {
  if (auto tile = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    auto shape = tile.getShape();
    uint64_t tileElements = 1;
    for (int64_t d : shape) {
      tileElements *= static_cast<uint64_t>(d);
    }
    if (tileElements == 0) {
      return 0.0;
    }
    return static_cast<double>(tile.getSizeBytes()) /
           static_cast<double>(tileElements);
  }
  return static_cast<double>(elementType.getIntOrFloatBitWidth()) / 8.0;
}

// Storage bytes for a tensor at on-device storage type. The tensor shape is
// already scalar (see getScalarTensorShape); for TileType elements we use the
// per-scalar tile byte size so BFP8 weights come out as ~1 byte/element.
inline uint64_t getTensorMemoryBytes(mlir::RankedTensorType tt) {
  uint64_t volume = getScalarVolume(tt);
  if (volume == 0) {
    return 0;
  }
  double bytes = static_cast<double>(volume) *
                 getBytesPerScalarElement(tt.getElementType());
  return static_cast<uint64_t>(bytes + 0.5);
}

// Returns Hout and Wout for a Conv2dOp using the explicit attributes on the
// op itself. We deliberately do not look at the result tensor shape because
// the output is flattened to (1, 1, N*Hout*Wout, C) and is harder to invert.
inline std::pair<int64_t, int64_t> getConv2dOutputSpatial(ttnn::Conv2dOp op) {
  int64_t Hin = op.getInputHeight();
  int64_t Win = op.getInputWidth();
  auto kernel = op.getKernelSize();
  auto stride = op.getStride();
  auto padding = op.getPadding();
  auto dilation = op.getDilation();

  int64_t KH = kernel[0];
  int64_t KW = kernel[1];
  int64_t sH = stride[0];
  int64_t sW = stride[1];
  int64_t dH = dilation[0];
  int64_t dW = dilation[1];

  int64_t pTop, pBottom, pLeft, pRight;
  if (padding.size() == 2) {
    pTop = pBottom = padding[0];
    pLeft = pRight = padding[1];
  } else {
    pTop = padding[0];
    pBottom = padding[1];
    pLeft = padding[2];
    pRight = padding[3];
  }

  int64_t Hout = (Hin + pTop + pBottom - dH * (KH - 1) - 1) / sH + 1;
  int64_t Wout = (Win + pLeft + pRight - dW * (KW - 1) - 1) / sW + 1;
  return {Hout, Wout};
}

struct AggregatedMetrics {
  uint64_t totalOps = 0;
  uint64_t totalOpsWithOutputTensor = 0;
  uint64_t totalShardableOps = 0;
  uint64_t shardedOps = 0;
  uint64_t effectivelyShardedOps = 0;
  uint64_t shardedAndSpilledOps = 0;
  uint64_t dramSpilledOps = 0;
  uint64_t systemMemoryOps = 0;

  double shardedPercentage = 0.0;
  double effectivelyShardedPercentage = 0.0;
  double shardedAndSpilledPercentage = 0.0;
  double spilledPercentage = 0.0;
  double systemMemoryPercentage = 0.0;

  void calculatePercentages() {
    if (totalShardableOps == 0) {
      return;
    }
    shardedPercentage =
        (static_cast<double>(shardedOps) / totalShardableOps) * 100.0;
    effectivelyShardedPercentage =
        (static_cast<double>(effectivelyShardedOps) / totalShardableOps) *
        100.0;
    shardedAndSpilledPercentage =
        (static_cast<double>(shardedAndSpilledOps) / totalShardableOps) * 100.0;
    spilledPercentage =
        (static_cast<double>(dramSpilledOps) / totalShardableOps) * 100.0;
    systemMemoryPercentage =
        (static_cast<double>(systemMemoryOps) / totalShardableOps) * 100.0;
  }
};

class TTNNCollectPerfMetrics
    : public impl::TTNNCollectPerfMetricsBase<TTNNCollectPerfMetrics> {

private:
  // Get layout information as string for individual op analysis
  std::string getLayoutInfo(Operation *op) {
    assert(op->getNumResults());
    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    assert(tensorType);
    auto encoding = tensorType.getEncoding();
    assert(encoding);
    return llvm::formatv("{0}", encoding).str();
  }

  // Get location as string for individual op analysis
  std::string getLocationString(Operation *op) {
    if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc())) {
      return nameLoc.getName().str();
    }
    return "unknown";
  }

  // Analyze ops individually
  OperationMetrics analyzeOperation(Operation *op) {
    OperationMetrics metrics;
    metrics.opName = op->getName().getStringRef().str();
    metrics.location = getLocationString(op);
    metrics.layoutInfo = getLayoutInfo(op);
    metrics.isSharded = utils::producesShardedL1Layout(op);
    metrics.hasSystemMemory = utils::producesSystemMemoryLayout(op);

    return metrics;
  }

  // Find operations that are spilled to DRAM by analyzing toMemoryConfigOps
  void identifyDRAMSpills(func::FuncOp funcOp,
                          llvm::DenseSet<Value> &spilledValues) {
    funcOp.walk([&](ttnn::ToMemoryConfigOp toMemoryConfigOp) {
      if (utils::producesDRAMLayout(toMemoryConfigOp)) {
        // Mark the input operand's defining op's result as spilled
        Value input = toMemoryConfigOp.getInput();
        if (input.getDefiningOp() &&
            utils::producesShardedL1Layout(input.getDefiningOp())) {
          spilledValues.insert(input);
        }
      }
    });
  }

  void addSummaryToJson(llvm::json::Object &jsonOutput,
                        const AggregatedMetrics &aggregatedMetrics) {
    llvm::json::Object summary;
    summary["total_ops"] = static_cast<int64_t>(aggregatedMetrics.totalOps);
    summary["total_ops_with_output_tensor"] =
        static_cast<int64_t>(aggregatedMetrics.totalOpsWithOutputTensor);
    summary["total_shardable_ops"] =
        static_cast<int64_t>(aggregatedMetrics.totalShardableOps);
    summary["sharded_ops"] = static_cast<int64_t>(aggregatedMetrics.shardedOps);
    summary["effectively_sharded_ops"] =
        static_cast<int64_t>(aggregatedMetrics.effectivelyShardedOps);
    summary["sharded_and_spilled_ops"] =
        static_cast<int64_t>(aggregatedMetrics.shardedAndSpilledOps);
    summary["dram_spilled_ops"] =
        static_cast<int64_t>(aggregatedMetrics.dramSpilledOps);
    summary["system_memory_ops"] =
        static_cast<int64_t>(aggregatedMetrics.systemMemoryOps);
    summary["sharded_percentage"] = aggregatedMetrics.shardedPercentage;
    summary["effectively_sharded_percentage"] =
        aggregatedMetrics.effectivelyShardedPercentage;
    summary["spilled_percentage"] = aggregatedMetrics.spilledPercentage;
    summary["system_memory_percentage"] =
        aggregatedMetrics.systemMemoryPercentage;
    jsonOutput["summary"] = std::move(summary);
  }

  void
  addVerboseOutputToJson(llvm::json::Object &jsonOutput,
                         const std::vector<OperationMetrics> &operationDetails,
                         const llvm::StringMap<int> &operationTypeCounts) {
    llvm::json::Array operations;
    for (const auto &opMetrics : operationDetails) {
      llvm::json::Object opJson;
      opJson["operation"] = opMetrics.opName;
      opJson["location"] = opMetrics.location;
      opJson["is_sharded"] = opMetrics.isSharded;
      opJson["is_spilled_to_dram"] = opMetrics.isSpilledToDRAM;
      opJson["has_system_memory"] = opMetrics.hasSystemMemory;
      opJson["layout_info"] = opMetrics.layoutInfo;
      operations.push_back(std::move(opJson));
    }
    jsonOutput["shardable_operations"] = std::move(operations);

    // Counts of all operation types
    llvm::json::Object operationTypeBreakdown;
    for (const auto &pair : operationTypeCounts) {
      operationTypeBreakdown[pair.first()] = pair.second;
    }
    jsonOutput["operation_type_breakdown"] = std::move(operationTypeBreakdown);
  }

  // Pull DRAM bandwidth and peak FLOPS from a hardcoded per-arch table. The
  // values are not present in the SystemDescAttr today, so we look them up
  // from the arch + grid dims. Single-chip n150 only — see design doc.
  void populateHardwareLimits(PerfTargets &t, ttcore::ChipDescAttr chipDesc) {
    auto arch = chipDesc.getArch().getValue();
    auto gridShape = chipDesc.getGrid();
    uint64_t numCores = 1;
    for (int64_t d : gridShape) {
      if (d > 0) {
        numCores *= static_cast<uint64_t>(d);
      }
    }
    t.workerGridCores = numCores;

    switch (arch) {
    case ttcore::Arch::WormholeB0:
      t.arch = "wormhole_b0";
      // Wormhole B0 n150: 12 channels of GDDR6, 288 GB/s aggregate.
      t.dramBandwidthBytesPerSec = 288ULL * 1000ULL * 1000ULL * 1000ULL;
      // 4 TFLOPS / engine at LoFi @ 1 GHz times worker grid.
      t.peakFlopsLofi = 4ULL * 1000ULL * 1000ULL * 1000ULL * 1000ULL * numCores;
      t.peakFlopsHifi2 = t.peakFlopsLofi / 2ULL;
      t.peakFlopsHifi3 = (t.peakFlopsLofi * 4ULL) / 12ULL; // ~1.33 / engine
      t.peakFlopsHifi4 = t.peakFlopsLofi / 4ULL;
      break;
    case ttcore::Arch::Blackhole:
      t.arch = "blackhole";
      // Placeholder values. Refine when we evaluate on Blackhole.
      t.dramBandwidthBytesPerSec = 512ULL * 1000ULL * 1000ULL * 1000ULL;
      t.peakFlopsLofi = 8ULL * 1000ULL * 1000ULL * 1000ULL * 1000ULL * numCores;
      t.peakFlopsHifi2 = t.peakFlopsLofi / 2ULL;
      t.peakFlopsHifi3 = (t.peakFlopsLofi * 4ULL) / 12ULL;
      t.peakFlopsHifi4 = t.peakFlopsLofi / 4ULL;
      break;
    }
  }

  // Trace a Value back to the originating BlockArgument of `funcOp` through
  // pass-through ops (ttcore.load_cached, layout conversions, typecasts).
  // Returns the BlockArgument if found, else nullptr.
  BlockArgument traceToFuncArg(Value v, func::FuncOp funcOp) {
    while (v) {
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(v)) {
        if (blockArg.getOwner()->getParentOp() == funcOp) {
          return blockArg;
        }
        return nullptr;
      }
      Operation *def = v.getDefiningOp();
      if (!def || def->getNumOperands() == 0) {
        return nullptr;
      }
      // ttcore.load_cached(@const_eval, [%arg, ...]) — first operand is the
      // upstream func arg for the embedding-feeding case in the forward
      // function. For other pass-through ops (to_layout, typecast,
      // to_memory_config), the first operand is the input tensor we want to
      // follow.
      v = def->getOperand(0);
    }
    return nullptr;
  }

  // Walk forward function arguments and classify them as
  // params/constants vs KV cache vs runtime inputs. Numbers are in scalar
  // elements; bytes use on-device storage type (so BFP8 weights are 1 byte
  // per element through the TileType byte path).
  //
  // Also identifies the embedding-feeding parameter args (the ones consumed
  // by a ttnn.embedding op) and tracks them separately so callers can compute
  // an "effective" DRAM-bound weight memory that excludes them — see the
  // PerfTargets fields for why.
  void collectArgumentStats(PerfTargets &t, func::FuncOp funcOp) {
    // First, find every BlockArgument that flows into a ttnn.embedding op as
    // the weight operand. The set is per-function and usually has 0 or 1
    // entries (some models have multiple embeddings, e.g. position embeddings).
    llvm::DenseSet<unsigned> embeddingArgIndices;
    funcOp.walk([&](ttnn::EmbeddingOp embeddingOp) {
      if (auto blockArg = traceToFuncArg(embeddingOp.getWeight(), funcOp)) {
        embeddingArgIndices.insert(blockArg.getArgNumber());
      }
    });

    for (BlockArgument arg : funcOp.getArguments()) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType) {
        continue;
      }
      uint64_t scalarVol = getScalarVolume(tensorType);
      uint64_t bytes = getTensorMemoryBytes(tensorType);
      unsigned idx = arg.getArgNumber();

      if (ttcore::isKVCacheArgument(funcOp, idx)) {
        t.kvCacheCount += scalarVol;
        t.kvCacheMemoryBytes += bytes;
        continue;
      }

      auto argType = ttcore::getFunctionArgumentType(funcOp, idx);
      if (argType == ttcore::ArgumentType::Input) {
        t.inputCount += scalarVol;
        t.inputMemoryBytes += bytes;
      } else {
        // Parameter, Constant, or Default.
        t.paramCount += scalarVol;
        t.paramMemoryBytes += bytes;
        if (embeddingArgIndices.contains(idx)) {
          t.embeddingParamCount += scalarVol;
          t.embeddingParamMemoryBytes += bytes;
        }
      }
    }

    // Effective weight = total params minus the embedding lookup weight,
    // since the embedding op only touches batch_size rows per step (sparse
    // access). Even when the model uses tied weights, only the LM head matmul
    // reads the full V*H tensor on each forward; the embedding lookup does
    // not. Subtracting here makes the DRAM roofline match what a tied compile
    // would actually read.
    t.effectiveParamCount = t.paramCount > t.embeddingParamCount
                                ? t.paramCount - t.embeddingParamCount
                                : 0;
    t.effectiveParamMemoryBytes =
        t.paramMemoryBytes > t.embeddingParamMemoryBytes
            ? t.paramMemoryBytes - t.embeddingParamMemoryBytes
            : 0;
  }

  // Walk all ops in the forward function and count FLOPs for matmul-class
  // ops only. Higher-level ops like scaled_dot_product_attention decompose
  // into matmuls by the time TTNNCollectPerfMetrics runs, so we pick them up
  // automatically without per-op accounting.
  void collectComputeStats(PerfTargets &t, func::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      if (auto matmul = mlir::dyn_cast<ttnn::MatmulOp>(op)) {
        auto aType = mlir::cast<RankedTensorType>(matmul.getA().getType());
        auto outType =
            mlir::cast<RankedTensorType>(matmul.getResult().getType());
        auto aShape = getScalarTensorShape(aType);
        auto outShape = getScalarTensorShape(outType);
        if (aShape.size() < 2 || outShape.empty()) {
          return;
        }
        int64_t k = matmul.getTransposeA() ? aShape[aShape.size() - 2]
                                           : aShape[aShape.size() - 1];
        uint64_t outVol = 1;
        for (int64_t d : outShape) {
          if (d <= 0) {
            return;
          }
          outVol *= static_cast<uint64_t>(d);
        }
        uint64_t flops = 2ULL * outVol * static_cast<uint64_t>(k);
        t.matmulFlops += flops;
        t.totalFlops += flops;
        return;
      }
      if (auto linear = mlir::dyn_cast<ttnn::LinearOp>(op)) {
        auto aType = mlir::cast<RankedTensorType>(linear.getA().getType());
        auto outType =
            mlir::cast<RankedTensorType>(linear.getResult().getType());
        auto aShape = getScalarTensorShape(aType);
        auto outShape = getScalarTensorShape(outType);
        if (aShape.size() < 2 || outShape.empty()) {
          return;
        }
        int64_t k = linear.getTransposeA() ? aShape[aShape.size() - 2]
                                           : aShape[aShape.size() - 1];
        uint64_t outVol = 1;
        for (int64_t d : outShape) {
          if (d <= 0) {
            return;
          }
          outVol *= static_cast<uint64_t>(d);
        }
        uint64_t flops = 2ULL * outVol * static_cast<uint64_t>(k);
        // Bias add is +outVol FLOPs, negligible vs the matmul.
        if (linear.getBias()) {
          flops += outVol;
        }
        t.linearFlops += flops;
        t.totalFlops += flops;
        return;
      }
      if (auto conv = mlir::dyn_cast<ttnn::Conv2dOp>(op)) {
        int64_t N = conv.getBatchSize();
        int64_t Cin = conv.getInChannels();
        int64_t Cout = conv.getOutChannels();
        int64_t groups = conv.getGroups();
        auto kernel = conv.getKernelSize();
        if (kernel.size() < 2 || groups == 0) {
          return;
        }
        auto [Hout, Wout] = getConv2dOutputSpatial(conv);
        if (Hout <= 0 || Wout <= 0) {
          return;
        }
        uint64_t flops =
            2ULL * static_cast<uint64_t>(N) * static_cast<uint64_t>(Hout) *
            static_cast<uint64_t>(Wout) * static_cast<uint64_t>(Cin) *
            static_cast<uint64_t>(Cout) * static_cast<uint64_t>(kernel[0]) *
            static_cast<uint64_t>(kernel[1]) / static_cast<uint64_t>(groups);
        t.conv2dFlops += flops;
        t.totalFlops += flops;
        return;
      }
      if (auto sm = mlir::dyn_cast<ttnn::SparseMatmulOp>(op)) {
        auto aType = mlir::cast<RankedTensorType>(sm.getA().getType());
        auto outType = mlir::cast<RankedTensorType>(sm.getResult().getType());
        auto aShape = getScalarTensorShape(aType);
        auto outShape = getScalarTensorShape(outType);
        if (aShape.size() < 2 || outShape.empty()) {
          return;
        }
        int64_t k = aShape[aShape.size() - 1];
        uint64_t outVol = 1;
        for (int64_t d : outShape) {
          if (d <= 0) {
            return;
          }
          outVol *= static_cast<uint64_t>(d);
        }
        uint64_t flops = 2ULL * outVol * static_cast<uint64_t>(k);
        t.sparseMatmulFlops += flops;
        t.totalFlops += flops;
        return;
      }
    });
  }

  PerfTargets computePerfTargets(ModuleOp module, func::FuncOp forwardFunc) {
    PerfTargets t;
    auto systemDescAttr = module->getAttrOfType<ttcore::SystemDescAttr>(
        ttcore::SystemDescAttr::name);
    if (!systemDescAttr) {
      llvm::errs() << "TTNNCollectPerfMetrics: no ttcore.system_desc on the "
                      "module; skipping perf target estimate.\n";
      return t;
    }
    auto chipDescs = systemDescAttr.getChipDescs();
    t.chipCountInSystemDesc = chipDescs.size();
    if (t.chipCountInSystemDesc == 0) {
      llvm::errs() << "TTNNCollectPerfMetrics: system desc has no chip "
                      "descriptors; skipping perf target estimate.\n";
      return t;
    }
    if (t.chipCountInSystemDesc > 1) {
      llvm::errs() << "TTNNCollectPerfMetrics: system desc has "
                   << t.chipCountInSystemDesc
                   << " chips; perf target estimate uses only chip 0 "
                      "(single-chip n150 assumption).\n";
    }
    populateHardwareLimits(t, chipDescs[0]);

    collectArgumentStats(t, forwardFunc);
    collectComputeStats(t, forwardFunc);

    // Effective weight bytes that have to be read from DRAM each step. We
    // exclude the embedding-feeding parameter because the input embedding
    // lookup only touches one row per token (sparse access, ~kB per step),
    // not the full V*H weight — and when tied weights are preserved, that
    // same tensor serves the LM head matmul once, which we account for via
    // the matmul FLOPs and the const-eval'd LM-head weight (a separate arg
    // in tt-xla's untied compile). See PerfTargets fields for context.
    uint64_t totalWeightBytes =
        t.effectiveParamMemoryBytes + t.kvCacheMemoryBytes;
    if (t.dramBandwidthBytesPerSec > 0) {
      t.dramRooflineTimeSec = static_cast<double>(totalWeightBytes) /
                              static_cast<double>(t.dramBandwidthBytesPerSec);
    }
    auto flops = static_cast<double>(t.totalFlops);
    if (t.peakFlopsLofi > 0) {
      t.computeRooflineTimeSecLofi =
          flops / static_cast<double>(t.peakFlopsLofi);
    }
    if (t.peakFlopsHifi2 > 0) {
      t.computeRooflineTimeSecHifi2 =
          flops / static_cast<double>(t.peakFlopsHifi2);
    }
    if (t.peakFlopsHifi3 > 0) {
      t.computeRooflineTimeSecHifi3 =
          flops / static_cast<double>(t.peakFlopsHifi3);
    }
    if (t.peakFlopsHifi4 > 0) {
      t.computeRooflineTimeSecHifi4 =
          flops / static_cast<double>(t.peakFlopsHifi4);
    }
    // Default math fidelity for the headline `bound`/`top_perf_*` fields is
    // HiFi2. Rationale: tt-mlir's TTNN matmuls run BFP8 weights × BF16
    // activations, which is the HiFi2 path on Wormhole B0 (LoFi requires
    // BFP4/BFP2 inputs, HiFi3/HiFi4 are for FP16/FP32-accumulate paths). LoFi
    // is too optimistic for the workloads we estimate against — at 256
    // TFLOPS it always shows decode as DRAM-bound and over-promises prefill.
    // We keep all four `compute_time_sec_*` fields in the JSON so downstream
    // consumers can re-derive the bound at a different fidelity if they know
    // a model uses something else.
    t.bound = t.dramRooflineTimeSec >= t.computeRooflineTimeSecHifi2
                  ? "dram"
                  : "compute";
    t.topPerfTimeSec = topPerfTimeFromRoofline(t.dramRooflineTimeSec,
                                               t.computeRooflineTimeSecHifi2);
    t.topPerfSamplesPerSec =
        t.topPerfTimeSec > 0.0 ? 1.0 / t.topPerfTimeSec : 0.0;

    return t;
  }

  void addPerfTargetsToJson(llvm::json::Object &jsonOutput,
                            const PerfTargets &t) {
    llvm::json::Object pt;
    pt["arch"] = t.arch;
    pt["chip_count_in_system_desc"] =
        static_cast<int64_t>(t.chipCountInSystemDesc);
    pt["single_chip_assumption"] = true;
    pt["worker_grid_cores"] = static_cast<int64_t>(t.workerGridCores);
    pt["dram_bandwidth_bytes_per_sec"] =
        static_cast<int64_t>(t.dramBandwidthBytesPerSec);

    llvm::json::Object peak;
    peak["lofi"] = static_cast<int64_t>(t.peakFlopsLofi);
    peak["hifi2"] = static_cast<int64_t>(t.peakFlopsHifi2);
    peak["hifi3"] = static_cast<int64_t>(t.peakFlopsHifi3);
    peak["hifi4"] = static_cast<int64_t>(t.peakFlopsHifi4);
    pt["peak_flops"] = std::move(peak);

    llvm::json::Object params;
    params["count"] = static_cast<int64_t>(t.paramCount);
    params["memory_bytes"] = static_cast<int64_t>(t.paramMemoryBytes);
    params["memory_gb"] =
        static_cast<double>(t.paramMemoryBytes) / (1024.0 * 1024.0 * 1024.0);
    params["embedding_count"] = static_cast<int64_t>(t.embeddingParamCount);
    params["embedding_memory_bytes"] =
        static_cast<int64_t>(t.embeddingParamMemoryBytes);
    params["effective_count"] = static_cast<int64_t>(t.effectiveParamCount);
    params["effective_memory_bytes"] =
        static_cast<int64_t>(t.effectiveParamMemoryBytes);
    params["effective_memory_gb"] =
        static_cast<double>(t.effectiveParamMemoryBytes) /
        (1024.0 * 1024.0 * 1024.0);
    pt["params"] = std::move(params);

    llvm::json::Object kv;
    kv["count"] = static_cast<int64_t>(t.kvCacheCount);
    kv["memory_bytes"] = static_cast<int64_t>(t.kvCacheMemoryBytes);
    kv["memory_gb"] =
        static_cast<double>(t.kvCacheMemoryBytes) / (1024.0 * 1024.0 * 1024.0);
    pt["kv_cache"] = std::move(kv);

    llvm::json::Object inp;
    inp["count"] = static_cast<int64_t>(t.inputCount);
    inp["memory_bytes"] = static_cast<int64_t>(t.inputMemoryBytes);
    pt["inputs"] = std::move(inp);

    llvm::json::Object compute;
    compute["total_flops"] = static_cast<int64_t>(t.totalFlops);
    llvm::json::Object breakdown;
    breakdown["matmul"] = static_cast<int64_t>(t.matmulFlops);
    breakdown["linear"] = static_cast<int64_t>(t.linearFlops);
    breakdown["conv2d"] = static_cast<int64_t>(t.conv2dFlops);
    breakdown["sparse_matmul"] = static_cast<int64_t>(t.sparseMatmulFlops);
    compute["breakdown"] = std::move(breakdown);
    pt["compute"] = std::move(compute);

    llvm::json::Object rl;
    rl["dram_time_sec"] = t.dramRooflineTimeSec;
    rl["compute_time_sec_lofi"] = t.computeRooflineTimeSecLofi;
    rl["compute_time_sec_hifi2"] = t.computeRooflineTimeSecHifi2;
    rl["compute_time_sec_hifi3"] = t.computeRooflineTimeSecHifi3;
    rl["compute_time_sec_hifi4"] = t.computeRooflineTimeSecHifi4;
    rl["bound"] = t.bound;
    rl["top_perf_time_sec"] = t.topPerfTimeSec;
    rl["top_perf_samples_per_sec"] = t.topPerfSamplesPerSec;
    pt["roofline"] = std::move(rl);

    jsonOutput["perf_targets"] = std::move(pt);
  }

  std::string generateAutoFilename(ModuleOp module) {
    std::string baseName = "UnnamedModule";

    // Try to get module name
    if (auto moduleSymName = module.getSymName()) {
      baseName = moduleSymName->str();
      if (baseName.front() == '@') {
        baseName = baseName.substr(1);
      }
    } else {
      // Try to get the first function name if module has no name
      bool foundFunction = false;
      module->walk([&](func::FuncOp funcOp) {
        if (!foundFunction && !ttmlir::utils::isConstEvalFunc(funcOp)) {
          baseName = funcOp.getName().str();
          foundFunction = true;
        }
      });
    }

    return "perf_metrics/" + baseName + "_perf_metrics.json";
  }

  void ensureDirectoryExists(StringRef filePath) {
    llvm::SmallString<256> dirPath(filePath);
    llvm::sys::path::remove_filename(dirPath);

    if (dirPath.empty()) {
      return; // Current directory always exists
    }
    std::error_code ec = llvm::sys::fs::create_directories(dirPath);
    if (ec) {
      llvm::report_fatal_error(Twine("Failed to create directory: ") + dirPath +
                               " - " + ec.message());
    }
  }

  void writeJsonToFile(llvm::json::Object jsonOutput, ModuleOp module) {
    std::string outputPath;
    if (!ttnnPerfMetricsOutputFile.empty()) {
      // User specified a custom output file
      outputPath = ttnnPerfMetricsOutputFile.getValue();
      if (!llvm::StringRef(outputPath).ends_with(".json")) {
        outputPath += ".json";
      }
    } else {
      // Generate automatic filename
      outputPath = generateAutoFilename(module);
    }

    ensureDirectoryExists(outputPath);

    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec);
    if (ec) {
      llvm::report_fatal_error(Twine("Failed to open output file: ") +
                               outputPath + " - " + ec.message());
    }

    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(jsonOutput)))
       << "\n";
    os.close();
  }

public:
  using impl::TTNNCollectPerfMetricsBase<
      TTNNCollectPerfMetrics>::TTNNCollectPerfMetricsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::vector<OperationMetrics> operationDetails;
    AggregatedMetrics aggregatedMetrics;
    llvm::DenseSet<Value> spilledValues;
    llvm::StringMap<int> operationTypeCounts;
    PerfTargets perfTargets;
    bool perfTargetsComputed = false;

    module->walk([&](func::FuncOp funcOp) {
      if (ttnnEnableTrace) {
        if (!ttmlir::utils::isTraceMainFunc(funcOp)) {
          return;
        }
      } else {
        if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
          return;
        }
      }

      // Single-chip perf target estimate. Only computed for the first matching
      // function; if there are multiple forwards we only score the first one.
      if (!perfTargetsComputed) {
        perfTargets = computePerfTargets(module, funcOp);
        perfTargetsComputed = true;
      }

      // First pass: identify DRAM spills
      identifyDRAMSpills(funcOp, spilledValues);

      // Second pass: analyze all operations
      funcOp->walk([&](Operation *op) {
        // Debug: Count operations by type
        std::string opName = op->getName().getStringRef().str();
        operationTypeCounts[opName]++;

        // Skip operations which never change (some appear only for
        // enable-trace=true)
        if (isa<ttnn::GetDeviceOp>(op)) {
          return;
        }
        aggregatedMetrics.totalOps++;

        // Skip operations without tensor results
        if (op->getNumResults() == 0) {
          return;
        }
        if (!mlir::isa<RankedTensorType>(op->getResult(0).getType())) {
          return;
        }
        aggregatedMetrics.totalOpsWithOutputTensor++;

        // Skip operations which make no sense to shard/spill, as they are
        // helpers
        if (isa<ttnn::ToMemoryConfigOp>(op)) {
          return;
        }

        // Only analyze TTNN operations, skip ttcore::load_cached
        if (!isa<TTNNDialect>(op->getDialect())) {
          return;
        }

        aggregatedMetrics.totalShardableOps++;

        OperationMetrics opMetrics = analyzeOperation(op);

        // Check if this operation's result is spilled and update aggregated
        // metrics
        if (opMetrics.isSharded) {
          Value result = op->getResult(0);
          if (spilledValues.count(result)) {
            opMetrics.isSpilledToDRAM = true;
            aggregatedMetrics.shardedAndSpilledOps++;
          } else {
            aggregatedMetrics.effectivelyShardedOps++;
          }
          aggregatedMetrics.shardedOps++;
        }

        if (utils::producesDRAMLayout(op)) {
          aggregatedMetrics.dramSpilledOps++;
        }

        if (opMetrics.hasSystemMemory) {
          aggregatedMetrics.systemMemoryOps++;
        }

        operationDetails.push_back(opMetrics);
      });
    });

    aggregatedMetrics.calculatePercentages();

    llvm::json::Object jsonOutput;
    addSummaryToJson(jsonOutput, aggregatedMetrics);

    if (perfTargetsComputed) {
      addPerfTargetsToJson(jsonOutput, perfTargets);
    }

    if (ttnnPerfMetricsVerboseOutputEnabled) {
      addVerboseOutputToJson(jsonOutput, operationDetails, operationTypeCounts);
    }

    writeJsonToFile(std::move(jsonOutput), module);
  }
};

} // namespace

} // namespace mlir::tt::ttnn
