# Matmul Program Config Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move matmul program config generation into MatmulRuleBook with shape-aware output hint filtering, so large-M prefill matmuls get BlockSharded (2D mcast) configs and decode matmuls can get DRAM-sharded configs.

**Architecture:** The MatmulRuleBook::getOutputHints() method becomes the single source of matmul program configs for the MemoryLayoutPropagation optimizer path. It deduplicates legal configs by (bufferType, memLayout), filters by M-dimension heuristics, generates per-config-type program configs, and returns them as output hints. A new isValidOutputHintForInputs() override prunes invalid (hint, input layout) combinations before backend validation.

**Tech Stack:** C++ / MLIR / LLVM, tt-mlir compiler infrastructure, llvm-lit testing

**Note on ShardSolver:** The `fillOpSpecificAttrs()` matmul branch in LegalOpConfigAnalysis and `getUniqueTestConfigsForMatmulLinear()` in OptimizerUtils are also used by ShardSolver (a separate optimizer path). They remain unchanged in this plan. MatmulRuleBook::getOutputHints() ignores the pre-baked attrs and generates its own.

---

### Task 1: Refactor MatmulProgramConfig.cpp into per-type generators

Split the monolithic `generateMatmulProgramConfig()` into four public functions. No behavior change yet — pure refactor.

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h`
- Modify: `lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp`

- [ ] **Step 1: Update the header with four new function declarations**

In `include/ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h`, replace the single declaration with four. Keep the old one as a wrapper for backward compatibility (ShardSolver path still calls it via LegalOpConfigAnalysis).

```cpp
// include/ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"

#include <optional>

namespace mlir::tt::ttnn {

/// Generate 1D mcast config for WidthSharded output (mcast_in0=true).
std::optional<mlir::Attribute>
generateMatmul1DWidthConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                            int64_t Kt, TTNNLayoutAttr outputLayout,
                            UnaryWithParamAttr fusedActivation,
                            int64_t maxSubblockSize, bool fuseBatch);

/// Generate 1D mcast config for HeightSharded output (mcast_in0=false).
std::optional<mlir::Attribute>
generateMatmul1DHeightConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                             int64_t Kt, TTNNLayoutAttr outputLayout,
                             UnaryWithParamAttr fusedActivation,
                             int64_t maxSubblockSize, bool fuseBatch);

/// Generate 2D mcast config for BlockSharded output.
std::optional<mlir::Attribute>
generateMatmul2DConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt, int64_t Kt,
                       TTNNLayoutAttr outputLayout,
                       UnaryWithParamAttr fusedActivation,
                       int64_t maxSubblockSize, bool fuseBatch);

/// Generate DRAM-sharded config for WidthSharded output with Mt==1 (decode).
std::optional<mlir::Attribute>
generateMatmulDRAMShardedConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                                int64_t Kt, TTNNLayoutAttr outputLayout,
                                UnaryWithParamAttr fusedActivation,
                                int64_t maxSubblockSize, bool fuseBatch);

/// Compute max subblock size from compute kernel config.
/// Exposed so MatmulRuleBook can call it in getOutputHints().
int64_t getMaxSubblockSize(DeviceComputeKernelConfigAttr computeConfig);

/// Legacy wrapper: dispatches to the per-type generators based on output
/// sharding. Kept for backward compatibility with ShardSolver path.
std::optional<mlir::Attribute>
generateMatmulProgramConfig(Operation *op, TTNNLayoutAttr outputLayout);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H
```

- [ ] **Step 2: Refactor the .cpp into four generators + legacy wrapper**

In `lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp`, extract the existing `generateMatmul1DProgramConfig` and `generateMatmul2DProgramConfig` static functions into the new public API. The internal logic is unchanged — this is a pure signature refactor.

The key changes:
- Rename `generateMatmul1DProgramConfig` → split into `generateMatmul1DWidthConfig` and `generateMatmul1DHeightConfig` (the existing function already branches on `mcastIn0`).
- Rename `generateMatmul2DProgramConfig` → `generateMatmul2DConfig`.
- Add `generateMatmulDRAMShardedConfig` as a new function (stub returning nullopt for now — implemented in Task 3).
- Make `getMaxSubblockSize` non-static and declare it in the header (MatmulRuleBook needs it in Task 2).
- Keep `generateMatmulProgramConfig` as a wrapper that extracts shapes from `op`, then dispatches to the per-type generators.

```cpp
// lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp

// Keep existing helpers: divUp, getMaxSubblockSize, largestDivisorUpTo unchanged.

std::optional<mlir::Attribute>
generateMatmul1DWidthConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                            int64_t Kt, TTNNLayoutAttr outputLayout,
                            UnaryWithParamAttr fusedActivation,
                            int64_t maxSubblockSize, bool fuseBatch) {
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);
  int64_t numCores = gridX * gridY;

  int64_t perCoreM = Mt;
  int64_t perCoreN = divUp(Nt, numCores);

  // in0_block_w heuristic for width-sharded (existing logic)
  constexpr int64_t kLargeNtThreshold = 128;
  int64_t in0BlockW;
  if (Nt > kLargeNtThreshold) {
    in0BlockW = (Kt % 2 == 0) ? 2 : 1;
  } else {
    if (Kt % 8 == 0) {
      in0BlockW = 8;
    } else if (Kt % 4 == 0) {
      in0BlockW = 4;
    } else if (Kt % 2 == 0) {
      in0BlockW = 2;
    } else {
      in0BlockW = 1;
    }
  }

  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;
  int64_t outSubblockH = 1;
  int64_t outSubblockW = largestDivisorUpTo(outBlockW, maxSubblockSize);

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);
  auto hopCoresAttr = CoreRangeSetAttr::get(ctx, {});

  return MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      fuseBatch, /*fusedActivation=*/fusedActivation, /*mcastIn0=*/true,
      /*gather_in0=*/false, hopCoresAttr, /*num_global_cb_receivers=*/0,
      /*untilize_out=*/false);
}

std::optional<mlir::Attribute>
generateMatmul1DHeightConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                             int64_t Kt, TTNNLayoutAttr outputLayout,
                             UnaryWithParamAttr fusedActivation,
                             int64_t maxSubblockSize, bool fuseBatch) {
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);
  int64_t numCores = gridX * gridY;

  int64_t perCoreM = divUp(Mt, numCores);
  int64_t perCoreN = Nt;
  int64_t in0BlockW = Kt;  // Hardware requirement: full K in shard width.

  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;
  int64_t outSubblockH = 1;
  int64_t outSubblockW = largestDivisorUpTo(outBlockW, maxSubblockSize);

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);
  auto hopCoresAttr = CoreRangeSetAttr::get(ctx, {});

  return MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      fuseBatch, /*fusedActivation=*/fusedActivation, /*mcastIn0=*/false,
      /*gather_in0=*/false, hopCoresAttr, /*num_global_cb_receivers=*/0,
      /*untilize_out=*/false);
}

std::optional<mlir::Attribute>
generateMatmul2DConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt, int64_t Kt,
                       TTNNLayoutAttr outputLayout,
                       UnaryWithParamAttr fusedActivation,
                       int64_t maxSubblockSize, bool fuseBatch) {
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);

  int64_t perCoreM = divUp(Mt, gridY);
  int64_t perCoreN = divUp(Nt, gridX);

  int64_t in0BlockW = (Kt % 2 == 0) ? 2 : 1;
  int64_t outSubblockH = 1;
  int64_t outSubblockW = largestDivisorUpTo(perCoreN, maxSubblockSize);
  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);

  return MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      /*transpose_mcast=*/false, /*fusedActivation=*/fusedActivation,
      fuseBatch);
}

std::optional<mlir::Attribute>
generateMatmulDRAMShardedConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                                int64_t Kt, TTNNLayoutAttr outputLayout,
                                UnaryWithParamAttr fusedActivation,
                                int64_t maxSubblockSize, bool fuseBatch) {
  // Stub — implemented in Task 3.
  return std::nullopt;
}

// Legacy wrapper for backward compatibility (ShardSolver path).
std::optional<mlir::Attribute>
generateMatmulProgramConfig(Operation *op, TTNNLayoutAttr outputLayout) {
  // Existing logic: extract shapes, compute fuseBatch, dispatch.
  // Keep this function EXACTLY as-is — it now calls the new per-type
  // generators internally instead of the old static functions.
  if (!outputLayout || !outputLayout.hasShardedL1TensorMemoryLayout()) {
    return std::nullopt;
  }

  TensorMemoryLayout outputMemLayout = outputLayout.getMemLayout().getValue();
  if (outputMemLayout != TensorMemoryLayout::WidthSharded &&
      outputMemLayout != TensorMemoryLayout::HeightSharded &&
      outputMemLayout != TensorMemoryLayout::BlockSharded) {
    return std::nullopt;
  }

  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> outShape = resultType.getShape();
  if (outShape.size() < 2) {
    return std::nullopt;
  }

  auto [inputA, inputB, activation] =
      llvm::TypeSwitch<Operation *,
                       std::tuple<Value, Value, std::optional<StringRef>>>(op)
          .Case<ttnn::MatmulOp, ttnn::LinearOp>([](auto matmulOp) {
            std::optional<StringRef> act;
            if (auto actAttr = matmulOp.getActivationAttr()) {
              act = actAttr.getValue();
            }
            return std::make_tuple(matmulOp.getA(), matmulOp.getB(), act);
          })
          .Default([](Operation *) {
            return std::make_tuple(nullptr, nullptr,
                                   std::optional<StringRef>{});
          });

  if (!inputA || !inputB) {
    return std::nullopt;
  }

  auto inputAType = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  auto inputBType = mlir::dyn_cast<RankedTensorType>(inputB.getType());
  if (!inputAType || !inputBType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> aShape = inputAType.getShape();
  llvm::ArrayRef<int64_t> bShape = inputBType.getShape();
  if (aShape.size() < 2 || bShape.size() < 2) {
    return std::nullopt;
  }

  bool fuseBatch = true;
  for (size_t i = 0; i < bShape.size() - 2; ++i) {
    if (bShape[i] != 1) {
      fuseBatch = false;
      break;
    }
  }

  int64_t M = outShape[outShape.size() - 2];
  int64_t N = outShape[outShape.size() - 1];
  int64_t K = aShape[aShape.size() - 1];
  int64_t Mt = divUp(M, TILE_HEIGHT);
  int64_t Nt = divUp(N, TILE_WIDTH);
  int64_t Kt = divUp(K, TILE_WIDTH);

  MLIRContext *ctx = op->getContext();
  UnaryWithParamAttr fusedActivation =
      ttnn::utils::getActivationAttr(ctx, activation);

  DeviceComputeKernelConfigAttr computeConfig = nullptr;
  if (auto computeConfigOp = dyn_cast<TTNNComputeKernelConfigOpInterface>(op)) {
    computeConfig = computeConfigOp.getComputeConfigAttr();
  }
  int64_t maxSubblockSize = getMaxSubblockSize(computeConfig);

  if (outputMemLayout == TensorMemoryLayout::BlockSharded) {
    return generateMatmul2DConfig(ctx, Mt, Nt, Kt, outputLayout,
                                  fusedActivation, maxSubblockSize, fuseBatch);
  }

  if (outputMemLayout == TensorMemoryLayout::WidthSharded) {
    return generateMatmul1DWidthConfig(ctx, Mt, Nt, Kt, outputLayout,
                                       fusedActivation, maxSubblockSize,
                                       fuseBatch);
  }

  return generateMatmul1DHeightConfig(ctx, Mt, Nt, Kt, outputLayout,
                                      fusedActivation, maxSubblockSize,
                                      fuseBatch);
}
```

- [ ] **Step 3: Build and verify no regressions**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds with no errors.

Run: `cmake --build build --target check-ttmlir 2>&1 | tail -20`
Expected: All existing tests pass (pure refactor, no behavior change).

- [ ] **Step 4: Commit**

```bash
git add include/ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h \
        lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp
git commit -m "refactor: split MatmulProgramConfig into per-type generators

Extract generateMatmul1DWidthConfig, generateMatmul1DHeightConfig,
generateMatmul2DConfig, and generateMatmulDRAMShardedConfig (stub) from
the monolithic generateMatmulProgramConfig. Legacy wrapper preserved for
ShardSolver backward compatibility. No behavior change."
```

---

### Task 2: Rewrite MatmulRuleBook::getOutputHints() with shape-aware filtering + config generation

This is the core architectural change. `getOutputHints()` generates its own program configs instead of relying on pre-baked attrs from LegalOpConfigAnalysis.

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h`
- Modify: `lib/Dialect/TTNN/Analysis/OpRules/MatmulRules.cpp`

- [ ] **Step 1: Add helper declarations to MatmulRules.h**

No new public method declarations needed yet (isValidOutputHintForInputs is Task 4). Just ensure the header includes MatmulProgramConfig.h.

```cpp
// include/ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/OpRuleBook.h"

namespace mlir::tt::ttnn {

struct MatmulRuleBook : OpRuleBook {
  /// Output hints: shape-aware filtering + program config generation.
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;

  /// Reject width-sharded inputs (accuracy issues).
  LayoutFilterFn getInputLayoutFilter() const override;

  /// Apply MatmulProgramConfig + fused activation dedup.
  void applyOpSpecificAttrs(Operation *op,
                            const BeamCandidate &candidate) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPRULES_MATMULRULES_H
```

- [ ] **Step 2: Rewrite getOutputHints() in MatmulRules.cpp**

Replace the existing implementation. The new version:
1. Deduplicates legalConfigs by (bufferType, memLayout) — simple helper, ignores opSpecificAttrs.
2. Filters by shape: reject WidthSharded for large Mt, HeightSharded for Mt<=1, L1 interleaved.
3. Generates program config(s) per surviving hint.

```cpp
// lib/Dialect/TTNN/Analysis/OpRules/MatmulRules.cpp

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {

// Threshold in tiles: Mt > this value triggers prefill path
// (reject WidthSharded, prefer BlockSharded 2D).
// 4 tiles = 128 rows. Matches where tt-metal models switch 1D -> 2D.
static constexpr int64_t kPrefillMtThreshold = 4;

static inline int64_t divUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Deduplicate legalConfigs by (bufferType, memLayout).
// Returns one representative partial layout per group, with no opSpecificAttrs.
static std::vector<TTNNLayoutAttr>
dedupByMemoryLayout(const std::vector<OpConfig> &configs) {
  struct Key {
    BufferType bufferType;
    TensorMemoryLayout memLayout;
    bool operator==(const Key &o) const {
      return bufferType == o.bufferType && memLayout == o.memLayout;
    }
  };
  struct KeyHash {
    size_t operator()(const Key &k) const {
      return llvm::hash_combine(k.bufferType, k.memLayout);
    }
  };

  std::unordered_map<Key, TTNNLayoutAttr, KeyHash> seen;
  for (const auto &cfg : configs) {
    if (!cfg.outputLayout) {
      continue;
    }
    auto memLayout = cfg.outputLayout.getMemLayout();
    if (!memLayout) {
      continue;
    }
    Key key{cfg.outputLayout.getBufferType(), memLayout.getValue()};
    if (seen.find(key) == seen.end()) {
      seen[key] = cfg.outputLayout.withIgnorePhysicalLayout(true);
    }
  }

  std::vector<TTNNLayoutAttr> result;
  result.reserve(seen.size());
  for (const auto &[key, layout] : seen) {
    result.push_back(layout);
  }
  return result;
}

// Extract M, K, N shapes and fuseBatch from a matmul/linear op.
static bool extractMatmulShapes(Operation *op, int64_t &Mt, int64_t &Nt,
                                int64_t &Kt, bool &fuseBatch,
                                std::optional<StringRef> &activation) {
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType.getRank() < 2) {
    return false;
  }

  auto [inputA, inputB, act] =
      llvm::TypeSwitch<Operation *,
                       std::tuple<Value, Value, std::optional<StringRef>>>(op)
          .Case<ttnn::MatmulOp, ttnn::LinearOp>([](auto matmulOp) {
            std::optional<StringRef> a;
            if (auto actAttr = matmulOp.getActivationAttr()) {
              a = actAttr.getValue();
            }
            return std::make_tuple(matmulOp.getA(), matmulOp.getB(), a);
          })
          .Default([](Operation *) {
            return std::make_tuple(nullptr, nullptr,
                                   std::optional<StringRef>{});
          });

  if (!inputA || !inputB) {
    return false;
  }

  auto aType = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  auto bType = mlir::dyn_cast<RankedTensorType>(inputB.getType());
  if (!aType || !bType || aType.getRank() < 2 || bType.getRank() < 2) {
    return false;
  }

  auto outShape = resultType.getShape();
  auto aShape = aType.getShape();
  auto bShape = bType.getShape();

  int64_t M = outShape[outShape.size() - 2];
  int64_t N = outShape[outShape.size() - 1];
  int64_t K = aShape[aShape.size() - 1];
  Mt = divUp(M, 32);
  Nt = divUp(N, 32);
  Kt = divUp(K, 32);

  fuseBatch = true;
  for (size_t i = 0; i < bShape.size() - 2; ++i) {
    if (bShape[i] != 1) {
      fuseBatch = false;
      break;
    }
  }

  activation = act;
  return true;
}

LayoutFilterFn MatmulRuleBook::getInputLayoutFilter() const {
  return layout_filter_utils::rejectWidthSharded;
}

OutputHints MatmulRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {
  // Extract shapes from the op.
  int64_t Mt, Nt, Kt;
  bool fuseBatch;
  std::optional<StringRef> activation;
  if (!extractMatmulShapes(op, Mt, Nt, Kt, fuseBatch, activation)) {
    // Fallback: return empty hints, let backend decide.
    return OutputHints{{OpConfig(TTNNLayoutAttr())}, {}};
  }

  MLIRContext *ctx = op->getContext();
  UnaryWithParamAttr fusedActivation =
      ttnn::utils::getActivationAttr(ctx, activation);

  DeviceComputeKernelConfigAttr computeConfig = nullptr;
  if (auto computeConfigOp =
          dyn_cast<TTNNComputeKernelConfigOpInterface>(op)) {
    computeConfig = computeConfigOp.getComputeConfigAttr();
  }
  int64_t maxSubblockSize = getMaxSubblockSize(computeConfig);

  // Deduplicate by (bufferType, memLayout).
  auto uniqueLayouts = dedupByMemoryLayout(legalConfigs);

  std::vector<OpConfig> hints;
  for (const auto &layout : uniqueLayouts) {
    auto memLayout = layout.getMemLayout();
    if (!memLayout) {
      continue;
    }
    auto memLayoutVal = memLayout.getValue();
    bool isL1 = layout.getBufferType() == BufferType::L1;

    // Filter L1 interleaved (existing rule).
    if (isL1 && memLayoutVal == TensorMemoryLayout::Interleaved) {
      continue;
    }

    // Shape-aware filtering.
    if (Mt > kPrefillMtThreshold &&
        memLayoutVal == TensorMemoryLayout::WidthSharded) {
      continue;
    }
    if (Mt <= 1 && memLayoutVal == TensorMemoryLayout::HeightSharded) {
      continue;
    }

    // Generate program config(s) for this output sharding.
    if (memLayoutVal == TensorMemoryLayout::WidthSharded) {
      auto cfg = generateMatmul1DWidthConfig(ctx, Mt, Nt, Kt, layout,
                                             fusedActivation, maxSubblockSize,
                                             fuseBatch);
      if (cfg) {
        hints.push_back(
            OpConfig(layout, MatmulAttrs{cfg, computeConfig}));
      }

      // Additional DRAM-sharded variant for decode (Mt == 1).
      if (Mt == 1) {
        auto dramCfg = generateMatmulDRAMShardedConfig(
            ctx, Mt, Nt, Kt, layout, fusedActivation, maxSubblockSize,
            fuseBatch);
        if (dramCfg) {
          hints.push_back(
              OpConfig(layout, MatmulAttrs{dramCfg, computeConfig}));
        }
      }
    } else if (memLayoutVal == TensorMemoryLayout::HeightSharded) {
      auto cfg = generateMatmul1DHeightConfig(ctx, Mt, Nt, Kt, layout,
                                              fusedActivation, maxSubblockSize,
                                              fuseBatch);
      if (cfg) {
        hints.push_back(
            OpConfig(layout, MatmulAttrs{cfg, computeConfig}));
      }
    } else if (memLayoutVal == TensorMemoryLayout::BlockSharded) {
      auto cfg = generateMatmul2DConfig(ctx, Mt, Nt, Kt, layout,
                                        fusedActivation, maxSubblockSize,
                                        fuseBatch);
      if (cfg) {
        hints.push_back(
            OpConfig(layout, MatmulAttrs{cfg, computeConfig}));
      }
    } else {
      // DRAM interleaved — no program config.
      hints.push_back(OpConfig(layout));
    }
  }

  // Always include a DRAM interleaved fallback (no program config).
  hints.push_back(OpConfig(TTNNLayoutAttr()));

  return OutputHints{hints, {}};
}

// applyOpSpecificAttrs remains UNCHANGED — keep existing implementation.
void MatmulRuleBook::applyOpSpecificAttrs(
    Operation *op, const BeamCandidate &candidate) const {
  auto matmulOp = dyn_cast<MatmulOp>(op);
  auto linearOp = dyn_cast<LinearOp>(op);
  if (!matmulOp && !linearOp) {
    return;
  }

  if (!std::holds_alternative<MatmulAttrs>(
          candidate.configHint.opSpecificAttrs)) {
    return;
  }
  MatmulAttrs matmulAttrs =
      std::get<MatmulAttrs>(candidate.configHint.opSpecificAttrs);
  if (!matmulAttrs.matmulProgramConfig.has_value()) {
    return;
  }
  auto programConfig = matmulAttrs.matmulProgramConfig.value();

  auto setConfigAndFixup = [&](auto concreteOp) {
    concreteOp.setMatmulProgramConfigAttr(programConfig);
    bool hasFusedActivation =
        llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
            .template Case<
                MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
                [](auto config) {
                  return config.getFusedActivation() != nullptr;
                })
            .Default([](mlir::Attribute) { return false; });
    if (hasFusedActivation) {
      concreteOp.removeActivationAttr();
    }
  };

  if (matmulOp) {
    setConfigAndFixup(matmulOp);
  } else {
    setConfigAndFixup(linearOp);
  }
}

} // namespace mlir::tt::ttnn
```

- [ ] **Step 3: Build and run tests**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds.

Run: `cmake --build build --target check-ttmlir 2>&1 | tail -20`
Expected: Tests pass. Some matmul tests may now produce different program configs due to shape-aware filtering (e.g., a 512x1024 matmul has Mt=16 > kPrefillMtThreshold=4, so WidthSharded is rejected and BlockSharded is preferred). Update FileCheck patterns in failing tests to match.

- [ ] **Step 4: Fix any failing lit tests**

The main test to check is `test/ttmlir/Dialect/TTNN/optimizer/matmul_program_config.mlir`. It uses a 512x1024 matmul (Mt=16), which will now be steered toward BlockSharded (2D mcast) instead of the current 1D mcast. Update the FileCheck pattern:

```
// CHECK: "ttnn.matmul"
// CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast
```

This pattern already matches both 1D and 2D mcast configs (both contain `matmul_multi_core_reuse_multi_cast` in their attribute name). Verify and adjust if needed.

- [ ] **Step 5: Commit**

```bash
git add include/ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h \
        lib/Dialect/TTNN/Analysis/OpRules/MatmulRules.cpp
git commit -m "feat: shape-aware matmul program config in MatmulRuleBook

Rewrite getOutputHints() to generate program configs directly instead
of relying on pre-baked attrs from LegalOpConfigAnalysis. Adds shape-
aware filtering: reject WidthSharded output for large Mt (prefill),
reject HeightSharded for Mt<=1 (decode). Config generation uses the
new per-type generators from MatmulProgramConfig.h."
```

---

### Task 3: Implement DRAM-sharded config generator

Fill in the `generateMatmulDRAMShardedConfig` stub from Task 1. This config is used by all tt-metal LLM models for decode matmuls (Mt==1).

**Files:**
- Modify: `lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp`

- [ ] **Step 1: Implement the generator**

The `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr` takes:
- `in0_block_w`: K tiles per block read from in0
- `per_core_m`: output M tiles per core (1 for decode)
- `per_core_n`: output N tiles per core
- `fused_activation`: optional activation

```cpp
std::optional<mlir::Attribute>
generateMatmulDRAMShardedConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                                int64_t Kt, TTNNLayoutAttr outputLayout,
                                UnaryWithParamAttr fusedActivation,
                                int64_t maxSubblockSize, bool fuseBatch) {
  // DRAM-sharded config only valid for decode pattern: Mt == 1.
  if (Mt != 1) {
    return std::nullopt;
  }

  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);
  int64_t numCores = gridX * gridY;

  int64_t perCoreM = 1;
  int64_t perCoreN = divUp(Nt, numCores);

  // in0_block_w: how many K-tiles to read per block.
  // Use largest divisor of Kt that keeps CB size reasonable.
  int64_t in0BlockW = (Kt % 2 == 0) ? 2 : 1;

  return MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
      ctx, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      /*fusedActivation=*/fusedActivation);
}
```

- [ ] **Step 2: Build and run tests**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds.

Run: `cmake --build build --target check-ttmlir 2>&1 | tail -20`
Expected: All tests pass. The DRAM-sharded config is only generated when Mt==1 AND WidthSharded output is in the legal configs, so it won't affect existing tests unless they happen to have Mt==1 matmuls with width-sharded output.

- [ ] **Step 3: Commit**

```bash
git add lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp
git commit -m "feat: add DRAM-sharded matmul program config generator

Implement generateMatmulDRAMShardedConfig for decode matmuls (Mt==1).
Emits MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
matching the config tt-metal LLM models use for decode."
```

---

### Task 4: Add isValidOutputHintForInputs() override

Prune invalid (hint, input layout) combinations before backend validation.

**Files:**
- Modify: `include/ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h`
- Modify: `lib/Dialect/TTNN/Analysis/OpRules/MatmulRules.cpp`

- [ ] **Step 1: Add declaration to MatmulRules.h**

```cpp
struct MatmulRuleBook : OpRuleBook {
  OutputHints
  getOutputHints(Operation *op,
                 const std::vector<OpConfig> &legalConfigs) const override;

  LayoutFilterFn getInputLayoutFilter() const override;

  /// Prune invalid (hint, input layout) combinations.
  bool isValidOutputHintForInputs(
      const OpConfig &hint,
      llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const override;

  void applyOpSpecificAttrs(Operation *op,
                            const BeamCandidate &candidate) const override;
};
```

- [ ] **Step 2: Implement in MatmulRules.cpp**

```cpp
bool MatmulRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint,
    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  // If no MatmulAttrs or no program config, accept (backend decides).
  if (!std::holds_alternative<MatmulAttrs>(hint.opSpecificAttrs)) {
    return true;
  }
  const auto &matmulAttrs = std::get<MatmulAttrs>(hint.opSpecificAttrs);
  if (!matmulAttrs.matmulProgramConfig.has_value()) {
    return true;
  }

  auto programConfig = matmulAttrs.matmulProgramConfig.value();

  // Need at least 2 inputs (activation + weight).
  if (inputLayouts.size() < 2) {
    return true;
  }

  auto getMemLayoutVal = [](TTNNLayoutAttr layout) -> std::optional<TensorMemoryLayout> {
    if (!layout) {
      return std::nullopt;
    }
    auto ml = layout.getMemLayout();
    if (!ml) {
      return std::nullopt;
    }
    return ml.getValue();
  };

  auto in0Mem = getMemLayoutVal(inputLayouts[0]);
  auto in1Mem = getMemLayoutVal(inputLayouts[1]);

  return llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
      .Case<MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(
          [&](auto config) {
            if (config.getMcastIn0()) {
              // mcast_in0=true: in0 should be WIDTH_SHARDED or interleaved
              // (reshard will handle conversion).
              // Reject if in0 is HEIGHT_SHARDED (wrong shard orientation).
              if (in0Mem == TensorMemoryLayout::HeightSharded) {
                return false;
              }
            } else {
              // mcast_in0=false: in0 should be HEIGHT_SHARDED or interleaved.
              // Reject if in0 is WIDTH_SHARDED.
              if (in0Mem == TensorMemoryLayout::WidthSharded) {
                return false;
              }
            }
            return true;
          })
      .Case<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          [&](auto) {
            // DRAM-sharded: in1 must be in DRAM.
            if (inputLayouts[1] &&
                inputLayouts[1].getBufferType() != BufferType::DRAM) {
              return false;
            }
            return true;
          })
      .Case<MatmulMultiCoreReuseMultiCastProgramConfigAttr>(
          [&](auto) {
            // 2D mcast: accepts BLOCK_SHARDED or INTERLEAVED in0.
            // Reject WIDTH_SHARDED in0 (wrong shard orientation for 2D).
            if (in0Mem == TensorMemoryLayout::WidthSharded) {
              return false;
            }
            return true;
          })
      .Default([](mlir::Attribute) { return true; });
}
```

- [ ] **Step 3: Build and run tests**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds.

Run: `cmake --build build --target check-ttmlir 2>&1 | tail -20`
Expected: All tests pass. The filter only rejects candidates that would have been rejected by backend validation anyway — it's an optimization, not a behavior change.

- [ ] **Step 4: Commit**

```bash
git add include/ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h \
        lib/Dialect/TTNN/Analysis/OpRules/MatmulRules.cpp
git commit -m "feat: add isValidOutputHintForInputs for matmul

Prune invalid (program config, input layout) combinations before
backend validation. Checks that input sharding orientation matches
what the program config type requires."
```

---

### Task 5: Improve 2D config heuristics for prefill

Update `generateMatmul2DConfig` to use tt-metal prefill patterns: `in0_block_w=1` for large Mt, `fuseBatch=false` for very large Mt.

**Files:**
- Modify: `lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp`

- [ ] **Step 1: Update generateMatmul2DConfig**

```cpp
std::optional<mlir::Attribute>
generateMatmul2DConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt, int64_t Kt,
                       TTNNLayoutAttr outputLayout,
                       UnaryWithParamAttr fusedActivation,
                       int64_t maxSubblockSize, bool fuseBatch) {
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);

  int64_t perCoreM = divUp(Mt, gridY);
  int64_t perCoreN = divUp(Nt, gridX);

  // For large Mt (prefill): stream K from DRAM one tile at a time.
  // Matches tt-metal prefill pattern. Minimizes L1 pressure since
  // activation is typically DRAM interleaved for prefill.
  // For small Mt: use larger in0_block_w for better tile reuse.
  int64_t in0BlockW;
  if (Mt > kPrefillMtThreshold) {
    in0BlockW = 1;
  } else {
    in0BlockW = (Kt % 2 == 0) ? 2 : 1;
  }

  // fuseBatch=false for very large sequences.
  // Matches tt-metal threshold (seq_len > 2048, i.e., Mt > 64).
  if (Mt > 64) {
    fuseBatch = false;
  }

  int64_t outSubblockH = 1;
  int64_t outSubblockW = largestDivisorUpTo(perCoreN, maxSubblockSize);
  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);

  return MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      /*transpose_mcast=*/false, /*fusedActivation=*/fusedActivation,
      fuseBatch);
}
```

Note: `kPrefillMtThreshold` is defined in MatmulRules.cpp (Task 2). For `generateMatmul2DConfig` to use it, either:
- Define it in MatmulProgramConfig.cpp too (duplicate but simple), or
- Move it to a shared header.

Simplest: define a local constant in MatmulProgramConfig.cpp:
```cpp
static constexpr int64_t kPrefillMtThreshold = 4;
```

- [ ] **Step 2: Build and run tests**

Run: `cmake --build build 2>&1 | tail -5`
Expected: Build succeeds.

Run: `cmake --build build --target check-ttmlir 2>&1 | tail -20`
Expected: Tests pass. The 2D config changes only affect BlockSharded output, which is now preferred for large Mt. Check that any tests with large matmuls and block-sharded output reflect updated `in0_block_w` and `fuse_batch` values.

- [ ] **Step 3: Commit**

```bash
git add lib/Dialect/TTNN/Analysis/MatmulProgramConfig.cpp
git commit -m "feat: improve 2D matmul config for prefill

Set in0_block_w=1 for large Mt (stream from DRAM, matching tt-metal
prefill). Set fuseBatch=false for Mt>64 (matching tt-metal seq_len>2048
threshold)."
```

---

### Task 6: End-to-end verification

Verify the full optimizer pipeline produces expected configs for different matmul sizes.

**Files:**
- Modify: `test/ttmlir/Dialect/TTNN/optimizer/matmul_program_config.mlir`

- [ ] **Step 1: Run the existing optimizer test and check output**

Run: `source env/activate && llvm-lit test/ttmlir/Dialect/TTNN/optimizer/matmul_program_config.mlir -v 2>&1 | tail -20`

Check whether it passes or fails. If it fails, the FileCheck pattern needs updating to match the new config type (512x1024 matmul has Mt=16 > 4, so it gets BlockSharded 2D instead of 1D).

- [ ] **Step 2: Update the test FileCheck if needed**

If the test fails because the optimizer now emits a 2D mcast config instead of 1D, update the check pattern. The existing pattern `matmul_multi_core_reuse_multi_cast` matches both. Verify the specific params match.

- [ ] **Step 3: Run full test suite**

Run: `cmake --build build --target check-ttmlir 2>&1 | tail -30`
Expected: All tests pass.

- [ ] **Step 4: Manual inspection on a prefill model (if available)**

If the llama 70b 2-layer prefill MLIR is available:

Run: `source env/activate && ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" llama_70b_2lyr_prefill_compiled.mlir -o /dev/null 2>&1 | grep -c "block_sharded"`

Verify that large-M matmuls get block-sharded output with 2D mcast configs.

- [ ] **Step 5: Commit any test updates**

```bash
git add test/ttmlir/Dialect/TTNN/optimizer/matmul_program_config.mlir
git commit -m "test: update matmul program config test for shape-aware sharding

Large-Mt matmul now prefers BlockSharded (2D mcast) over WidthSharded
(1D mcast). Update FileCheck to match."
```
