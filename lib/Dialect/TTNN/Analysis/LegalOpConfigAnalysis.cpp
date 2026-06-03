// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <vector>

namespace mlir::tt::ttnn {

static bool isOpEnabledForAnalysis(Operation *op) {
  // Enable only for specific ops.
  if (llvm::isa<ttnn::Conv2dOp, ttnn::Conv3dOp, ttnn::ConvTranspose2dOp,
                ttnn::MatmulOp, ttnn::LinearOp>(op)) {
    return true;
  }

  return false;
}

template <typename ConvOpT>
static void
applyConv2dConfigOverrides(ConvOpT op,
                           const Conv2dConfigOverrideParams &overrides,
                           std::vector<OpConfig> &analysisResult) {
  // Apply conv2d config overrides to all legal (layout) configurations of
  // current op.

  // If conv2d config is not set get default conv2d config.
  Conv2dConfigAttr conv2dConfigAttr = op.getConv2dConfigAttr();
  if (!conv2dConfigAttr) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "Conv2d config not set, using empty default config");
    conv2dConfigAttr = Conv2dConfigAttr::get(op.getContext());
  }
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Conv2d config before overrides: {}", conv2dConfigAttr);
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Overrides: {}", overrides);

  if (overrides.weightsDtype.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withWeightsDtype(*overrides.weightsDtype);
  }

  if (overrides.activation.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withActivation(*overrides.activation);
  }

  if (overrides.deallocateActivation.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withDeallocateActivation(
        *overrides.deallocateActivation);
  }

  if (overrides.reallocateHaloOutput.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withReallocateHaloOutput(
        *overrides.reallocateHaloOutput);
  }

  if (overrides.actBlockHOverride.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withActBlockHOverride(*overrides.actBlockHOverride);
  }

  if (overrides.actBlockWDiv.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withActBlockWDiv(*overrides.actBlockWDiv);
  }

  if (overrides.reshardIfNotOptimal.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withReshardIfNotOptimal(
        *overrides.reshardIfNotOptimal);
  }

  if (overrides.overrideShardingConfig.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withOverrideShardingConfig(
        *overrides.overrideShardingConfig);
  }

  if (overrides.shardLayout.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withShardLayout(*overrides.shardLayout);
  }

  if (overrides.coreGrid.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withCoreGrid(*overrides.coreGrid);
  }

  if (overrides.transposeShards.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withTransposeShards(*overrides.transposeShards);
  }

  if (overrides.outputLayout.has_value()) {
    conv2dConfigAttr =
        conv2dConfigAttr.withOutputLayout(*overrides.outputLayout);
  }

  if (overrides.enableActDoubleBuffer.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withEnableActDoubleBuffer(
        *overrides.enableActDoubleBuffer);
  }

  if (overrides.enableWeightsDoubleBuffer.has_value()) {
    conv2dConfigAttr = conv2dConfigAttr.withEnableWeightsDoubleBuffer(
        *overrides.enableWeightsDoubleBuffer);
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Conv2d config after overrides: {}", conv2dConfigAttr);

  // Set overridden conv2d config for all OpConfigs.
  for (OpConfig &opConfig : analysisResult) {
    assert(opConfig.isAttrUninitialized() &&
           "OpConfig should not have a config set before applying overrides");
    opConfig.opSpecificAttrs = Conv2dAttrs{conv2dConfigAttr, std::nullopt};
  }
}

static void
applyConv3dConfigOverrides(ttnn::Conv3dOp op,
                           const Conv3dConfigOverrideParams &overrides,
                           std::vector<OpConfig> &analysisResult) {
  // Build a Conv3dConfigAttr from the op's current config plus overrides.
  Conv3dConfigAttr base = op.getConv3dConfigAttr();

  auto pick = [](std::optional<uint32_t> override,
                 std::optional<uint32_t> existing) -> std::optional<uint32_t> {
    return override.has_value() ? override : existing;
  };

  std::optional<ttcore::DataType> weightsDtype = overrides.weightsDtype;
  std::optional<uint32_t> tOutBlock;
  std::optional<uint32_t> wOutBlock;
  std::optional<uint32_t> hOutBlock;
  std::optional<uint32_t> cOutBlock;
  std::optional<uint32_t> cInBlock;
  std::optional<ttcore::GridAttr> gridSize;
  if (base) {
    if (!weightsDtype.has_value()) {
      weightsDtype = base.getWeightsDtype();
    }
    tOutBlock = pick(overrides.tOutBlock, base.getTOutBlock());
    wOutBlock = pick(overrides.wOutBlock, base.getWOutBlock());
    hOutBlock = pick(overrides.hOutBlock, base.getHOutBlock());
    cOutBlock = pick(overrides.cOutBlock, base.getCOutBlock());
    cInBlock = pick(overrides.cInBlock, base.getCInBlock());
    gridSize = base.getComputeWithStorageGridSize();
  } else {
    tOutBlock = overrides.tOutBlock;
    wOutBlock = overrides.wOutBlock;
    hOutBlock = overrides.hOutBlock;
    cOutBlock = overrides.cOutBlock;
    cInBlock = overrides.cInBlock;
  }

  auto built =
      Conv3dConfigAttr::get(op.getContext(), weightsDtype, tOutBlock, wOutBlock,
                            hOutBlock, cOutBlock, cInBlock, gridSize);

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "Conv3d config after overrides: {}", built);

  for (OpConfig &opConfig : analysisResult) {
    assert(opConfig.isAttrUninitialized() &&
           "OpConfig should not have a config set before applying overrides");
    opConfig.opSpecificAttrs = Conv3dAttrs{built, std::nullopt};
  }
}

bool LegalOpConfigAnalysis::applyOverrides() {
  // For now, easiest way to initialize analysisResult is to copy the legal
  // configs here. Proper solution is that init() method is overridden in child
  // classes.
  analysisResult = analysisInput.legalConfigs;

  if (!isOpEnabledForAnalysis(op)) {
    return true;
  }

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&](auto convOp) {
        if (!analysisInput.conv2dConfigOverrides) {
          return false;
        }
        Conv2dConfigOverrideParams conv2dConfigOverrides;
        if (!isa<NameLoc>(op->getLoc())) {
          return false;
        }
        StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
        auto overrideConv2dIt =
            analysisInput.conv2dConfigOverrides->find(opLocName);
        if (overrideConv2dIt != analysisInput.conv2dConfigOverrides->end()) {
          conv2dConfigOverrides = overrideConv2dIt->getValue();
        }
        applyConv2dConfigOverrides(convOp, conv2dConfigOverrides,
                                   analysisResult);

        // Conv2d config overrides were applied, return true if all config
        // parameters were overridden, therefore no need to search for legal
        // configs.
        return conv2dConfigOverrides.fullConfigOverride();
      })
      .Case<ttnn::MatmulOp, ttnn::LinearOp>([](auto) {
        // Matmul/Linear ops don't use conv2d config overrides.
        return false;
      })
      .Case<ttnn::Conv3dOp>([&](ttnn::Conv3dOp convOp) {
        if (!analysisInput.conv3dConfigOverrides ||
            !isa<NameLoc>(op->getLoc())) {
          return false;
        }
        StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
        auto it = analysisInput.conv3dConfigOverrides->find(opLocName);
        if (it == analysisInput.conv3dConfigOverrides->end()) {
          return false;
        }
        const Conv3dConfigOverrideParams &overrides = it->getValue();
        applyConv3dConfigOverrides(convOp, overrides, analysisResult);
        // If every searchable field is pinned, skip the search.
        return overrides.fullConfigOverride();
      })
      .Default([](Operation *op) {
        llvm::llvm_unreachable_internal("Unsupported op type");
        return false;
      });
}

void LegalOpConfigAnalysis::fillOpSpecificAttrs() {
  llvm::TypeSwitch<Operation *>(op)
      .Case<ttnn::Conv2dOp, ttnn::ConvTranspose2dOp>([&](auto convOp) {
        assert(!analysisResult.empty() &&
               "Analysis result should not be empty after applying overrides");
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Filling op specific attrs for conv2d op {}, starting "
                     "with {} configs",
                     convOp, analysisResult.size());

        // It's possible that no conv2d config was applied in
        // applyConv2dConfigOverrides (e.g. when op does not have loc assigned)
        // so base config is empty.
        Conv2dConfigAttr conv2dConfigAttrBase =
            analysisResult.begin()->isAttrUninitialized()
                ? (convOp.getConv2dConfigAttr()
                       ? convOp.getConv2dConfigAttr()
                       : Conv2dConfigAttr::get(op->getContext()))
                : std::get<Conv2dAttrs>(analysisResult.begin()->opSpecificAttrs)
                      .conv2dConfig.value();

        // If weights dtype is not set, set it to the weight tensor dtype.
        if (!conv2dConfigAttrBase.getWeightsDtype().has_value()) {
          conv2dConfigAttrBase = conv2dConfigAttrBase.withWeightsDtype(
              ttcore::elementTypeToDataType(
                  convOp.getWeight().getType().getElementType()));
        }

        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Op {} Base conv2d config: {}", convOp.getLoc(),
                     conv2dConfigAttrBase);

        auto filterOut = [](const Conv2dConfigAttr &config) {
          //
          // Combinations that are invalid:
          // 1. reshard_if_not_optimal = true and shard_layout is not set.
          //
          return (config.hasReshardIfNotOptimal() &&
                  config.getReshardIfNotOptimal().getValue() &&
                  !config.hasShardLayout());
        };

        Conv2dConfigGenerator configGenerator(&convOp, conv2dConfigAttrBase,
                                              searchSpace, filterOut);

        std::vector<OpConfig> newLegalConfigs;
        auto addConfigs = [&](const Conv2dConfigAttr &configAttr) {
          for (const OpConfig &existingOpConfig : analysisResult) {
            // Create a new OpConfig pairing the existing layout with the new
            // conv config.
            newLegalConfigs.emplace_back(existingOpConfig.outputLayout,
                                         Conv2dAttrs{configAttr, std::nullopt});
          }
        };

        if (configGenerator.searchDone()) {
          // If search is done before any configs are generated, we will just
          // put base config in all possible layouts. This way we are ensuring
          // dtype and weights_dtype will be set.
          addConfigs(conv2dConfigAttrBase);
        } else {
          // Otherwise, generate all possible configs and add them to the
          // result.
          while (Conv2dConfigAttr configAttr =
                     configGenerator.getNextConfig()) {
            addConfigs(configAttr);
          }
        }

        analysisResult = std::move(newLegalConfigs);
        TTMLIR_TRACE(
            ttmlir::LogComponent::Optimizer,
            "Filled op specific attrs for conv2d op {}, ending with {} configs",
            convOp, analysisResult.size());
      })
      .Case<ttnn::Conv3dOp>([&](auto convOp) {
        assert(!analysisResult.empty() &&
               "Analysis result should not be empty after applying overrides");
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "Filling op specific attrs for conv3d op {}, starting "
                     "with {} configs",
                     convOp, analysisResult.size());

        // If applyConv3dConfigOverrides ran, the analysisResult's first
        // entry already carries the override config — use that as the base
        // so override-pinned fields flow through to the generator (which
        // skips fields with values already set on the base config).
        Conv3dConfigAttr baseConfig;
        if (!analysisResult.begin()->isAttrUninitialized() &&
            std::holds_alternative<Conv3dAttrs>(
                analysisResult.begin()->opSpecificAttrs)) {
          baseConfig =
              std::get<Conv3dAttrs>(analysisResult.begin()->opSpecificAttrs)
                  .conv3dConfig.value_or(nullptr);
        }
        if (!baseConfig) {
          baseConfig =
              convOp.getConv3dConfigAttr()
                  ? convOp.getConv3dConfigAttr()
                  : Conv3dConfigAttr::get(
                        op->getContext(),
                        /*weights_dtype=*/std::nullopt,
                        /*t_out_block=*/std::nullopt,
                        /*w_out_block=*/std::nullopt,
                        /*h_out_block=*/std::nullopt,
                        /*c_out_block=*/std::nullopt,
                        /*c_in_block=*/std::nullopt,
                        /*compute_with_storage_grid_size=*/std::nullopt);
        }

        // Shape-derived bounds used by the structural pre-filter below.
        // The kernel sizes and aligned channel counts come from op
        // attributes (the weight tensor's in-IR shape is raw 5D at this
        // point — TTNNPrepareConv3dWeights hasn't run yet — so dim 1 is
        // not yet a flattened patch dimension).
        constexpr int64_t TILE_WIDTH =
            mlir::tt::ttcore::TileType::getDefaultShape()[1];
        const int64_t cInPerGroup = convOp.getInChannels() / convOp.getGroups();
        const int64_t cInAligned =
            llvm::divideCeil(cInPerGroup, TILE_WIDTH) * TILE_WIDTH;
        const auto outputShape = convOp.getResult().getType().getShape();
        // Conv3dOp output is NDHWC (channel last). T, H, W are dims 1, 2, 3.
        const int64_t tOut = outputShape[1];
        const int64_t hOut = outputShape[2];
        const int64_t wOut = outputShape[3];
        const int64_t cOut = outputShape[4];
        const int64_t cOutAligned =
            llvm::divideCeil(cOut, TILE_WIDTH) * TILE_WIDTH;
        const auto kernelSize = convOp.getKernelSize();
        const int64_t kT = kernelSize[0];
        const int64_t kH = kernelSize[1];
        const int64_t kW = kernelSize[2];

        // Structural pre-filter. Rejects only candidates that are
        // structurally invalid for the runtime kernel:
        //   - c_in_block must divide kT*kH*kW*cInAligned (weight
        //     pre-pack constraint) and not exceed cInAligned.
        //   - c_out_block must divide cOutAligned and not exceed it.
        //   - t/h/w blocks must be positive and not exceed the
        //     corresponding output extent (tt-metal pads non-divisor
        //     trailing blocks at runtime).
        //   - h_out_block * w_out_block <= 256 (CB capacity limit
        //     baked into the kernel).
        // L1-fit is intentionally *not* checked here — OpModel is the
        // authoritative legality oracle and the validation pass below
        // queries it directly.
        auto filterOut = [=](const Conv3dConfigAttr &cfg) {
          if (auto cIn = cfg.getCInBlock()) {
            if (cIn.value() == 0 || cIn.value() > cInAligned ||
                (kT * kH * kW * cInAligned) % cIn.value() != 0) {
              return true;
            }
          }
          if (auto cOutBlock = cfg.getCOutBlock()) {
            if (cOutBlock.value() == 0 || cOutBlock.value() > cOutAligned ||
                cOutAligned % cOutBlock.value() != 0) {
              return true;
            }
          }
          if (auto t = cfg.getTOutBlock()) {
            if (t.value() == 0 || static_cast<int64_t>(t.value()) > tOut) {
              return true;
            }
          }
          if (auto h = cfg.getHOutBlock()) {
            if (h.value() == 0 || static_cast<int64_t>(h.value()) > hOut) {
              return true;
            }
          }
          if (auto w = cfg.getWOutBlock()) {
            if (w.value() == 0 || static_cast<int64_t>(w.value()) > wOut) {
              return true;
            }
          }
          if (cfg.getHOutBlock() && cfg.getWOutBlock()) {
            if (cfg.getHOutBlock().value() * cfg.getWOutBlock().value() > 256) {
              return true;
            }
          }
          return false;
        };

        // Collect generated configs into a vector first so we can rank them
        // empirically before crossing with output layouts. The non-greedy
        // OpConfigAnalysis (lib/Dialect/TTNN/Analysis/OpConfigAnalysis.cpp:35)
        // picks `legalConfigs[0]`, so config order determines what the
        // optimizer attaches in the non-greedy path. Greedy beam search will
        // additionally apply Conv3dRuleBook tiebreakers downstream.
        llvm::SmallVector<Conv3dConfigAttr> generatedConfigs;
        bool searched = forEachConv3dConfig(
            &convOp, baseConfig, conv3dSearchSpace, filterOut,
            [&](Conv3dConfigAttr cfg) { generatedConfigs.push_back(cfg); });
        if (!searched) {
          generatedConfigs.push_back(baseConfig);
        }

        // Two-stage ranking: a first-principles structural score
        // selects the top-K candidates for OpModel evaluation, then
        // OpModel provides authoritative legality + cycle-based ordering.
        //
        // Hand-tuned per-workload blocking oracles are intentionally not
        // consulted here. Their role is verification (does the runtime
        // of an optimizer-picked config land close to the oracle's
        // hand-tuned config?), not search input.
        //
        // Structural score (larger is better):
        //   1. -|h*w - kAspectTarget| where kAspectTarget is the device
        //      compute grid's tile count (8x8 = 64 cores, processing
        //      one 32x32 tile each → kAspectTarget output positions per
        //      launch saturates the grid on conv3d's vol2col+matmul M
        //      dimension). Aspect-saturating candidates rank first
        //      because under- or over-saturating wastes cores or
        //      inflates per-launch CB pressure past what L1 can hold;
        //      OpModel's constraint check is sometimes optimistic about
        //      L1 vs the full validation pipeline, so the pre-rank
        //      should not bet on extreme block sizes.
        //   2. blockVolume = t * h * w * cInBlock * cOutBlock
        //      Among aspect-saturating candidates, bigger blocks
        //      amortize kernel-launch overhead.
        //   3. -max(h, w)
        //      Aspect-tie breaker preferring balanced (8,4) over (16,2).
        //   4. Canonical (t, h, w, cIn, cOut) tail for determinism.
        constexpr int64_t kAspectTarget = 32;
        auto structuralScore = [&](const Conv3dConfigAttr &c) {
          uint32_t t = c.getTOutBlock().value_or(1);
          uint32_t h = c.getHOutBlock().value_or(1);
          uint32_t w = c.getWOutBlock().value_or(1);
          uint32_t cIn = c.getCInBlock().value_or(1);
          uint32_t cOut = c.getCOutBlock().value_or(1);
          int64_t aspectDist =
              std::abs(static_cast<int64_t>(h) * w - kAspectTarget);
          int64_t blockVolume = int64_t{t} * h * w * cIn * cOut;
          int64_t aspectBalance = -static_cast<int64_t>(std::max(h, w));
          return std::make_tuple(
              -aspectDist, blockVolume, aspectBalance, static_cast<int64_t>(t),
              static_cast<int64_t>(h), static_cast<int64_t>(w),
              static_cast<int64_t>(cIn), static_cast<int64_t>(cOut));
        };
        llvm::stable_sort(generatedConfigs, [&](const Conv3dConfigAttr &a,
                                                const Conv3dConfigAttr &b) {
          return structuralScore(a) > structuralScore(b);
        });

        // Default compute config attached to every emitted Conv3dOp. At
        // optimization-level=1 the TTNNSetComputeKernelConfig pass is
        // skipped (it defers to the optimizer), but Conv3dOp's runtime
        // kernel rejects IR without compute_config.
        auto defaultComputeCfg =
            DeviceComputeKernelConfigAttr::get(op->getContext())
                .withMathFidelity(MathFidelity::HiFi4)
                .withFp32DestAccEn(true);

        // OpModel-driven validation has two parts:
        //
        //   Legality: `getOpConstraints` runs a tt-metal graph query
        //     that reports whether a config is runnable (CB allocation,
        //     L1 fit, kernel preconditions). Works in mock-device mode,
        //     so this gate is always meaningful.
        //
        //   Cycle-based ranking: `getOpRuntime` runs a kernel
        //     simulation that returns a cycle estimate. Available only
        //     on real silicon; mock devices return
        //     "getOpRuntime is not supported in mock device mode". When
        //     unavailable, candidates are kept in structural order.
        //
        // Strategy: walk structurally-ranked candidates in batches of
        // kOpModelValidationBatch, querying constraints. Stop expanding
        // once at least one candidate is legality-validated. If every
        // candidate the search space produces is rejected by
        // constraints, the op is unrunnable on this device and we emit
        // a compile error rather than a silently broken flatbuffer.
        // Then, for the legality-validated set, try `getOpRuntime` for
        // cycle ranking; if it errors out (mock device), keep
        // structural order.
        constexpr size_t kOpModelValidationBatch = 64;
        const TTNNLayoutAttr representativeOutputLayout =
            analysisResult.begin()->outputLayout;

        // Probe layouts are read directly from the in-IR operand types.
        // Conv3dOp is in `enabledOpsForWorkaroundWithOptimizer`, so by
        // the time this analysis runs the workaround pass has already
        // pinned the input to RowMajor BF16 and the bias to Tile BF16;
        // the weight is whatever the optimizer chose (it gets prepared
        // post-optimizer by TTNNPrepareConv3dWeights, which inserts a
        // to_layout(Tile) before the kernel).
        std::vector<TTNNLayoutAttr> probeInputs;
        probeInputs.push_back(mlir::cast<TTNNLayoutAttr>(
            convOp.getInput().getType().getEncoding()));
        probeInputs.push_back(mlir::cast<TTNNLayoutAttr>(
            convOp.getWeight().getType().getEncoding()));
        if (convOp.getBias()) {
          probeInputs.push_back(mlir::cast<TTNNLayoutAttr>(
              convOp.getBias().getType().getEncoding()));
        }

        llvm::SmallVector<Conv3dConfigAttr> opModelLegal;
        size_t opModelExamined = 0;
        size_t opModelRejected = 0;
        std::string firstError;
        while (opModelLegal.empty() &&
               opModelExamined < generatedConfigs.size()) {
          size_t batchEnd = std::min(opModelExamined + kOpModelValidationBatch,
                                     generatedConfigs.size());
          for (size_t i = opModelExamined; i < batchEnd; ++i) {
            OpConfig probeConfig(
                representativeOutputLayout,
                Conv3dAttrs{generatedConfigs[i], defaultComputeCfg});
            llvm::Expected<op_model::OpConstraints> constraints =
                convOp.getOpConstraints(probeInputs, probeConfig);
            if (!constraints) {
              ++opModelRejected;
              if (firstError.empty()) {
                firstError = llvm::toString(constraints.takeError());
              } else {
                llvm::consumeError(constraints.takeError());
              }
              continue;
            }
            opModelLegal.push_back(generatedConfigs[i]);
          }
          opModelExamined = batchEnd;
        }

        if (opModelLegal.empty()) {
          // Every candidate the search space could produce was rejected
          // by OpModel's legality check with the runtime-contract
          // layouts. The op cannot run on this device under any
          // blocking we can express.
          convOp.emitError()
              << "Conv3dOp has no OpModel-runnable configuration: "
              << opModelRejected
              << " structurally-valid candidates were all rejected by "
                 "the tt-metal kernel simulation. First rejection: "
              << firstError
              << ". Consider widening Conv3dConfigSearchSpaceFactory's "
                 "caps or reducing the op's shape.";
          analysisResult.clear();
          return;
        }

        // Try cycle-based ranking on the legality-validated set. The
        // first runtime query doubles as a feature probe — if it returns
        // an error, the device cannot supply cycle estimates (mock
        // mode), so we keep structural order. Otherwise we collect
        // cycles for every legal candidate and sort ascending.
        llvm::SmallVector<Conv3dConfigAttr> reordered;
        reordered.reserve(generatedConfigs.size());
        {
          OpConfig firstProbe(
              representativeOutputLayout,
              Conv3dAttrs{opModelLegal.front(), defaultComputeCfg});
          llvm::Expected<size_t> firstRuntime =
              convOp.getOpRuntime(probeInputs, firstProbe);
          if (firstRuntime) {
            llvm::SmallVector<std::pair<Conv3dConfigAttr, size_t>>
                opModelRanked;
            opModelRanked.reserve(opModelLegal.size());
            opModelRanked.emplace_back(opModelLegal.front(),
                                       firstRuntime.get());
            for (size_t i = 1; i < opModelLegal.size(); ++i) {
              OpConfig probe(representativeOutputLayout,
                             Conv3dAttrs{opModelLegal[i], defaultComputeCfg});
              llvm::Expected<size_t> runtime =
                  convOp.getOpRuntime(probeInputs, probe);
              if (!runtime) {
                // Stale failure on a config that just passed
                // constraints. Keep it but order it after the
                // cycle-known ones via std::numeric_limits::max().
                llvm::consumeError(runtime.takeError());
                opModelRanked.emplace_back(opModelLegal[i],
                                           std::numeric_limits<size_t>::max());
                continue;
              }
              opModelRanked.emplace_back(opModelLegal[i], runtime.get());
            }
            llvm::stable_sort(opModelRanked,
                              [](const std::pair<Conv3dConfigAttr, size_t> &a,
                                 const std::pair<Conv3dConfigAttr, size_t> &b) {
                                return a.second < b.second;
                              });
            for (const auto &entry : opModelRanked) {
              reordered.push_back(entry.first);
            }
          } else {
            // Runtime estimates unavailable (typically mock device).
            // Keep structural order of legality-passing candidates.
            llvm::consumeError(firstRuntime.takeError());
            for (Conv3dConfigAttr cfg : opModelLegal) {
              reordered.push_back(cfg);
            }
          }
        }
        // Append the post-examined-batch tail (untested by OpModel) in
        // structural order so the beam search has fallback candidates.
        for (size_t i = opModelExamined; i < generatedConfigs.size(); ++i) {
          reordered.push_back(generatedConfigs[i]);
        }
        generatedConfigs = std::move(reordered);

        std::vector<OpConfig> newLegalConfigs;
        for (Conv3dConfigAttr configAttr : generatedConfigs) {
          for (const OpConfig &existingOpConfig : analysisResult) {
            newLegalConfigs.emplace_back(
                existingOpConfig.outputLayout,
                Conv3dAttrs{configAttr, defaultComputeCfg});
          }
        }

        analysisResult = std::move(newLegalConfigs);
        TTMLIR_TRACE(
            ttmlir::LogComponent::Optimizer,
            "Filled op specific attrs for conv3d op {}, ending with {} configs",
            convOp, analysisResult.size());
      })
      .Case<ttnn::MatmulOp, ttnn::LinearOp>([&](auto matmulOp) {
        // Generate matmul program config for each output layout.
        for (OpConfig &opConfig : analysisResult) {
          auto programConfig =
              generateMatmulProgramConfig(op, opConfig.outputLayout);
          if (programConfig.has_value()) {
            opConfig.opSpecificAttrs =
                MatmulAttrs{programConfig.value(), matmulOp.getComputeConfig()};
          }
          TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                       "Filled op specific attrs for matmul/linear op {}, "
                       "\n\t op config: {}",
                       ttmlir::opToString(matmulOp), opConfig);
        }
      })
      .Default([](Operation *op) -> void {
        op->emitError("Unsupported op type");
        llvm::llvm_unreachable_internal("Unsupported op type");
      });
}

void LegalOpConfigAnalysis::analysisImplementation() {
  if (!isOpEnabledForAnalysis(op)) {
    return;
  }

  fillOpSpecificAttrs();

  if (analysisResult.empty()) {
    op->emitError("No legal config found for the operation");
    llvm::llvm_unreachable_internal("No legal config found for the operation");
  }
}

} // namespace mlir::tt::ttnn
