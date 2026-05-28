// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
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

        // Shape-specific divisibility filter. The empirical rules:
        //   c_in_block | (kT * kH * kW * c_in_aligned)
        //   t/h/w_out_block | T/H/W_out (only for currently-set fields)
        //   h_out_block * w_out_block <= 256
        const auto kernelSize = convOp.getKernelSize();
        const int64_t kT = kernelSize[0];
        const int64_t kH = kernelSize[1];
        const int64_t kW = kernelSize[2];
        constexpr int64_t TILE_WIDTH =
            mlir::tt::ttcore::TileType::getDefaultShape()[1];
        // Use op attribute (not the weight tensor) because by this point the
        // weight has already been collapsed to 2D by PrepareConv3dWeightsOp,
        // so dim 1 of the weight tensor is out_channels, not c_in_per_group.
        const int64_t cInPerGroup = convOp.getInChannels() / convOp.getGroups();
        const int64_t cInAligned =
            llvm::divideCeil(cInPerGroup, TILE_WIDTH) * TILE_WIDTH;
        const auto outputShape = convOp.getResult().getType().getShape();
        // Conv3dOp output is NDHWC (channel last). Drop the batch dim and
        // channel dim; T, H, W are dims 1, 2, 3.
        const int64_t tOut = outputShape[1];
        const int64_t hOut = outputShape[2];
        const int64_t wOut = outputShape[3];
        const int64_t cOut = outputShape[4];
        const int64_t cOutAligned =
            llvm::divideCeil(cOut, TILE_WIDTH) * TILE_WIDTH;

        auto filterOut = [=](const Conv3dConfigAttr &cfg) {
          if (auto cIn = cfg.getCInBlock()) {
            if (cIn.value() == 0 || cIn.value() > cInAligned ||
                (kT * kH * kW * cInAligned) % cIn.value() != 0) {
              return true;
            }
          }
          if (auto cOutBlock = cfg.getCOutBlock()) {
            // c_out_block must be tile-aligned and not exceed c_out_aligned.
            if (cOutBlock.value() == 0 || cOutBlock.value() > cOutAligned ||
                cOutAligned % cOutBlock.value() != 0) {
              return true;
            }
          }
          if (auto t = cfg.getTOutBlock()) {
            if (t.value() == 0 || tOut % t.value() != 0) {
              return true;
            }
          }
          if (auto h = cfg.getHOutBlock()) {
            if (h.value() == 0 || hOut % h.value() != 0) {
              return true;
            }
          }
          if (auto w = cfg.getWOutBlock()) {
            if (w.value() == 0 || wOut % w.value() != 0) {
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

        // Empirical scoring rules derived from tt-metal's
        // `bruteforce_conv3d_sweep.py` and verified against the hand-tuned
        // `_BLOCKINGS` oracle table (14 of 15 production entries use
        // h·w == 32):
        //   1. Prefer h_out_block * w_out_block close to 32 — the matmul
        //      aspect-ratio sweet spot for the tt-metal Conv3d kernel.
        //   2. Larger t_out_block — more temporal voxels per launch.
        //   3. Larger c_in_block — fewer C-in passes per output voxel.
        //   4. Larger c_out_block — fewer C-out passes.
        //   5. For T*H*W > 10_000 prefer c_out >= 96; small c_out * large
        //      spatial never wins.
        const int64_t spatialVolume = tOut * hOut * wOut;
        auto score = [&](const Conv3dConfigAttr &c) {
          uint32_t t = c.getTOutBlock().value_or(1);
          uint32_t h = c.getHOutBlock().value_or(1);
          uint32_t w = c.getWOutBlock().value_or(1);
          uint32_t cIn = c.getCInBlock().value_or(1);
          uint32_t cOut = c.getCOutBlock().value_or(1);
          int64_t hwAspect = std::abs(static_cast<int64_t>(h) * w - 32);
          int64_t largeSpatialBonus =
              (spatialVolume > 10000 && cOut >= 96) ? 1000 : 0;
          return std::make_tuple(-hwAspect, static_cast<int64_t>(t),
                                 static_cast<int64_t>(cIn),
                                 static_cast<int64_t>(cOut), largeSpatialBonus);
        };
        llvm::stable_sort(generatedConfigs, [&](const Conv3dConfigAttr &a,
                                                const Conv3dConfigAttr &b) {
          return score(a) > score(b);
        });

        std::vector<OpConfig> newLegalConfigs;
        for (Conv3dConfigAttr configAttr : generatedConfigs) {
          for (const OpConfig &existingOpConfig : analysisResult) {
            newLegalConfigs.emplace_back(existingOpConfig.outputLayout,
                                         Conv3dAttrs{configAttr, std::nullopt});
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
