// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPCONFIGANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPCONFIGANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/Conv2dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"
#include <vector>

namespace mlir::tt::ttnn {

struct LegalOpConfigAnalysisInput {
  // Legal configs found by LegalOpLayoutAnalysis.
  std::vector<OpConfig> legalConfigs;

  // Conv2d config overrides.
  llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides;

  // Conv3d config overrides.
  llvm::StringMap<Conv3dConfigOverrideParams> *conv3dConfigOverrides;

  LegalOpConfigAnalysisInput()
      : conv2dConfigOverrides(nullptr), conv3dConfigOverrides(nullptr) {}

  LegalOpConfigAnalysisInput(
      std::vector<OpConfig> legalConfigs,
      llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides,
      llvm::StringMap<Conv3dConfigOverrideParams> *conv3dConfigOverrides =
          nullptr)
      : legalConfigs(legalConfigs),
        conv2dConfigOverrides(conv2dConfigOverrides),
        conv3dConfigOverrides(conv3dConfigOverrides) {}

  bool operator==(const LegalOpConfigAnalysisInput &rhs) const {
    return legalConfigs == rhs.legalConfigs &&
           conv2dConfigOverrides == rhs.conv2dConfigOverrides &&
           conv3dConfigOverrides == rhs.conv3dConfigOverrides;
  }

  bool operator!=(const LegalOpConfigAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct Conv2dConfigSearchSpaceFactory {
  static Conv2dConfigSearchSpace get() {
    static Conv2dConfigSearchSpace searchSpace;

    // Return empty search space for now.
    // TODO(rpavlovicTT): Enable search space for conv2d configs when priority
    // is set.

    // 0 is best (allows max ntiles based on input). Must be multiple of 32.
    // Between non-zero values, prefer larger (less restrictive).
    // Ordered by preference: 0 (best), 64, 32.
    searchSpace.actBlockHOverride = {0, 64, 32};

    searchSpace.deallocateActivation = {true};

    // searchSpace.reshardIfNotOptimal = {false, true};

    return searchSpace;
  }
};

// Default shape-independent Conv3d search space.
//
// Candidate sets are produced by structural generators. Each generator
// emits every value consistent with the structural constraints of its
// field; LegalOpConfigAnalysis then applies a shape-specific structural
// filter (divisibility, bounds, h*w <= 256) and OpModel-driven
// validation that rejects candidates the tt-metal conv3d kernel cannot
// run on this device. Per-workload blocking oracles are not consulted
// here — their role is downstream verification, not search input.
struct Conv3dConfigSearchSpaceFactory {
  static Conv3dConfigSearchSpace get() {
    static const Conv3dConfigSearchSpace searchSpace = [] {
      Conv3dConfigSearchSpace s;

      // c_in_block / c_out_block: every multiple of TILE_WIDTH up to
      // kMaxChannelBlock.
      //
      // tt-metal's conv3d kernel processes channels in TILE_WIDTH (32)
      // groups, and PrepareConv3dWeightsOp pre-packs the weight using
      // `alignment = TILE_WIDTH` (see TTNNPrepareConv3dWeights). The
      // structural filter additionally enforces
      // `c_in_block <= cInAligned`,
      // `(kT*kH*kW*cInAligned) % c_in_block == 0`,
      // `c_out_block <= cOutAligned`, and
      // `cOutAligned % c_out_block == 0`. Non-multiples can satisfy the
      // divisibility filter only by coincidence and would fail at
      // runtime.
      //
      // kMaxChannelBlock = 256 is the practical cap on the cartesian
      // product; workloads with cInAligned or cOutAligned beyond it
      // fall back to the largest tile-aligned divisor under the cap
      // selected by OpModel-driven ranking.
      constexpr uint32_t kTileWidth = 32;
      constexpr uint32_t kMaxChannelBlock = 256;
      for (uint32_t c = kTileWidth; c <= kMaxChannelBlock; c += kTileWidth) {
        s.cInBlock.push_back(c);
        s.cOutBlock.push_back(c);
      }

      // t_out_block: every positive integer in [1, kMaxTOutBlock].
      //
      // Temporal blocks face no tile-alignment or divisibility
      // constraint; the structural filter checks only
      // `0 < t_out_block <= T_out` (tt-metal's conv3d kernel pads
      // non-divisor trailing blocks at runtime).
      //
      // kMaxTOutBlock = 16 is a structural cap above the T_out of
      // typical 3D-conv workloads; values exceeding any given op's
      // T_out are filtered out per-op.
      constexpr uint32_t kMaxTOutBlock = 16;
      for (uint32_t t = 1; t <= kMaxTOutBlock; ++t) {
        s.tOutBlock.push_back(t);
      }

      // h_out_block / w_out_block: powers of 2 in [1, kMaxSpatialBlock].
      //
      // Spatial blocks are restricted to powers of 2: non-power-of-2
      // values misalign with tt-metal's 32x32 tile grid for matmul
      // aspect, and DRAM page strides favour 2^n. The structural filter
      // caps each block at the corresponding output extent and enforces
      // `h_out_block * w_out_block <= 256`.
      //
      // kMaxSpatialBlock = 32 is the largest power of 2 that fits the
      // h*w <= 256 budget paired with any nontrivial counterpart.
      constexpr uint32_t kMaxSpatialBlock = 32;
      for (uint32_t v = 1; v <= kMaxSpatialBlock; v *= 2) {
        s.hOutBlock.push_back(v);
        s.wOutBlock.push_back(v);
      }

      return s;
    }();
    return searchSpace;
  }
};

// This analysis takes legal configs found by LegalOpLayoutAnalysis and applies
// op config overrides (such as conv2d config overrides). Also, it searches
// through all legal op specific attributes and applies cartesian product of
// them with legal output layouts.
class LegalOpConfigAnalysis
    : public TTNNAnalysis<LegalOpConfigAnalysisInput, std::vector<OpConfig>> {
public:
  using TTNNAnalysis<LegalOpConfigAnalysisInput,
                     std::vector<OpConfig>>::TTNNAnalysis;

  LegalOpConfigAnalysis(Operation *op) : TTNNAnalysis(op) {}

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

  // Fills op specific attributes for all legal configs. Currently, it generates
  // configs for conv2d ops. Result will be cartesian product of all generated
  // configs with legal output layouts. Configs are generated by
  // Conv2dConfigGenerator within search space.
  void fillOpSpecificAttrs();

  // Search space for conv2d config. Shared across all conv2d ops.
  Conv2dConfigSearchSpace searchSpace = Conv2dConfigSearchSpaceFactory::get();

  // Search space for conv3d config. Shared across all conv3d ops.
  Conv3dConfigSearchSpace conv3dSearchSpace =
      Conv3dConfigSearchSpaceFactory::get();
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPCONFIGANALYSIS_H
