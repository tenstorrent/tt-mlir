// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingCostModel.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

#include "stablehlo/dialect/StablehloOps.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

#define DEBUG_TYPE "auto-sharding"

// =============================================================================
// AutoSharding Pass
// =============================================================================
//
// Algorithm overview:
//   1. Extract the mesh from the module and identify all shardable axes.
//   2. Enumerate valid per-argument sharding configurations. For small search
//      spaces, exhaustive Cartesian product is used. For large models,
//      shape-class grouping, layer-position analysis, or subgraph-template
//      matching reduces the space to a tractable size.
//   3. For each configuration: clone the module, apply sharding hints, run the
//      full remaining StableHLO pipeline (propagation, reshard-to-collectives,
//      canonicalization), and score the result using a cost model.
//   4. The cost model computes:
//        net_cost = (communication_cost + output_gather_cost)
//                 - (memory_benefit + compute_benefit)
//      where communication_cost sums weighted CCL ops, output_gather_cost
//      penalizes sharded outputs, memory_benefit estimates savings from
//      distributing tensors, and compute_benefit estimates FLOPs reduction.
//   5. Select the configuration with the lowest net_cost and apply it to the
//      original module.
//
// Search strategies (automatically selected based on search space size):
//   - Exhaustive: full Cartesian product (for ≤50k configs).
//   - Shape-class exhaustive: group args by shape, enumerate class configs.
//   - Layer-position exhaustive: detect repeating layer patterns, group by
//     position within the period.
//   - Hierarchical (greedy + pairwise): coordinate descent over shape classes
//     or individual arguments, followed by pairwise refinement.
//   - Subgraph-wise: match against known architecture templates (e.g. llama),
//     search within each subgraph family independently.
//
// Known limitations:
//   - Cost model uses heuristic weights — not calibrated to specific hardware.
//   - Subgraph templates are hardcoded for known architectures.
//
// =============================================================================

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_AUTOSHARDINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Data structures.
//===----------------------------------------------------------------------===//

// How the search space is explored. Selected automatically by
// enumerateBaseConfigs / analyzeModule based on search space size.
enum class SearchStrategy {
  Exhaustive,              // Full Cartesian product of all per-arg options.
  ShapeClassExhaustive,    // Group args by tensor shape, enumerate group combos.
  LayerPositionExhaustive, // Detect repeating layer pattern, enumerate by
                           // position within the period.
  Hierarchical,            // Greedy coordinate descent + pairwise refinement
                           // over shape classes or individual args.
  SubgraphWise,            // Match a known architecture template (e.g. llama),
                           // search each subgraph family independently.
};

// Mesh metadata extracted from the module. Only axes with size > 1 (i.e.,
// actually shardable) are stored.
struct MeshInfo {
  llvm::StringRef meshName;
  // Each entry is (axisName, axisSize), with size > 1.
  llvm::SmallVector<std::pair<std::string, int64_t>> axes;
};

// A group of function arguments that share the same tensor shape and therefore
// the same set of valid sharding options. Used to reduce the search space by
// assigning one sharding decision per group instead of per argument.
struct ShapeClass {
  std::string key;                          // Shape signature, e.g. "32x128xf32".
  llvm::SmallVector<size_t> argIndices;     // Indices of args in this class.
  llvm::SmallVector<TensorShardOption> options; // Valid sharding options for
                                                // any arg in this class.
};

// Detected repeating structure in the argument list. For transformer-like
// models, arguments follow a pattern: [prefix] [layer0..layerN] x repeats [suffix].
//
// Example: 3 prefix args, period 11, 28 repeats, 0 suffix args:
//   args 0..2 are prefix, args 3..13 are layer 0, args 14..24 are layer 1, ...
struct RepeatPattern {
  size_t prefixLen;  // Number of non-repeating args before the repeating block.
  size_t period;     // Number of args per layer (the repeating unit).
  size_t numRepeats; // How many times the period repeats.
  size_t suffixLen;  // Number of non-repeating args after the repeating block.
};

// One logical group of layer positions within a SubgraphTemplate.
// E.g., "attention" might cover positions {0,1,2,3,8,9} within a layer period.
struct SubgraphFamily {
  std::string name;                         // E.g. "attention", "mlp".
  llvm::SmallVector<size_t> periodPositions; // Positions within the period.
};

// A known architecture template that maps layer positions to subgraph families.
// Used by SubgraphWise search to decompose the search into per-family sweeps.
struct SubgraphTemplate {
  std::string archName;                       // E.g. "llama".
  size_t period;                              // Expected period (must match RepeatPattern).
  llvm::SmallVector<SubgraphFamily> families; // Subgraph families in this arch.
};

// A subgraph family with its resolved shape-class group indices, used during
// SubgraphWise search to iterate over families.
struct FamilySearch {
  std::string name;                        // Family name (e.g. "attention") or "other".
  llvm::SmallVector<size_t> groupIndices;  // Indices into AnalysisResult::shapeClasses.
};

// Everything needed to run the sharding search. Constructed by analyzeModule(),
// then passed to the search and application functions.
struct AnalysisResult {
  // --- Core fields (always populated) ---
  MeshInfo meshInfo;
  func::FuncOp originalFuncOp;
  llvm::SmallVector<std::string> shardableAxisNames; // Names of axes with size > 1.
  llvm::SmallVector<int64_t> shardableAxisSizes;     // Sizes, parallel to above.

  // --- Search strategy ---
  SearchStrategy strategy = SearchStrategy::Exhaustive;
  std::string strategyLabel;   // Human-readable label for logs/summary.

  // --- Sharding configurations ---
  // For Exhaustive/ShapeClass/LayerPosition: populated by analyzeModule().
  // For Hierarchical/SubgraphWise: populated during the search itself.
  llvm::SmallVector<ShardingConfig> configs;
  size_t numBaseConfigs = 0;   // Count of pre-constraint-expansion configs
                               // (for summary output only).

  // --- Constraint candidates (intermediate ops eligible for sharding) ---
  llvm::SmallVector<ConstraintCandidate> candidates;

  // --- Search space reduction data ---
  llvm::SmallVector<llvm::SmallVector<TensorShardOption>> perArgOptions;
  llvm::SmallVector<size_t> prunedArgs;     // Args too small to shard.
  llvm::SmallVector<ShapeClass> shapeClasses;

  // --- Layer/subgraph structure (populated when detected) ---
  std::optional<RepeatPattern> repeatPattern;
  std::optional<SubgraphTemplate> subgraphTemplate;

  // --- Output/debug ---
  std::string dumpRoot;      // Empty if dumping is disabled.
  std::string manualRefPath; // Path to manual-sharding MLIR for comparison.
};

// Cost evaluation result for a single sharding variant.
struct VariantResult {
  size_t idx;        // Index into AnalysisResult::configs.
  std::string label; // Human-readable sharding description, e.g.
                     // "[arg0:[S,R], arg1:[R]]".
  bool succeeded;    // Whether the variant successfully lowered.
  double cost;       // Net cost (commCost - memBenefit). Only valid if
                     // succeeded.
  double commCost;   // Communication cost from CCL ops.
  double memBenefit; // Memory benefit from tensor distribution.
};

// Result of the sharding search.
struct SearchResult {
  size_t bestIdx;  // Index of the winning config in AnalysisResult::configs.
  double bestCost; // Net cost of the winning config.
  llvm::SmallVector<VariantResult> results; // Per-variant results (for summary
                                            // output). Is empty if dumping
                                            // is disabled.
};

//===----------------------------------------------------------------------===//
// Mesh extraction.
//===----------------------------------------------------------------------===//

// Extract full mesh information including all shardable axes.
static std::optional<MeshInfo> extractMeshInfo(ModuleOp &module) {
  llvm::SmallVector<mlir::sdy::MeshOp> meshOps =
      shardy_utils::getMeshOps(module);
  if (meshOps.empty()) {
    return std::nullopt;
  }

  MeshInfo info;
  info.meshName = meshOps[0].getSymName();
  for (auto axisAttr : meshOps[0].getMeshAttr().getAxes()) {
    if (axisAttr.getSize() > 1) {
      info.axes.emplace_back(axisAttr.getName().str(), axisAttr.getSize());
    }
  }
  return info;
}

//===----------------------------------------------------------------------===//
// Manual sharding reference parsing.
//===----------------------------------------------------------------------===//

// Parse a manual-sharding reference MLIR file, extracting the per-argument
// TensorShardOptions. Used to compare auto-sharding results against a
// hand-tuned sharding in the summary output.
static llvm::SmallVector<TensorShardOption>
parseManualShardings(const std::string &filePath,
                     const llvm::SmallVector<std::string> &axisNames) {
  llvm::SmallVector<TensorShardOption> result;

  auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (!bufferOrErr) {
    llvm::errs() << "AutoSharding: cannot read manual ref file: " << filePath
                 << "\n";
    return result;
  }

  llvm::StringRef content = (*bufferOrErr)->getBuffer();
  size_t pos = 0;
  while (pos < content.size()) {
    size_t shardingStart = content.find("#sdy.sharding<@", pos);
    if (shardingStart == llvm::StringRef::npos) {
      break;
    }

    size_t bracketStart = content.find('[', shardingStart);
    if (bracketStart == llvm::StringRef::npos) {
      break;
    }

    size_t bracketEnd = content.find(']', bracketStart);
    if (bracketEnd == llvm::StringRef::npos) {
      break;
    }

    llvm::StringRef dimList = content.slice(bracketStart + 1, bracketEnd);
    TensorShardOption argSharding;

    size_t dimPos = 0;
    while (dimPos < dimList.size()) {
      size_t openBrace = dimList.find('{', dimPos);
      if (openBrace == llvm::StringRef::npos) {
        break;
      }
      size_t closeBrace = dimList.find('}', openBrace);
      if (closeBrace == llvm::StringRef::npos) {
        break;
      }
      llvm::StringRef dimContent = dimList.slice(openBrace + 1, closeBrace);

      DimShardSpec dimSpec;
      if (!dimContent.empty()) {
        size_t searchPos = 0;
        while (searchPos < dimContent.size()) {
          size_t quoteStart = dimContent.find('"', searchPos);
          if (quoteStart == llvm::StringRef::npos) {
            break;
          }
          size_t quoteEnd = dimContent.find('"', quoteStart + 1);
          if (quoteEnd == llvm::StringRef::npos) {
            break;
          }
          llvm::StringRef axisName =
              dimContent.slice(quoteStart + 1, quoteEnd);
          for (size_t i = 0; i < axisNames.size(); ++i) {
            if (axisNames[i] == axisName) {
              dimSpec.push_back(i);
              break;
            }
          }
          searchPos = quoteEnd + 1;
        }
      }
      argSharding.push_back(std::move(dimSpec));
      dimPos = closeBrace + 1;
    }

    result.push_back(std::move(argSharding));
    pos = bracketEnd + 1;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Configuration enumeration.
//===----------------------------------------------------------------------===//

// Multi-axis configuration enumeration helpers.

// Scan the module for intermediate ops eligible for sharding constraints
// (e.g. dot_general, reduce_scatter). Returns up to maxCandidates candidates,
// prioritized by op type and position.
static llvm::SmallVector<ConstraintCandidate>
collectConstraintCandidates(ModuleOp &module, int64_t maxCandidates) {
  llvm::SmallVector<ConstraintCandidate> candidates;
  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return candidates;
  }
  func::FuncOp funcOp = *funcOps.begin();

  size_t opIdx = 0;
  for (auto &op : funcOp.getBody().front()) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      ++opIdx;
      continue;
    }
    if (op.getNumResults() > 0) {
      if (auto tensorType =
              dyn_cast<RankedTensorType>(op.getResult(0).getType())) {
        candidates.push_back({opIdx, tensorType.getRank()});
        if (static_cast<int64_t>(candidates.size()) >= maxCandidates) {
          break;
        }
      }
    }
    ++opIdx;
  }
  return candidates;
}

// Enumerate all valid per-dimension sharding options for a tensor of the given
// rank across numAxes mesh axes. Each option assigns at most one axis per
// dimension (no axis reuse). Returns the set of valid TensorShardOption values.
static llvm::SmallVector<TensorShardOption>
enumerateValidDimShardings(int64_t rank, size_t numAxes) {
  llvm::SmallVector<TensorShardOption> options;

  size_t totalCombinations = 1;
  for (size_t a = 0; a < numAxes; ++a) {
    totalCombinations *= (static_cast<size_t>(rank) + 1);
  }

  for (size_t combo = 0; combo < totalCombinations; ++combo) {
    TensorShardOption dimAxes(static_cast<size_t>(rank));
    size_t remaining = combo;
    for (size_t a = 0; a < numAxes; ++a) {
      size_t target = remaining % (static_cast<size_t>(rank) + 1);
      remaining /= (static_cast<size_t>(rank) + 1);
      if (target < static_cast<size_t>(rank)) {
        dimAxes[target].push_back(a);
      }
    }
    options.push_back(std::move(dimAxes));
  }
  return options;
}

// Build the set of valid sharding options for each function argument.
// Arguments with fewer than kMinShardableElements are added to prunedArgs
// and assigned a single "all replicated" option.
static llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
enumeratePerArgOptions(ModuleOp &module, llvm::SmallVector<size_t> &prunedArgs,
                       size_t numAxes) {
  constexpr int64_t kMinShardableElements = 4096;
  llvm::SmallVector<llvm::SmallVector<TensorShardOption>> perArgOptions;

  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return perArgOptions;
  }
  func::FuncOp funcOp = *funcOps.begin();

  for (auto arg : funcOp.getArguments()) {
    auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensorType) {
      perArgOptions.push_back({TensorShardOption()});
      continue;
    }

    int64_t rank = tensorType.getRank();
    int64_t numElements = 1;
    for (auto dim : tensorType.getShape()) {
      numElements *= dim;
    }

    if (numElements < kMinShardableElements) {
      prunedArgs.push_back(arg.getArgNumber());
      llvm::errs() << "AutoSharding: pruning arg " << arg.getArgNumber()
                   << " (" << numElements
                   << " elements) -- too small to shard\n";
      perArgOptions.push_back(
          {TensorShardOption(static_cast<size_t>(rank))});
    } else {
      perArgOptions.push_back(enumerateValidDimShardings(rank, numAxes));
    }
  }

  return perArgOptions;
}

//===----------------------------------------------------------------------===//
// Shape class grouping.
//===----------------------------------------------------------------------===//

// Return a canonical string key for a tensor shape, e.g. "32x128xf32".
// Used to group arguments with identical shapes into ShapeClasses.
static std::string getShapeKey(RankedTensorType type) {
  std::string key;
  llvm::raw_string_ostream os(key);
  auto shape = type.getShape();
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (i > 0) {
      os << "x";
    }
    os << shape[i];
  }
  os << "x";
  type.getElementType().print(os);
  return key;
}

//===----------------------------------------------------------------------===//
// Subgraph template database.
//===----------------------------------------------------------------------===//

// Return the built-in list of known architecture templates (e.g. llama).
// Each template maps layer positions within a repeat period to subgraph
// families for the SubgraphWise search strategy.
static llvm::SmallVector<SubgraphTemplate> getKnownTemplates() {
  llvm::SmallVector<SubgraphTemplate> templates;

  SubgraphTemplate llama;
  llama.archName = "llama";
  llama.period = 11;
  llama.families.push_back({"attention", {0, 1, 2, 3, 8, 9}});
  llama.families.push_back({"mlp", {5, 6, 10}});
  templates.push_back(std::move(llama));

  return templates;
}

// Try to match the detected repeat pattern against a known architecture
// template. Returns the matching template if the period matches and all
// template families can be mapped to existing shape-class groups.
static std::optional<SubgraphTemplate>
matchSubgraphTemplate(const RepeatPattern &pattern,
                      const llvm::SmallVector<ShapeClass> &groups) {
  auto templates = getKnownTemplates();
  for (auto &tmpl : templates) {
    if (tmpl.period != pattern.period) {
      continue;
    }

    bool hasViableFamily = false;
    for (const auto &family : tmpl.families) {
      size_t shardableCount = 0;
      for (size_t pos : family.periodPositions) {
        std::string groupPrefix = "layer_pos" + std::to_string(pos) + "_";
        for (const auto &g : groups) {
          if (g.key.find(groupPrefix) == 0) {
            ++shardableCount;
            break;
          }
        }
      }
      if (shardableCount >= 2) {
        hasViableFamily = true;
        break;
      }
    }

    if (hasViableFamily) {
      llvm::errs() << "AutoSharding: matched subgraph template '"
                   << tmpl.archName << "' (period=" << tmpl.period << ")\n";
      return tmpl;
    }
  }
  return std::nullopt;
}

// Detect a repeating pattern in the argument shape sequence (e.g., transformer
// layers). Tries candidate periods from kMinRepeats upward and returns the
// first pattern that accounts for all arguments. Returns nullopt if no
// repeating structure is found.
static std::optional<RepeatPattern>
detectRepeatPeriod(const llvm::SmallVector<std::string> &shapes) {
  size_t N = shapes.size();
  constexpr size_t kMinRepeats = 3;

  for (size_t P = 1; P <= N / kMinRepeats; ++P) {
    for (size_t prefix = 0; prefix + kMinRepeats * P <= N; ++prefix) {
      for (size_t suffix : {static_cast<size_t>(0), P}) {
        if (prefix + kMinRepeats * P + suffix > N) {
          continue;
        }
        size_t repeating = N - prefix - suffix;
        if (repeating % P != 0) {
          continue;
        }
        size_t repeats = repeating / P;
        if (repeats < kMinRepeats) {
          continue;
        }

        bool match = true;
        for (size_t i = prefix + P; i < prefix + repeating; ++i) {
          if (shapes[i] != shapes[prefix + (i - prefix) % P]) {
            match = false;
            break;
          }
        }

        if (match) {
          return RepeatPattern{prefix, P, repeats, suffix};
        }
      }
    }
  }
  return std::nullopt;
}

// Group arguments by their position within a repeating layer pattern.
// All args at the same layer position (e.g., "all query weight matrices")
// share one sharding decision, dramatically reducing the search space for
// models with many identical layers.
static llvm::SmallVector<ShapeClass> buildLayerPositionGroups(
    ModuleOp &module,
    const llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
        &perArgOptions,
    const RepeatPattern &pattern) {
  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return {};
  }
  func::FuncOp funcOp = *funcOps.begin();

  llvm::SmallVector<ShapeClass> groups;

  for (size_t i = 0; i < pattern.prefixLen; ++i) {
    if (i >= perArgOptions.size() || perArgOptions[i].size() <= 1) {
      continue;
    }
    auto tensorType =
        dyn_cast<RankedTensorType>(funcOp.getArgument(i).getType());
    if (!tensorType) {
      continue;
    }
    ShapeClass sc;
    sc.key = "prefix_" + std::to_string(i) + "_" + getShapeKey(tensorType);
    sc.argIndices.push_back(i);
    sc.options = perArgOptions[i];
    groups.push_back(std::move(sc));
  }

  for (size_t pos = 0; pos < pattern.period; ++pos) {
    size_t firstArgIdx = pattern.prefixLen + pos;
    if (firstArgIdx >= perArgOptions.size() ||
        perArgOptions[firstArgIdx].size() <= 1) {
      continue;
    }
    auto tensorType =
        dyn_cast<RankedTensorType>(funcOp.getArgument(firstArgIdx).getType());
    if (!tensorType) {
      continue;
    }

    ShapeClass sc;
    sc.key = "layer_pos" + std::to_string(pos) + "_" + getShapeKey(tensorType);
    sc.options = perArgOptions[firstArgIdx];
    for (size_t r = 0; r < pattern.numRepeats; ++r) {
      sc.argIndices.push_back(pattern.prefixLen + r * pattern.period + pos);
    }
    groups.push_back(std::move(sc));
  }

  size_t suffixStart = pattern.prefixLen + pattern.numRepeats * pattern.period;
  size_t totalArgs = funcOp.getNumArguments();
  for (size_t i = suffixStart; i < totalArgs; ++i) {
    if (i >= perArgOptions.size() || perArgOptions[i].size() <= 1) {
      continue;
    }
    auto tensorType =
        dyn_cast<RankedTensorType>(funcOp.getArgument(i).getType());
    if (!tensorType) {
      continue;
    }

    size_t posInBlock = i - suffixStart;
    std::string suffixKey = getShapeKey(tensorType);
    bool folded = false;

    if (posInBlock < pattern.period) {
      size_t refArgIdx = pattern.prefixLen + posInBlock;
      auto refType =
          dyn_cast<RankedTensorType>(funcOp.getArgument(refArgIdx).getType());
      if (refType && getShapeKey(refType) == suffixKey) {
        std::string groupKey =
            "layer_pos" + std::to_string(posInBlock) + "_" + suffixKey;
        for (auto &g : groups) {
          if (g.key == groupKey) {
            g.argIndices.push_back(i);
            folded = true;
            break;
          }
        }
      }
    }

    if (!folded) {
      ShapeClass sc;
      sc.key = "suffix_" + std::to_string(posInBlock) + "_" + suffixKey;
      sc.argIndices.push_back(i);
      sc.options = perArgOptions[i];
      groups.push_back(std::move(sc));
    }
  }

  for (const auto &sc : groups) {
    llvm::errs() << "AutoSharding: layer-position group '" << sc.key << "' -> "
                 << sc.argIndices.size() << " args, " << sc.options.size()
                 << " options\n";
  }

  return groups;
}

// Group arguments by tensor shape. Arguments with the same shape get the same
// set of valid sharding options and are assigned a single group decision.
// Fallback grouping when no layer-position pattern is detected.
static llvm::SmallVector<ShapeClass> buildShapeClasses(
    ModuleOp &module,
    const llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
        &perArgOptions) {
  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return {};
  }
  func::FuncOp funcOp = *funcOps.begin();

  llvm::StringMap<size_t> keyToIndex;
  llvm::SmallVector<ShapeClass> classes;

  for (auto arg : funcOp.getArguments()) {
    size_t argIdx = arg.getArgNumber();
    if (argIdx >= perArgOptions.size() || perArgOptions[argIdx].size() <= 1) {
      continue;
    }

    auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensorType) {
      continue;
    }

    std::string key = getShapeKey(tensorType);
    auto it = keyToIndex.find(key);
    if (it == keyToIndex.end()) {
      keyToIndex[key] = classes.size();
      ShapeClass sc;
      sc.key = key;
      sc.options = perArgOptions[argIdx];
      sc.argIndices.push_back(argIdx);
      classes.push_back(std::move(sc));
    } else {
      classes[it->second].argIndices.push_back(argIdx);
    }
  }

  for (const auto &sc : classes) {
    llvm::errs() << "AutoSharding: shape class '" << sc.key << "' -> "
                 << sc.argIndices.size() << " args, " << sc.options.size()
                 << " options\n";
  }

  return classes;
}

// Expand a shape-class-level config (one sharding per class) into a full
// per-argument config by broadcasting each class's sharding to all its member
// args. Pruned args keep their default (replicated) sharding.
static ShardingConfig expandShapeClassConfig(
    const ShardingConfig &classConfig,
    const llvm::SmallVector<ShapeClass> &shapeClasses,
    const llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
        &perArgOptions) {
  ShardingConfig config;
  size_t numArgs = perArgOptions.size();

  for (size_t a = 0; a < numArgs; ++a) {
    config.argDimSharding.push_back(perArgOptions[a][0]);
  }

  for (size_t c = 0; c < shapeClasses.size(); ++c) {
    for (size_t argIdx : shapeClasses[c].argIndices) {
      config.argDimSharding[argIdx] = classConfig.argDimSharding[c];
    }
  }

  return config;
}

// Recursive helper for cartesianProduct(). Builds configs one argument at a
// time by choosing each valid option for the current arg.
static void cartesianProductHelper(
    const llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
        &perArgOptions,
    size_t argIdx, ShardingConfig &current,
    llvm::SmallVector<ShardingConfig> &result) {
  if (argIdx == perArgOptions.size()) {
    result.push_back(current);
    return;
  }
  for (const auto &option : perArgOptions[argIdx]) {
    current.argDimSharding.push_back(option);
    cartesianProductHelper(perArgOptions, argIdx + 1, current, result);
    current.argDimSharding.pop_back();
  }
}

// Compute the full Cartesian product of per-argument (or per-group) sharding
// options. Each resulting ShardingConfig has one TensorShardOption per entry.
static llvm::SmallVector<ShardingConfig> cartesianProduct(
    const llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
        &perArgOptions) {
  llvm::SmallVector<ShardingConfig> result;
  ShardingConfig current;
  cartesianProductHelper(perArgOptions, 0, current, result);
  return result;
}

// Enumerate the base sharding configurations using a cascading fallback
// strategy. Returns empty to signal the caller to use Hierarchical search.
//
// Strategy selection cascade:
//   1. Full Cartesian product of per-arg options <= kMaxExhaustiveConfigs?
//      -> return all configs (Exhaustive strategy)
//   2. Repeating layer pattern detected?
//      a. Build layer-position groups, group product <= kMaxExhaustiveConfigs?
//         -> return configs (LayerPositionExhaustive strategy)
//      b. Else -> return empty (caller will use Hierarchical or SubgraphWise)
//   3. Build shape classes, class product <= kMaxExhaustiveConfigs?
//      -> return configs (ShapeClassExhaustive strategy)
//   4. Else -> return empty (caller will use Hierarchical)
static llvm::SmallVector<ShardingConfig> enumerateBaseConfigs(
    ModuleOp &module,
    llvm::SmallVector<llvm::SmallVector<TensorShardOption>> &perArgOptions,
    llvm::SmallVector<size_t> &prunedArgs,
    llvm::SmallVector<ShapeClass> &shapeClasses, size_t numAxes) {
  perArgOptions = enumeratePerArgOptions(module, prunedArgs, numAxes);
  if (perArgOptions.empty()) {
    return {};
  }

  constexpr int64_t kMaxExhaustiveConfigs = 50000;
  int64_t searchSpace = 1;
  bool overflow = false;
  for (const auto &opts : perArgOptions) {
    searchSpace *= static_cast<int64_t>(opts.size());
    if (searchSpace > kMaxExhaustiveConfigs) {
      overflow = true;
      break;
    }
  }

  if (!overflow) {
    llvm::errs() << "AutoSharding: full search space = " << searchSpace
                 << " configs, using exhaustive enumeration\n";
    return cartesianProduct(perArgOptions);
  }

  auto funcOps = module.getOps<func::FuncOp>();
  if (!funcOps.empty()) {
    func::FuncOp funcOp = *funcOps.begin();
    llvm::SmallVector<std::string> shapes;
    for (auto arg : funcOp.getArguments()) {
      auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
      shapes.push_back(tensorType ? getShapeKey(tensorType) : "_non_tensor_");
    }

    auto pattern = detectRepeatPeriod(shapes);
    if (pattern) {
      llvm::errs() << "AutoSharding: detected repeating layer pattern: prefix="
                   << pattern->prefixLen << ", period=" << pattern->period
                   << ", repeats=" << pattern->numRepeats
                   << ", suffix=" << pattern->suffixLen << "\n";

      shapeClasses = buildLayerPositionGroups(module, perArgOptions, *pattern);

      if (!shapeClasses.empty()) {
        llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
            perGroupOptions;
        for (const auto &sc : shapeClasses) {
          perGroupOptions.push_back(sc.options);
        }

        int64_t groupSpace = 1;
        bool groupOverflow = false;
        for (const auto &opts : perGroupOptions) {
          groupSpace *= static_cast<int64_t>(opts.size());
          if (groupSpace > kMaxExhaustiveConfigs) {
            groupOverflow = true;
            break;
          }
        }

        if (!groupOverflow) {
          llvm::errs()
              << "AutoSharding: layer-position search space = " << groupSpace
              << " configs (" << shapeClasses.size()
              << " groups), using layer-position exhaustive enumeration\n";
          auto groupConfigs = cartesianProduct(perGroupOptions);
          llvm::SmallVector<ShardingConfig> fullConfigs;
          fullConfigs.reserve(groupConfigs.size());
          for (const auto &gc : groupConfigs) {
            fullConfigs.push_back(
                expandShapeClassConfig(gc, shapeClasses, perArgOptions));
          }
          return fullConfigs;
        }

        llvm::errs()
            << "AutoSharding: layer-position space too large (>"
            << kMaxExhaustiveConfigs
            << "), will use hierarchical search over layer-position groups\n";
        return {};
      }
    }
  }

  shapeClasses = buildShapeClasses(module, perArgOptions);
  if (shapeClasses.empty()) {
    llvm::errs() << "AutoSharding: no shape classes found, "
                 << "will use hierarchical search\n";
    return {};
  }

  llvm::SmallVector<llvm::SmallVector<TensorShardOption>> perClassOptions;
  for (const auto &sc : shapeClasses) {
    perClassOptions.push_back(sc.options);
  }

  int64_t classSpace = 1;
  bool classOverflow = false;
  for (const auto &opts : perClassOptions) {
    classSpace *= static_cast<int64_t>(opts.size());
    if (classSpace > kMaxExhaustiveConfigs) {
      classOverflow = true;
      break;
    }
  }

  if (!classOverflow) {
    llvm::errs() << "AutoSharding: shape-class search space = " << classSpace
                 << " configs (" << shapeClasses.size()
                 << " classes), using shape-class exhaustive enumeration\n";
    auto classConfigs = cartesianProduct(perClassOptions);
    llvm::SmallVector<ShardingConfig> fullConfigs;
    fullConfigs.reserve(classConfigs.size());
    for (const auto &cc : classConfigs) {
      fullConfigs.push_back(
          expandShapeClassConfig(cc, shapeClasses, perArgOptions));
    }
    return fullConfigs;
  }

  llvm::errs() << "AutoSharding: even shape-class space too large (>"
               << kMaxExhaustiveConfigs
               << "), will use hierarchical search over shape classes\n";
  return {};
}

// Expand base configs with constraint candidates. For each base config, generate
// one variant per constraint candidate (with each valid sharding for that
// candidate). Returns the full set of configs to evaluate.
static llvm::SmallVector<ShardingConfig> expandWithConstraints(
    const llvm::SmallVector<ShardingConfig> &baseConfigs,
    const llvm::SmallVector<ConstraintCandidate> &candidates,
    size_t numAxes) {
  if (candidates.empty()) {
    return baseConfigs;
  }

  llvm::SmallVector<llvm::SmallVector<TensorShardOption>> perCandidateOptions;
  for (const auto &cand : candidates) {
    perCandidateOptions.push_back(
        enumerateValidDimShardings(cand.rank, numAxes));
  }

  size_t constraintCombinations = 1;
  for (const auto &opts : perCandidateOptions) {
    constraintCombinations *= (opts.size() + 1);
  }

  size_t totalConfigs = baseConfigs.size() * constraintCombinations;
  constexpr size_t kMaxConfigs = 50000;
  if (totalConfigs > kMaxConfigs) {
    llvm::errs() << "AutoSharding: constraint expansion would produce "
                 << totalConfigs << " configs (cap=" << kMaxConfigs
                 << "), skipping constraint expansion\n";
    return baseConfigs;
  }

  llvm::errs() << "AutoSharding: expanding " << baseConfigs.size()
               << " base configs x " << constraintCombinations
               << " constraint combos = " << totalConfigs << " total\n";

  llvm::SmallVector<ShardingConfig> combined;
  combined.reserve(totalConfigs);

  for (const auto &t1 : baseConfigs) {
    for (size_t ci = 0; ci < constraintCombinations; ++ci) {
      ShardingConfig config;
      config.argDimSharding = t1.argDimSharding;

      size_t remaining = ci;
      for (size_t k = 0; k < candidates.size(); ++k) {
        size_t numOptions = perCandidateOptions[k].size() + 1;
        size_t choice = remaining % numOptions;
        remaining /= numOptions;

        if (choice == 0) {
          config.constraintTargets.push_back(std::nullopt);
        } else {
          config.constraintTargets.push_back(
              perCandidateOptions[k][choice - 1]);
        }
      }
      combined.push_back(std::move(config));
    }
  }
  return combined;
}

//===----------------------------------------------------------------------===//
// Sharding application.
//===----------------------------------------------------------------------===//

// Apply arg-level sharding hints using ShardingConfig (multi-axis, with
// optional sharding constraints on intermediate ops).
static void
applyShardingHints(ModuleOp module, const ShardingConfig &config,
                   StringRef meshName,
                   const llvm::SmallVector<std::string> &axisNames,
                   const llvm::SmallVector<ConstraintCandidate> &candidates) {
  MLIRContext *context = module.getContext();

  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return;
  }
  func::FuncOp funcOp = *funcOps.begin();

  auto buildDimShardings =
      [&](const TensorShardOption &shardOption)
          -> llvm::SmallVector<mlir::sdy::DimensionShardingAttr> {
    constexpr bool isClosed = false;
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
    for (const auto &dimAxes : shardOption) {
      llvm::SmallVector<mlir::sdy::AxisRefAttr> axisRefs;
      for (size_t axisIdx : dimAxes) {
        if (axisIdx < axisNames.size()) {
          axisRefs.push_back(
              mlir::sdy::AxisRefAttr::get(context, axisNames[axisIdx]));
        }
      }
      dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
          context, axisRefs, isClosed));
    }
    return dimShardings;
  };

  for (size_t argIdx = 0; argIdx < config.argDimSharding.size(); ++argIdx) {
    auto dimShardings = buildDimShardings(config.argDimSharding[argIdx]);

    auto sharding = mlir::sdy::TensorShardingAttr::get(
        context, meshName, dimShardings, /*replicatedAxes=*/{},
        /*unreducedAxes=*/{});

    auto existingDict = funcOp.getArgAttrDict(argIdx);
    std::optional<mlir::DictionaryAttr> optDict =
        existingDict ? std::optional(existingDict) : std::nullopt;
    auto newDict = shardy_utils::addDictionaryAttrSdyShardingAnnotation(
        context, sharding, optDict);
    funcOp.setArgAttrs(argIdx, newDict);
  }

  if (config.constraintTargets.empty() || candidates.empty()) {
    return;
  }

  llvm::SmallVector<Operation *> bodyOps;
  for (auto &op : funcOp.getBody().front()) {
    bodyOps.push_back(&op);
  }

  OpBuilder builder(context);

  for (size_t k = 0;
       k < candidates.size() && k < config.constraintTargets.size(); ++k) {
    if (!config.constraintTargets[k]) {
      continue;
    }

    const auto &target = *config.constraintTargets[k];
    size_t opIdx = candidates[k].opIndex;
    if (opIdx >= bodyOps.size()) {
      continue;
    }

    Operation *targetOp = bodyOps[opIdx];
    if (targetOp->getNumResults() == 0) {
      continue;
    }

    Value result = targetOp->getResult(0);

    auto dimShardings = buildDimShardings(target);
    auto sharding = mlir::sdy::TensorShardingAttr::get(
        context, meshName, dimShardings, /*replicatedAxes=*/{},
        /*unreducedAxes=*/{});

    llvm::SmallVector<OpOperand *> usesToReplace;
    for (OpOperand &use : result.getUses()) {
      usesToReplace.push_back(&use);
    }

    builder.setInsertionPointAfter(targetOp);
    auto constraintOp = builder.create<mlir::sdy::ShardingConstraintOp>(
        targetOp->getLoc(), result.getType(), result, sharding);

    for (OpOperand *use : usesToReplace) {
      use->set(constraintOp.getResult());
    }
  }
}

// Remove stablehlo.custom_call @tt.mark_argument pass-through ops.
static void stripMarkArgumentCalls(ModuleOp module) {
  SmallVector<mlir::stablehlo::CustomCallOp> toErase;
  module.walk([&](mlir::stablehlo::CustomCallOp callOp) {
    if (callOp.getCallTargetName() != "tt.mark_argument") {
      return;
    }
    if (callOp.getNumOperands() != 1 || callOp.getNumResults() != 1) {
      return;
    }
    if (callOp.getOperand(0).getType() != callOp.getResult(0).getType()) {
      return;
    }
    callOp.getResult(0).replaceAllUsesWith(callOp.getOperand(0));
    toErase.push_back(callOp);
  });
  for (auto callOp : toErase) {
    callOp->erase();
  }
}

//===----------------------------------------------------------------------===//
// Pipeline and I/O helpers.
//===----------------------------------------------------------------------===//

// Add the StableHLO-to-TTIR lowering passes that run after sharding hints
// have been applied. This is the pipeline used to evaluate each candidate config.
static void addRemainingStableHLOPasses(OpPassManager &pm) {
  pm.addPass(createDecoupleConstFanoutPass());
  pm.addPass(createDecomposeCustomCallTuplesPass());
  pm.addPass(createFlattenOrConvertCompositesPass());
  pm.addPass(createRegisterCustomShardingRulePass());

  pm.addPass(mlir::sdy::createApplyShardingConstraintsPass());

  mlir::sdy::PropagationOptions propagationOptions;
  propagationOptions.conservativePropagation = true;
  pm.addPass(mlir::sdy::createUserPriorityPropagationPass(propagationOptions));

  pm.nest<func::FuncOp>().addPass(
      mlir::sdy::createShardingConstraintToReshardPass());

  pm.addPass(createReplicateNonSplittableConstantsPass());
  pm.addPass(createInsertExplicitReshardsPass());
  pm.addPass(createWrapUnderManualComputationPass());

  pm.nest<func::FuncOp>().addPass(mlir::sdy::createReshardToCollectivesPass());

  pm.addPass(createShardyCCLCanonicalizationPass());
  pm.addPass(createAnnotateLocalShapesPass());
  pm.addPass(createUpdateGlobalToLocalShapesPass());
  pm.addPass(createReoutlineCompositePass());
  pm.addPass(mlir::sdy::createCloseShardingsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

// Write one dimension's sharding to os: "R" if replicated, "S" followed by
// axis indices if sharded (e.g. "S0", "S01").
static void formatDimSharding(llvm::raw_ostream &os,
                              const DimShardSpec &axes) {
  if (axes.empty()) {
    os << "R";
  } else {
    os << "S";
    for (size_t axisIdx : axes) {
      os << axisIdx;
    }
  }
}

// Format a multi-axis ShardingConfig as a human-readable string,
// e.g. "[arg0:[S0,R], arg1:[R,S1] | cst0:[S0]]".
static std::string formatConfig(const ShardingConfig &config) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "[";
  for (size_t a = 0; a < config.argDimSharding.size(); ++a) {
    if (a > 0) {
      os << ", ";
    }
    os << "arg" << a << ":[";
    for (size_t d = 0; d < config.argDimSharding[a].size(); ++d) {
      if (d > 0) {
        os << ",";
      }
      formatDimSharding(os, config.argDimSharding[a][d]);
    }
    os << "]";
  }
  for (size_t c = 0; c < config.constraintTargets.size(); ++c) {
    os << ", c" << c << ":";
    if (!config.constraintTargets[c]) {
      os << "none";
    } else {
      os << "[";
      for (size_t d = 0; d < config.constraintTargets[c]->size(); ++d) {
        if (d > 0) {
          os << ",";
        }
        formatDimSharding(os, (*config.constraintTargets[c])[d]);
      }
      os << "]";
    }
  }
  os << "]";
  return result;
}

// Generate a filesystem-safe directory name for a variant dump, encoding the
// variant index and a condensed sharding description.
static std::string configDirName(size_t idx, const ShardingConfig &config) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << llvm::format("variant_%02zu", idx);
  for (size_t a = 0; a < config.argDimSharding.size(); ++a) {
    os << "_arg" << a << "-";
    for (const auto &axes : config.argDimSharding[a]) {
      formatDimSharding(os, axes);
    }
  }
  for (size_t c = 0; c < config.constraintTargets.size(); ++c) {
    os << "_c" << c << "-";
    if (!config.constraintTargets[c]) {
      os << "none";
    } else {
      for (const auto &axes : *config.constraintTargets[c]) {
        formatDimSharding(os, axes);
      }
    }
  }
  return result;
}

// Create a timestamped dump directory under `baseDir`.
static std::string createDumpRoot(StringRef baseDir) {
  llvm::SmallString<256> path;
  if (baseDir.empty()) {
    path = ".";
  } else {
    path = baseDir;
  }

  if (auto ec = llvm::sys::fs::create_directories(path)) {
    llvm::errs() << "AutoSharding: failed to create dump directory " << path
                 << ": " << ec.message() << "\n";
    return "";
  }
  return path.str().str();
}

// Serialize a module's MLIR text representation to a file.
// Returns true on success, false on I/O error.
static bool dumpModuleToFile(ModuleOp module, StringRef filePath) {
  std::error_code ec;
  llvm::raw_fd_ostream fos(filePath, ec);
  if (ec) {
    llvm::errs() << "AutoSharding: failed to write " << filePath << ": "
                 << ec.message() << "\n";
    return false;
  }
  module->print(fos);
  fos << "\n";
  return true;
}

//===----------------------------------------------------------------------===//
// Core algorithm: analyze, evaluate configs, and apply the best one.
//===----------------------------------------------------------------------===//

// Evaluate a single sharding config by cloning the module, applying hints,
// running the pipeline, and scoring the result. Returns a VariantResult
// with succeeded=false if lowering fails.
static VariantResult evaluateSingleConfig(ModuleOp rootModule,
                                          const AnalysisResult &analysis,
                                          size_t idx,
                                          const ShardingConfig &config,
                                          int64_t maxElementsOverride = 0) {
  MLIRContext *context = rootModule.getContext();
  ModuleOp clonedModule = cast<ModuleOp>(rootModule->clone());
  auto cleanup =
      llvm::make_scope_exit([&clonedModule] { clonedModule->erase(); });

  stripMarkArgumentCalls(clonedModule);
  applyShardingHints(clonedModule, config, analysis.meshInfo.meshName,
                     analysis.shardableAxisNames, analysis.candidates);

  PassManager pm(context, ModuleOp::getOperationName(),
                 PassManager::Nesting::Implicit);
  addRemainingStableHLOPasses(pm);

  std::string label = formatConfig(config);
  if (failed(pm.run(clonedModule))) {
    return {idx, std::move(label), false, 0.0, 0.0, 0.0};
  }

  ShardingResult sr = evaluate(clonedModule, config,
                               analysis.originalFuncOp,
                               analysis.shardableAxisSizes,
                               maxElementsOverride);
  return {idx, std::move(label), true, sr.netCost,
          sr.communicationCost, sr.memoryBenefit};
}

// Analyze the module: extract mesh, enumerate configs, select search strategy.
static std::optional<AnalysisResult>
analyzeModule(ModuleOp rootModule, int64_t maxConstraintCandidates,
              StringRef dumpDir, StringRef manualRef) {
  // Phase 1: Mesh extraction — find the mesh and its shardable axes.
  auto meshInfoOpt = extractMeshInfo(rootModule);
  if (!meshInfoOpt) {
    rootModule.emitWarning("AutoSharding: no mesh found in module, skipping");
    return std::nullopt;
  }

  AnalysisResult analysis;
  analysis.meshInfo = *meshInfoOpt;

  for (const auto &[name, size] : analysis.meshInfo.axes) {
    if (size > 1) {
      analysis.shardableAxisNames.push_back(name);
      analysis.shardableAxisSizes.push_back(size);
    }
  }
  if (analysis.shardableAxisNames.empty()) {
    llvm::errs() << "AutoSharding: no shardable axes (all size 1), "
                 << "applying all-replicated config\n";
    return std::nullopt;
  }

  llvm::errs() << "AutoSharding: mesh='" << analysis.meshInfo.meshName
               << "', shardable axes=[";
  for (size_t i = 0; i < analysis.shardableAxisNames.size(); ++i) {
    if (i > 0) {
      llvm::errs() << ", ";
    }
    llvm::errs() << "'" << analysis.shardableAxisNames[i] << "' (size="
                 << analysis.shardableAxisSizes[i] << ")";
  }
  llvm::errs() << "]\n";

  auto funcOps = rootModule.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    rootModule.emitWarning("AutoSharding: no FuncOp found, skipping");
    return std::nullopt;
  }
  if (std::distance(funcOps.begin(), funcOps.end()) > 1) {
    rootModule.emitWarning(
        "AutoSharding: multiple FuncOps found (e.g. from composite ops); "
        "only the first will be used for sharding enumeration");
  }
  analysis.originalFuncOp = *funcOps.begin();

  // Phase 2: Config enumeration — build per-arg options and base configs.
  size_t numAxes = analysis.shardableAxisNames.size();
  llvm::SmallVector<ShardingConfig> baseConfigs = enumerateBaseConfigs(
      rootModule, analysis.perArgOptions, analysis.prunedArgs,
      analysis.shapeClasses, numAxes);
  analysis.numBaseConfigs = baseConfigs.size();

  // Phase 3: Strategy selection — choose search strategy based on search
  // space size and detected structure (layer pattern, subgraph template).
  bool usesLayerPositionGroups = false;
  if (!analysis.shapeClasses.empty()) {
    for (const auto &sc : analysis.shapeClasses) {
      if (sc.key.find("layer_pos") == 0 || sc.key.find("prefix_") == 0 ||
          sc.key.find("suffix_") == 0) {
        usesLayerPositionGroups = true;
        break;
      }
    }
  }

  if (usesLayerPositionGroups) {
    llvm::SmallVector<std::string> shapeSeq;
    for (auto arg : analysis.originalFuncOp.getArguments()) {
      auto tt = dyn_cast<RankedTensorType>(arg.getType());
      shapeSeq.push_back(tt ? getShapeKey(tt) : "_non_tensor_");
    }
    analysis.repeatPattern = detectRepeatPeriod(shapeSeq);
  }

  if (baseConfigs.empty() && !analysis.perArgOptions.empty()) {
    if (usesLayerPositionGroups && analysis.repeatPattern) {
      analysis.subgraphTemplate = matchSubgraphTemplate(
          *analysis.repeatPattern, analysis.shapeClasses);
    }

    if (analysis.subgraphTemplate) {
      analysis.strategy = SearchStrategy::SubgraphWise;
      analysis.strategyLabel =
          "subgraph-wise exhaustive (template: " +
          analysis.subgraphTemplate->archName + ")";
    } else {
      analysis.strategy = SearchStrategy::Hierarchical;
      if (usesLayerPositionGroups) {
        analysis.strategyLabel =
            "hierarchical over layer-position groups (greedy + pairwise)";
      } else if (!analysis.shapeClasses.empty()) {
        analysis.strategyLabel =
            "hierarchical over shape classes (greedy + pairwise)";
      } else {
        analysis.strategyLabel = "hierarchical (greedy + pairwise)";
      }
    }
    llvm::errs() << "AutoSharding: using " << analysis.strategyLabel << "\n";
  } else if (baseConfigs.empty()) {
    rootModule.emitWarning("AutoSharding: no configs enumerated, skipping");
    return std::nullopt;
  } else {
    if (usesLayerPositionGroups) {
      analysis.strategy = SearchStrategy::LayerPositionExhaustive;
      analysis.strategyLabel = "layer-position exhaustive";
    } else if (!analysis.shapeClasses.empty()) {
      analysis.strategy = SearchStrategy::ShapeClassExhaustive;
      analysis.strategyLabel = "shape-class exhaustive";
    } else {
      analysis.strategy = SearchStrategy::Exhaustive;
      analysis.strategyLabel = "exhaustive";
    }
    llvm::errs() << "AutoSharding: " << baseConfigs.size()
                 << " base configs (" << analysis.strategyLabel << ")\n";
  }

  // Phase 4: Constraint collection and config expansion.
  analysis.candidates =
      collectConstraintCandidates(rootModule, maxConstraintCandidates);
  llvm::errs() << "AutoSharding: " << analysis.candidates.size()
               << " constraint candidate(s)\n";

  if (analysis.strategy != SearchStrategy::Hierarchical &&
      analysis.strategy != SearchStrategy::SubgraphWise) {
    analysis.configs =
        expandWithConstraints(baseConfigs, analysis.candidates, numAxes);

    llvm::errs() << "AutoSharding: evaluating " << analysis.configs.size()
                 << " total configurations (base x constraints)\n";
  }

  // Phase 5: Dump setup and manual reference path.
  if (!dumpDir.empty()) {
    analysis.dumpRoot = createDumpRoot(dumpDir);
    if (!analysis.dumpRoot.empty()) {
      llvm::errs() << "AutoSharding: dumping to " << analysis.dumpRoot << "\n";
    }
  }

  analysis.manualRefPath = manualRef.str();

  return analysis;
}

// Evaluate all sharding configs by cloning the module, applying hints, running
// the remaining StableHLO passes, and scoring the resulting CCL ops.
// Returns the search result with the best config, or nullopt if all failed.
static std::optional<SearchResult>
evaluateConfigs(ModuleOp rootModule, const AnalysisResult &analysis,
                bool shouldDumpVariants) {
  MLIRContext *context = rootModule.getContext();

  double bestCost = std::numeric_limits<double>::infinity();
  size_t bestIdx = 0;
  bool anySucceeded = false;

  llvm::SmallVector<VariantResult> results;
  bool collectResults = !analysis.dumpRoot.empty();

  for (size_t i = 0; i < analysis.configs.size(); ++i) {
    ModuleOp clonedModule = cast<ModuleOp>(rootModule->clone());
    auto cleanup =
        llvm::make_scope_exit([&clonedModule] { clonedModule->erase(); });

    stripMarkArgumentCalls(clonedModule);
    applyShardingHints(clonedModule, analysis.configs[i],
                       analysis.meshInfo.meshName, analysis.shardableAxisNames,
                       analysis.candidates);

    std::string variantDir;
    if (shouldDumpVariants && collectResults) {
      std::string varName = configDirName(i, analysis.configs[i]);
      llvm::SmallString<256> vdir(analysis.dumpRoot);
      llvm::sys::path::append(vdir, varName);
      llvm::sys::fs::create_directories(vdir);
      variantDir = vdir.str().str();

      llvm::SmallString<256> hloPath(vdir);
      llvm::sys::path::append(hloPath, "01_stablehlo_with_hints.mlir");
      dumpModuleToFile(clonedModule, hloPath);
    }

    PassManager pm(context, ModuleOp::getOperationName(),
                   PassManager::Nesting::Implicit);
    addRemainingStableHLOPasses(pm);

    if (failed(pm.run(clonedModule))) {
      llvm::errs() << "AutoSharding: config " << i << " "
                   << formatConfig(analysis.configs[i])
                   << " failed to lower\n";
      if (collectResults) {
        results.push_back(
            {i, formatConfig(analysis.configs[i]), false, 0.0, 0.0, 0.0});
      }
      continue;
    }

    if (!variantDir.empty()) {
      llvm::SmallString<256> cclPath(variantDir);
      llvm::sys::path::append(cclPath, "02_stablehlo_with_ccls.mlir");
      dumpModuleToFile(clonedModule, cclPath);
    }

    ShardingResult sr =
        evaluate(clonedModule, analysis.configs[i],
                 analysis.originalFuncOp, analysis.shardableAxisSizes);
    llvm::errs() << "AutoSharding: config " << i << " "
                 << formatConfig(analysis.configs[i])
                 << " comm=" << sr.communicationCost
                 << " benefit=" << sr.memoryBenefit << " net=" << sr.netCost
                 << "\n";

    if (collectResults) {
      results.push_back({i, formatConfig(analysis.configs[i]), true,
                         sr.netCost, sr.communicationCost, sr.memoryBenefit});
    }
    anySucceeded = true;
    if (sr.netCost < bestCost) {
      bestCost = sr.netCost;
      bestIdx = i;
    }
  }

  if (!anySucceeded) {
    return std::nullopt;
  }

  return SearchResult{bestIdx, bestCost, std::move(results)};
}

// Hierarchical search for when the search space is too large for exhaustive
// enumeration. Uses a two-phase strategy:
//
//   Phase 1 — Greedy Coordinate Descent: starting from the all-replicated
//     config, iterate over shape classes (or individual args if no classes).
//     For each class, try all options while holding others fixed; keep the
//     best. Repeat for up to kMaxIterations passes until convergence.
//     This finds a good baseline quickly but can miss interactions between
//     classes/args that change independently.
//
//   Phase 2 — Pairwise Refinement: take the classes/args that changed in
//     Phase 1 (plus the largest unchanged ones, up to kMaxInterestingClasses).
//     For each pair, try all option combinations jointly. This catches the
//     most important cross-class interactions without full combinatorial cost.
//
// Operates over shape classes when available (reduces the unit of search from
// individual args to groups), otherwise falls back to per-arg search.
static std::optional<SearchResult>
evaluateConfigsHierarchical(ModuleOp rootModule, AnalysisResult &analysis) {
  const auto &perArgOptions = analysis.perArgOptions;
  size_t numArgs = perArgOptions.size();
  bool useShapeClasses = !analysis.shapeClasses.empty();

  ShardingConfig bestConfig;
  for (size_t a = 0; a < numArgs; ++a) {
    bestConfig.argDimSharding.push_back(perArgOptions[a][0]);
  }

  llvm::SmallVector<ShardingConfig> allConfigs;
  llvm::SmallVector<VariantResult> allResults;
  size_t bestIdx = 0;
  double bestCost = std::numeric_limits<double>::infinity();

  auto tryConfig = [&](const ShardingConfig &config) -> bool {
    size_t idx = allConfigs.size();
    allConfigs.push_back(config);

    auto vr = evaluateSingleConfig(rootModule, analysis, idx, config);
    allResults.push_back(vr);

    llvm::errs() << "AutoSharding: eval " << idx;
    if (vr.succeeded) {
      llvm::errs() << " comm=" << vr.commCost << " benefit=" << vr.memBenefit
                   << " net=" << vr.cost << "\n";
    } else {
      llvm::errs() << " FAILED\n";
    }

    if (vr.succeeded && vr.cost < bestCost) {
      bestCost = vr.cost;
      bestIdx = idx;
      return true;
    }
    return false;
  };

  tryConfig(bestConfig);

  constexpr int kMaxIterations = 5;

  if (useShapeClasses) {
    const auto &classes = analysis.shapeClasses;
    size_t numClasses = classes.size();

    llvm::errs() << "AutoSharding: Phase 1 - Greedy Coordinate Descent over "
                 << numClasses << " shape classes\n";

    llvm::SmallVector<size_t> changedClasses;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
      bool improved = false;
      for (size_t ci = 0; ci < numClasses; ++ci) {
        auto currentSharding =
            bestConfig.argDimSharding[classes[ci].argIndices[0]];

        for (size_t optIdx = 0; optIdx < classes[ci].options.size();
             ++optIdx) {
          if (classes[ci].options[optIdx] == currentSharding) {
            continue;
          }

          ShardingConfig candidate = bestConfig;
          for (size_t argIdx : classes[ci].argIndices) {
            candidate.argDimSharding[argIdx] = classes[ci].options[optIdx];
          }
          tryConfig(candidate);
        }

        bestConfig = allConfigs[bestIdx];

        if (bestConfig.argDimSharding[classes[ci].argIndices[0]] !=
            currentSharding) {
          improved = true;
          if (llvm::find(changedClasses, ci) == changedClasses.end()) {
            changedClasses.push_back(ci);
          }
        }
      }

      llvm::errs() << "AutoSharding: Phase 1 iteration " << (iter + 1)
                   << " best cost=" << bestCost << "\n";
      if (!improved) {
        break;
      }
    }

    size_t phase1Evals = allConfigs.size();
    llvm::errs() << "AutoSharding: Phase 1 complete: " << phase1Evals
                 << " evaluations, best cost=" << bestCost << ", "
                 << changedClasses.size() << " classes changed\n";

    llvm::errs()
        << "AutoSharding: Phase 2 - Pairwise Refinement over shape classes\n";

    constexpr size_t kMaxInterestingClasses = 20;
    llvm::SmallVector<size_t> interestingClasses;

    for (auto idx : changedClasses) {
      interestingClasses.push_back(idx);
    }

    for (size_t ci = 0; ci < numClasses; ++ci) {
      if (interestingClasses.size() >= kMaxInterestingClasses) {
        break;
      }
      if (llvm::find(changedClasses, ci) != changedClasses.end()) {
        continue;
      }
      interestingClasses.push_back(ci);
    }

    llvm::errs() << "AutoSharding: Phase 2 interesting classes: "
                 << interestingClasses.size() << "\n";

    size_t phase2Evals = 0;
    for (size_t i = 0; i < interestingClasses.size(); ++i) {
      for (size_t j = i + 1; j < interestingClasses.size(); ++j) {
        size_t classA = interestingClasses[i];
        size_t classB = interestingClasses[j];

        for (size_t optA = 0; optA < classes[classA].options.size(); ++optA) {
          for (size_t optB = 0; optB < classes[classB].options.size();
               ++optB) {
            auto currentA =
                bestConfig.argDimSharding[classes[classA].argIndices[0]];
            auto currentB =
                bestConfig.argDimSharding[classes[classB].argIndices[0]];
            if (classes[classA].options[optA] == currentA &&
                classes[classB].options[optB] == currentB) {
              continue;
            }

            ShardingConfig candidate = bestConfig;
            for (size_t argIdx : classes[classA].argIndices) {
              candidate.argDimSharding[argIdx] =
                  classes[classA].options[optA];
            }
            for (size_t argIdx : classes[classB].argIndices) {
              candidate.argDimSharding[argIdx] =
                  classes[classB].options[optB];
            }

            if (tryConfig(candidate)) {
              bestConfig = allConfigs[bestIdx];
            }
            ++phase2Evals;
          }
        }
      }
    }

    llvm::errs() << "AutoSharding: Phase 2 complete: " << phase2Evals
                 << " pair evaluations, final best cost=" << bestCost << "\n";

  } else {
    llvm::errs() << "AutoSharding: Phase 1 - Greedy Coordinate Descent\n";

    llvm::SmallVector<size_t> changedArgs;

    for (int iter = 0; iter < kMaxIterations; ++iter) {
      bool improved = false;
      for (size_t argIdx = 0; argIdx < numArgs; ++argIdx) {
        if (perArgOptions[argIdx].size() <= 1) {
          continue;
        }

        auto currentArgSharding = bestConfig.argDimSharding[argIdx];

        for (size_t optIdx = 0; optIdx < perArgOptions[argIdx].size();
             ++optIdx) {
          if (perArgOptions[argIdx][optIdx] == currentArgSharding) {
            continue;
          }

          ShardingConfig candidate = bestConfig;
          candidate.argDimSharding[argIdx] = perArgOptions[argIdx][optIdx];
          tryConfig(candidate);
        }

        bestConfig = allConfigs[bestIdx];

        if (bestConfig.argDimSharding[argIdx] != currentArgSharding) {
          improved = true;
          if (llvm::find(changedArgs, argIdx) == changedArgs.end()) {
            changedArgs.push_back(argIdx);
          }
        }
      }

      llvm::errs() << "AutoSharding: Phase 1 iteration " << (iter + 1)
                   << " best cost=" << bestCost << "\n";
      if (!improved) {
        break;
      }
    }

    size_t phase1Evals = allConfigs.size();
    llvm::errs() << "AutoSharding: Phase 1 complete: " << phase1Evals
                 << " evaluations, best cost=" << bestCost << ", "
                 << changedArgs.size() << " args changed\n";

    llvm::errs() << "AutoSharding: Phase 2 - Pairwise Refinement\n";

    constexpr size_t kMaxInterestingArgs = 20;
    llvm::SmallVector<size_t> interestingArgs;

    for (auto idx : changedArgs) {
      interestingArgs.push_back(idx);
    }

    auto funcOps = rootModule.getOps<func::FuncOp>();
    func::FuncOp funcOp = *funcOps.begin();

    llvm::SmallVector<std::pair<int64_t, size_t>> argSizes;
    for (auto arg : funcOp.getArguments()) {
      auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType) {
        continue;
      }
      size_t aIdx = arg.getArgNumber();
      if (perArgOptions[aIdx].size() <= 1) {
        continue;
      }
      if (llvm::find(changedArgs, aIdx) != changedArgs.end()) {
        continue;
      }

      int64_t numElements = 1;
      for (auto dim : tensorType.getShape()) {
        numElements *= dim;
      }
      argSizes.push_back({numElements, aIdx});
    }

    llvm::sort(argSizes, [](const auto &a, const auto &b) {
      return a.first > b.first;
    });

    for (const auto &[size, idx] : argSizes) {
      if (interestingArgs.size() >= kMaxInterestingArgs) {
        break;
      }
      interestingArgs.push_back(idx);
    }

    llvm::errs() << "AutoSharding: Phase 2 interesting args: "
                 << interestingArgs.size() << "\n";

    size_t phase2Evals = 0;
    for (size_t i = 0; i < interestingArgs.size(); ++i) {
      for (size_t j = i + 1; j < interestingArgs.size(); ++j) {
        size_t argA = interestingArgs[i];
        size_t argB = interestingArgs[j];

        for (size_t optA = 0; optA < perArgOptions[argA].size(); ++optA) {
          for (size_t optB = 0; optB < perArgOptions[argB].size(); ++optB) {
            if (perArgOptions[argA][optA] ==
                    bestConfig.argDimSharding[argA] &&
                perArgOptions[argB][optB] ==
                    bestConfig.argDimSharding[argB]) {
              continue;
            }

            ShardingConfig candidate = bestConfig;
            candidate.argDimSharding[argA] = perArgOptions[argA][optA];
            candidate.argDimSharding[argB] = perArgOptions[argB][optB];

            if (tryConfig(candidate)) {
              bestConfig = allConfigs[bestIdx];
            }
            ++phase2Evals;
          }
        }
      }
    }

    llvm::errs() << "AutoSharding: Phase 2 complete: " << phase2Evals
                 << " pair evaluations, final best cost=" << bestCost << "\n";
  }

  llvm::errs() << "AutoSharding: Total evaluations: " << allConfigs.size()
               << "\n";

  analysis.configs = std::move(allConfigs);

  if (bestCost == std::numeric_limits<double>::infinity()) {
    return std::nullopt;
  }

  return SearchResult{bestIdx, bestCost, std::move(allResults)};
}

// Subgraph-wise search for models that match a known architecture template
// (e.g. llama). Decomposes the search by subgraph family:
//
//   1. Compute layer-local maxElements for cost normalization (so that a
//      single large embedding table doesn't dominate the cost across all
//      layers).
//
//   2. Map each template family (e.g. "attention", "mlp") to its
//      shape-class group indices. Groups not covered by any family are
//      collected into an "other" bucket.
//
//   3. For each family, exhaustively search the Cartesian product of its
//      group options (capped at kMaxExhaustiveConfigs). Each family search
//      is greedy: we lock in the best sharding for one family before moving
//      to the next, so inter-family interactions are only captured through
//      their shared cost evaluation.
//
// This strategy is effective for architectures with clearly separable
// subgraphs (attention vs MLP) where the optimal sharding for one subgraph
// rarely depends on the sharding of another.
static std::optional<SearchResult>
evaluateConfigsSubgraphWise(ModuleOp rootModule, AnalysisResult &analysis) {
  const auto &perArgOptions = analysis.perArgOptions;
  const auto &groups = analysis.shapeClasses;
  const auto &tmpl = *analysis.subgraphTemplate;
  size_t numArgs = perArgOptions.size();

  constexpr int64_t kMaxExhaustiveConfigs = 50000;

  int64_t layerMaxElements = 0;
  if (analysis.repeatPattern) {
    auto funcOps = rootModule.getOps<func::FuncOp>();
    if (!funcOps.empty()) {
      func::FuncOp funcOp = *funcOps.begin();
      size_t periodStart = analysis.repeatPattern->prefixLen;
      size_t periodEnd = periodStart + analysis.repeatPattern->period;
      for (size_t i = periodStart; i < periodEnd && i < numArgs; ++i) {
        if (auto tt = dyn_cast<RankedTensorType>(
                funcOp.getArgument(i).getType())) {
          layerMaxElements =
              std::max(layerMaxElements, tt.getNumElements());
        }
      }
    }
  }
  if (layerMaxElements <= 0) {
    layerMaxElements = computeMaxElements(analysis.originalFuncOp);
  }
  llvm::errs() << "AutoSharding: subgraph-wise using layer-local "
               << "maxElements=" << layerMaxElements << " (vs global="
               << computeMaxElements(analysis.originalFuncOp) << ")\n";

  ShardingConfig bestConfig;
  for (size_t a = 0; a < numArgs; ++a) {
    bestConfig.argDimSharding.push_back(perArgOptions[a][0]);
  }

  llvm::SmallVector<ShardingConfig> allConfigs;
  llvm::SmallVector<VariantResult> allResults;
  size_t bestIdx = 0;
  double bestCost = std::numeric_limits<double>::infinity();

  auto tryConfig = [&](const ShardingConfig &config) -> bool {
    size_t idx = allConfigs.size();
    allConfigs.push_back(config);

    auto vr = evaluateSingleConfig(rootModule, analysis, idx, config,
                                   layerMaxElements);
    allResults.push_back(vr);

    llvm::errs() << "AutoSharding: eval " << idx;
    if (vr.succeeded) {
      llvm::errs() << " comm=" << vr.commCost << " benefit=" << vr.memBenefit
                   << " net=" << vr.cost << "\n";
    } else {
      llvm::errs() << " FAILED\n";
    }

    if (vr.succeeded && vr.cost < bestCost) {
      bestCost = vr.cost;
      bestIdx = idx;
      return true;
    }
    return false;
  };

  tryConfig(bestConfig);

  auto findGroupForPosition = [&](size_t pos) -> std::optional<size_t> {
    std::string prefix = "layer_pos" + std::to_string(pos) + "_";
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      if (groups[gi].key.find(prefix) == 0) {
        return gi;
      }
    }
    return std::nullopt;
  };

  llvm::DenseSet<size_t> templatePositions;
  for (const auto &family : tmpl.families) {
    for (size_t pos : family.periodPositions) {
      templatePositions.insert(pos);
    }
  }

  llvm::SmallVector<size_t> otherGroupIndices;
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    bool coveredByFamily = false;
    for (const auto &family : tmpl.families) {
      for (size_t pos : family.periodPositions) {
        std::string prefix = "layer_pos" + std::to_string(pos) + "_";
        if (groups[gi].key.find(prefix) == 0) {
          coveredByFamily = true;
          break;
        }
      }
      if (coveredByFamily) {
        break;
      }
    }
    if (!coveredByFamily) {
      otherGroupIndices.push_back(gi);
    }
  }

  llvm::SmallVector<FamilySearch> familySearches;
  for (const auto &family : tmpl.families) {
    FamilySearch fs;
    fs.name = family.name;
    for (size_t pos : family.periodPositions) {
      auto gi = findGroupForPosition(pos);
      if (gi) {
        fs.groupIndices.push_back(*gi);
      }
    }
    if (!fs.groupIndices.empty()) {
      familySearches.push_back(std::move(fs));
    }
  }

  if (!otherGroupIndices.empty()) {
    FamilySearch fs;
    fs.name = "other";
    fs.groupIndices = otherGroupIndices;
    familySearches.push_back(std::move(fs));
  }

  for (const auto &fs : familySearches) {
    llvm::errs() << "AutoSharding: searching subgraph family '" << fs.name
                 << "' (" << fs.groupIndices.size() << " groups)\n";

    llvm::SmallVector<llvm::SmallVector<TensorShardOption>>
        familyGroupOptions;
    for (size_t gi : fs.groupIndices) {
      familyGroupOptions.push_back(groups[gi].options);
      llvm::errs() << "  group '" << groups[gi].key << "': "
                   << groups[gi].options.size() << " options, "
                   << groups[gi].argIndices.size() << " args\n";
    }

    int64_t familySpace = 1;
    bool familyOverflow = false;
    for (const auto &opts : familyGroupOptions) {
      familySpace *= static_cast<int64_t>(opts.size());
      if (familySpace > kMaxExhaustiveConfigs) {
        familyOverflow = true;
        break;
      }
    }

    if (familyOverflow) {
      llvm::errs() << "AutoSharding: family '" << fs.name
                   << "' search space too large (>" << kMaxExhaustiveConfigs
                   << "), using greedy within family\n";

      constexpr int kMaxIterations = 5;
      for (int iter = 0; iter < kMaxIterations; ++iter) {
        bool improved = false;
        for (size_t fi = 0; fi < fs.groupIndices.size(); ++fi) {
          size_t gi = fs.groupIndices[fi];
          auto currentSharding =
              bestConfig.argDimSharding[groups[gi].argIndices[0]];

          for (size_t optIdx = 0; optIdx < groups[gi].options.size();
               ++optIdx) {
            if (groups[gi].options[optIdx] == currentSharding) {
              continue;
            }

            ShardingConfig candidate = bestConfig;
            for (size_t argIdx : groups[gi].argIndices) {
              candidate.argDimSharding[argIdx] = groups[gi].options[optIdx];
            }
            tryConfig(candidate);
          }

          bestConfig = allConfigs[bestIdx];
          if (bestConfig.argDimSharding[groups[gi].argIndices[0]] !=
              currentSharding) {
            improved = true;
          }
        }
        if (!improved) {
          break;
        }
      }
    } else {
      llvm::errs() << "AutoSharding: family '" << fs.name
                   << "' search space = " << familySpace
                   << " configs, using exhaustive\n";

      auto familyConfigs = cartesianProduct(familyGroupOptions);
      for (const auto &fc : familyConfigs) {
        ShardingConfig candidate = bestConfig;

        for (size_t fi = 0; fi < fs.groupIndices.size(); ++fi) {
          size_t gi = fs.groupIndices[fi];
          for (size_t argIdx : groups[gi].argIndices) {
            candidate.argDimSharding[argIdx] = fc.argDimSharding[fi];
          }
        }

        tryConfig(candidate);
      }

      bestConfig = allConfigs[bestIdx];
    }

    llvm::errs() << "AutoSharding: family '" << fs.name
                 << "' done, best cost=" << bestCost << "\n";
  }

  llvm::errs() << "AutoSharding: subgraph-wise search complete, "
               << allConfigs.size() << " total evaluations, best cost="
               << bestCost << "\n";

  analysis.configs = std::move(allConfigs);

  if (bestCost == std::numeric_limits<double>::infinity()) {
    return std::nullopt;
  }

  return SearchResult{bestIdx, bestCost, std::move(allResults)};
}

// Write a human-readable summary table of all variant results.
static void writeSummary(const AnalysisResult &analysis,
                         const SearchResult &search) {
  llvm::SmallString<256> summaryPath(analysis.dumpRoot);
  llvm::sys::path::append(summaryPath, "summary.txt");
  std::error_code ec;
  llvm::raw_fd_ostream fos(summaryPath, ec);
  if (ec) {
    return;
  }

  // --- Section 1: Header metadata (mesh, strategy, grouping info) ---
  fos << "Auto Sharding Summary\n";
  fos << "=======================\n";
  fos << "Mesh: " << analysis.meshInfo.meshName << "\n";
  fos << "Shardable axes:";
  for (size_t i = 0; i < analysis.shardableAxisNames.size(); ++i) {
    fos << " " << analysis.shardableAxisNames[i]
        << " (size=" << analysis.shardableAxisSizes[i] << ")";
  }
  fos << "\n";
  fos << "Search strategy: " << analysis.strategyLabel << "\n";

  if (!analysis.prunedArgs.empty()) {
    fos << "Pruned args (too small to shard): ";
    for (size_t i = 0; i < analysis.prunedArgs.size(); ++i) {
      if (i > 0) {
        fos << ", ";
      }
      fos << "arg" << analysis.prunedArgs[i];
    }
    fos << "\n";
  }

  // --- Section 2: Shape class / layer-position group table ---
  if (!analysis.shapeClasses.empty()) {
    if (analysis.strategy == SearchStrategy::LayerPositionExhaustive ||
        analysis.strategy == SearchStrategy::SubgraphWise) {
      fos << "Layer-position groups: " << analysis.shapeClasses.size();
      if (analysis.repeatPattern) {
        fos << " (prefix=" << analysis.repeatPattern->prefixLen
            << ", period=" << analysis.repeatPattern->period
            << ", repeats=" << analysis.repeatPattern->numRepeats
            << ", suffix=" << analysis.repeatPattern->suffixLen << ")";
      }
      fos << "\n";
    } else {
      fos << "Shape classes: " << analysis.shapeClasses.size() << "\n";
    }
    for (const auto &sc : analysis.shapeClasses) {
      fos << "  '" << sc.key << "' -> " << sc.argIndices.size() << " args, "
          << sc.options.size() << " options\n";
    }
  }

  if (analysis.subgraphTemplate) {
    fos << "Subgraph template: " << analysis.subgraphTemplate->archName
        << " (period=" << analysis.subgraphTemplate->period << ")\n";
    for (const auto &family : analysis.subgraphTemplate->families) {
      fos << "  family '" << family.name << "': positions [";
      for (size_t i = 0; i < family.periodPositions.size(); ++i) {
        if (i > 0) {
          fos << ", ";
        }
        fos << family.periodPositions[i];
      }
      fos << "]\n";
    }
  }

  if (analysis.strategy != SearchStrategy::Hierarchical &&
      analysis.strategy != SearchStrategy::SubgraphWise) {
    fos << "Base configs: " << analysis.numBaseConfigs << "\n";
    fos << "Constraint candidates: " << analysis.candidates.size() << "\n";
  }
  fos << "Total configs evaluated: " << search.results.size() << "\n";
  fos << "Cost model: per-CCL latency + volume-weighted bandwidth + "
      << "critical-path penalty + output-gather cost, "
      << "parameter multiplier=3.0"
      << " compute-benefit-weight=1.0\n\n";

  // --- Section 3: Config results table (one row per evaluated variant) ---
  fos << "Config  Sharding" << std::string(50, ' ')
      << "Status  Comm      Benefit   Net\n";
  fos << std::string(110, '-') << "\n";
  for (const auto &r : search.results) {
    fos << llvm::format("%-8zu", r.idx);

    fos << r.label;
    if (r.label.size() < 58) {
      fos.indent(58 - r.label.size());
    }

    if (r.succeeded) {
      fos << "OK      " << llvm::format("%-10.3f", r.commCost)
          << llvm::format("%-10.3f", r.memBenefit)
          << llvm::format("%.3f", r.cost);
    } else {
      fos << "FAILED  -";
    }
    if (r.succeeded && r.idx == search.bestIdx) {
      fos << "  <-- WINNER";
    }
    fos << "\n";
  }

  // --- Section 4: Winner details ---
  fos << "\nSelected: config " << search.bestIdx << " "
      << formatConfig(analysis.configs[search.bestIdx])
      << " with net cost=" << llvm::format("%.3f", search.bestCost) << "\n";

  // --- Section 5: Per-position comparison (SubgraphWise only) ---
  // For SubgraphWise results, print the winning sharding for each layer
  // position alongside the manual reference (if provided), so the user
  // can quickly see where auto and manual shardings differ.
  if (analysis.strategy == SearchStrategy::SubgraphWise &&
      analysis.repeatPattern && analysis.subgraphTemplate) {
    const auto &winConfig = analysis.configs[search.bestIdx];
    const auto &rp = *analysis.repeatPattern;
    const auto &tmplRef = *analysis.subgraphTemplate;
    const auto &grps = analysis.shapeClasses;

    llvm::SmallVector<TensorShardOption> manualShardings;
    bool hasManual = false;
    if (!analysis.manualRefPath.empty()) {
      manualShardings = parseManualShardings(analysis.manualRefPath,
                                             analysis.shardableAxisNames);
      hasManual = !manualShardings.empty();
    }

    fos << "\nWinner Sharding by Layer Position (period=" << rp.period
        << ")\n";
    fos << std::string(80, '=') << "\n";
    if (hasManual) {
      fos << "  pos  group                        auto        "
             "manual      family\n";
      fos << std::string(80, '-') << "\n";
    }

    auto formatSharding = [](const TensorShardOption &dims) {
      std::string s = "[";
      for (size_t d = 0; d < dims.size(); ++d) {
        if (d > 0) {
          s += ",";
        }
        if (dims[d].empty()) {
          s += "R";
        } else {
          s += "S";
          for (size_t axisIdx : dims[d]) {
            s += std::to_string(axisIdx);
          }
        }
      }
      s += "]";
      return s;
    };

    auto findGroupForPos2 = [&](size_t pos) -> std::optional<size_t> {
      std::string pfx = "layer_pos" + std::to_string(pos) + "_";
      for (size_t gi = 0; gi < grps.size(); ++gi) {
        if (grps[gi].key.find(pfx) == 0) {
          return gi;
        }
      }
      return std::nullopt;
    };

    auto getFamilyForPos = [&](size_t pos) -> std::string {
      for (const auto &fam : tmplRef.families) {
        for (size_t fi = 0; fi < fam.periodPositions.size(); ++fi) {
          if (fam.periodPositions[fi] == pos) {
            return fam.name;
          }
        }
      }
      return "other";
    };

    for (size_t pos = 0; pos < rp.period; ++pos) {
      auto gi = findGroupForPos2(pos);
      std::string famName = getFamilyForPos(pos);

      fos << "  pos " << llvm::format("%-3zu", pos);
      if (gi) {
        size_t repArgIdx = grps[*gi].argIndices[0];
        std::string autoSharding =
            formatSharding(winConfig.argDimSharding[repArgIdx]);
        fos << llvm::format("%-28s", grps[*gi].key.c_str())
            << llvm::format("%-12s", autoSharding.c_str());

        if (hasManual) {
          size_t manualArgIdx = rp.prefixLen + pos;
          std::string manualShardingStr =
              (manualArgIdx < manualShardings.size())
                  ? formatSharding(manualShardings[manualArgIdx])
                  : "?";
          bool match = (manualArgIdx < manualShardings.size()) &&
                       (winConfig.argDimSharding[repArgIdx] ==
                        manualShardings[manualArgIdx]);
          fos << llvm::format("%-12s", manualShardingStr.c_str())
              << (match ? " " : "*") << "(" << famName << ")";
        } else {
          fos << "(" << famName << ")";
        }
      } else {
        fos << "(pruned)";
        if (hasManual) {
          size_t manualArgIdx = rp.prefixLen + pos;
          if (manualArgIdx < manualShardings.size()) {
            std::string manualShardingStr =
                formatSharding(manualShardings[manualArgIdx]);
            fos << std::string(28, ' ') << std::string(12, ' ')
                << llvm::format("%-12s", manualShardingStr.c_str());
          }
        }
      }
      fos << "\n";
    }

    if (rp.prefixLen > 0 || rp.suffixLen > 0) {
      fos << "\n  Prefix/Suffix args:\n";
      for (size_t gi = 0; gi < grps.size(); ++gi) {
        bool isLayerGroup = grps[gi].key.find("layer_pos") == 0;
        if (!isLayerGroup && !grps[gi].argIndices.empty()) {
          size_t argIdx = grps[gi].argIndices[0];
          std::string autoSharding =
              formatSharding(winConfig.argDimSharding[argIdx]);
          fos << "  " << llvm::format("%-34s", grps[gi].key.c_str())
              << llvm::format("%-12s", autoSharding.c_str());

          if (hasManual && argIdx < manualShardings.size()) {
            std::string manualShardingStr =
                formatSharding(manualShardings[argIdx]);
            bool match =
                (winConfig.argDimSharding[argIdx] == manualShardings[argIdx]);
            fos << llvm::format("%-12s", manualShardingStr.c_str())
                << (match ? " " : "*");
          }
          fos << "(" << grps[gi].argIndices.size() << " args)\n";
        }
      }
    }

    if (hasManual) {
      fos << "\n  (* = auto differs from manual)\n";
    }
  }
}

// Apply the winning sharding config to the original module and optionally
// dump the winning MLIR + summary.
static void applyBestConfig(ModuleOp rootModule, const AnalysisResult &analysis,
                            const SearchResult &search) {
  MLIRContext *context = rootModule.getContext();

  LLVM_DEBUG(llvm::dbgs() << "AutoSharding: selected config " << search.bestIdx
                          << " "
                          << formatConfig(analysis.configs[search.bestIdx])
                          << " with net cost=" << search.bestCost << "\n");
  llvm::errs() << "AutoSharding: selected config " << search.bestIdx << " "
               << formatConfig(analysis.configs[search.bestIdx])
               << " with net cost=" << search.bestCost << "\n";
  applyShardingHints(rootModule, analysis.configs[search.bestIdx],
                     analysis.meshInfo.meshName, analysis.shardableAxisNames,
                     analysis.candidates);

  if (!analysis.dumpRoot.empty()) {
    llvm::SmallString<256> winnerHintsPath(analysis.dumpRoot);
    llvm::sys::path::append(winnerHintsPath,
                            "winner_stablehlo_with_hints.mlir");
    dumpModuleToFile(rootModule, winnerHintsPath);

    ModuleOp winnerModule = cast<ModuleOp>(rootModule->clone());
    stripMarkArgumentCalls(winnerModule);
    PassManager winnerPM(context, ModuleOp::getOperationName(),
                         PassManager::Nesting::Implicit);
    addRemainingStableHLOPasses(winnerPM);
    if (succeeded(winnerPM.run(winnerModule))) {
      llvm::SmallString<256> winnerCCLPath(analysis.dumpRoot);
      llvm::sys::path::append(winnerCCLPath,
                              "winner_stablehlo_with_ccls.mlir");
      dumpModuleToFile(winnerModule, winnerCCLPath);
    }
    winnerModule->erase();

    writeSummary(analysis, search);
  }
}

//===----------------------------------------------------------------------===//
// AutoShardingPass — thin wrapper that validates, analyzes, searches, applies.
//===----------------------------------------------------------------------===//

class AutoShardingPass : public impl::AutoShardingPassBase<AutoShardingPass> {
public:
  using impl::AutoShardingPassBase<AutoShardingPass>::AutoShardingPassBase;

  void runOnOperation() final {
    ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();

    context->exitMultiThreadedExecution();
    auto restoreGuard = llvm::make_scope_exit(
        [context] { context->enterMultiThreadedExecution(); });

    auto analysisOpt =
        analyzeModule(rootModule, maxConstraintCandidates, dumpDir, manualRef);
    if (!analysisOpt) {
      return;
    }
    auto &analysis = *analysisOpt;

    std::optional<SearchResult> searchResult;
    switch (analysis.strategy) {
    case SearchStrategy::SubgraphWise:
      searchResult = evaluateConfigsSubgraphWise(rootModule, analysis);
      break;
    case SearchStrategy::Hierarchical:
      searchResult = evaluateConfigsHierarchical(rootModule, analysis);
      break;
    case SearchStrategy::Exhaustive:
    case SearchStrategy::ShapeClassExhaustive:
    case SearchStrategy::LayerPositionExhaustive:
      searchResult = evaluateConfigs(rootModule, analysis, dumpVariants);
      break;
    }

    if (!searchResult) {
      rootModule.emitError(
          "AutoSharding: all sharding configurations failed to lower");
      signalPassFailure();
      return;
    }

    applyBestConfig(rootModule, analysis, *searchResult);
  }
};

} // namespace
} // namespace mlir::tt::stablehlo
