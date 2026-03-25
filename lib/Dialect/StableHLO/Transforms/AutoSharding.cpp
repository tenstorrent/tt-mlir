// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/MemoryBuffer.h"

#include <limits>

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_AUTOSHARDINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper data structures.
//===----------------------------------------------------------------------===//

struct MeshInfo {
  llvm::StringRef meshName;
  llvm::SmallVector<std::pair<std::string, int64_t>> axes;

  llvm::SmallVector<std::string> getShardableAxes() const {
    llvm::SmallVector<std::string> results;
    for (const auto &[name, size] : axes) {
      if (size > 1) {
        results.push_back(name);
      }
    }
    return results;
  }
};

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

// Parse sdy.sharding attributes from a manual MLIR file for comparison.
// Each arg's sharding is extracted from #sdy.sharding<@mesh, [dims]> where
// {} = replicated, {"axis_name"} = sharded on that dimension.
static llvm::SmallVector<llvm::SmallVector<bool>>
parseManualShardings(const std::string &filePath) {
  llvm::SmallVector<llvm::SmallVector<bool>> result;

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
    llvm::SmallVector<bool> argSharding;

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
      argSharding.push_back(!dimContent.empty());
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

// Collect tensor-typed op results in the function body that are eligible for
// sdy.sharding_constraint insertion. Skips terminators and non-tensor results.
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

// Enumerate valid per-dim sharding options for a tensor of given rank.
// With a single shardable axis, at most one dim can be sharded on it.
// Returns: [all-replicated, dim0-sharded, dim1-sharded, ...].
static llvm::SmallVector<llvm::SmallVector<bool>>
enumerateValidDimShardings(int64_t rank) {
  llvm::SmallVector<llvm::SmallVector<bool>> options;

  options.push_back(llvm::SmallVector<bool>(rank, false));

  for (int64_t d = 0; d < rank; ++d) {
    llvm::SmallVector<bool> sharded(rank, false);
    sharded[d] = true;
    options.push_back(sharded);
  }
  return options;
}

// Enumerate valid per-arg sharding options with shape-aware pruning.
// For each argument, returns (rank+1) options (all-R + shard-one-dim) unless
// the tensor is too small to benefit from sharding, in which case only the
// all-replicated option is returned.
static llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
enumeratePerArgOptions(ModuleOp &module, llvm::SmallVector<size_t> &prunedArgs) {
  constexpr int64_t kMinShardableElements = 4096;
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>> perArgOptions;

  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return perArgOptions;
  }
  func::FuncOp funcOp = *funcOps.begin();

  for (auto arg : funcOp.getArguments()) {
    auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensorType) {
      perArgOptions.push_back({llvm::SmallVector<bool>()});
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
          {llvm::SmallVector<bool>(static_cast<size_t>(rank), false)});
    } else {
      perArgOptions.push_back(enumerateValidDimShardings(rank));
    }
  }

  return perArgOptions;
}

// Shape class: groups function arguments that share the same tensor shape
// and element type. In repeated transformer architectures, tensors with
// identical shapes play identical structural roles and share the same
// optimal sharding.
struct ShapeClass {
  std::string key;
  llvm::SmallVector<size_t> argIndices;
  llvm::SmallVector<llvm::SmallVector<bool>> options;
};

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

// Repeating layer pattern detected in the argument shape sequence.
struct RepeatPattern {
  size_t prefixLen;
  size_t period;
  size_t numRepeats;
  size_t suffixLen;
};

//===----------------------------------------------------------------------===//
// Subgraph template database.
//===----------------------------------------------------------------------===//

// A functional subgraph within a repeating layer block (e.g., attention, MLP).
// Positions refer to indices within the repeat period.
struct SubgraphFamily {
  std::string name;
  llvm::SmallVector<size_t> periodPositions;
};

// Architecture-specific template mapping repeat period to subgraph families.
struct SubgraphTemplate {
  std::string archName;
  size_t period;
  llvm::SmallVector<SubgraphFamily> families;
};

static llvm::SmallVector<SubgraphTemplate> getKnownTemplates() {
  llvm::SmallVector<SubgraphTemplate> templates;

  // Llama-family: period=11
  //   pos 0: rotation cos  (1x8x128x128)
  //   pos 1: k_proj         (1024x3072)
  //   pos 2: rotation sin  (1x8x128x128)
  //   pos 3: v_proj         (1024x3072)
  //   pos 4: rms_norm       (3072) -- pruned
  //   pos 5: gate_proj      (3072x8192)
  //   pos 6: down_proj      (8192x3072)
  //   pos 7: rms_norm       (3072) -- pruned
  //   pos 8: q_proj         (3072x3072)
  //   pos 9: o_proj         (3072x3072)
  //   pos 10: up_proj       (8192x3072)
  SubgraphTemplate llama;
  llama.archName = "llama";
  llama.period = 11;
  llama.families.push_back({"attention", {0, 1, 2, 3, 8, 9}});
  llama.families.push_back({"mlp", {5, 6, 10}});
  templates.push_back(std::move(llama));

  return templates;
}

// Match a detected repeat pattern against the known template database.
// Returns the matched template if the period matches and the template has
// at least one family with >= 2 shardable groups.
static std::optional<SubgraphTemplate>
matchSubgraphTemplate(const RepeatPattern &pattern,
                      const llvm::SmallVector<ShapeClass> &groups) {
  auto templates = getKnownTemplates();
  for (auto &tmpl : templates) {
    if (tmpl.period != pattern.period) {
      continue;
    }

    // Verify at least one family has >= 2 shardable groups present.
    bool hasViableFamily = false;
    for (const auto &family : tmpl.families) {
      size_t shardableCount = 0;
      for (size_t pos : family.periodPositions) {
        std::string groupPrefix =
            "layer_pos" + std::to_string(pos) + "_";
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

// Auto-detect a repeating period in the sequence of argument shapes.
// Transformers have a fixed block of args per layer that repeats N times,
// preceded by a non-repeating prefix (embeddings, position encodings, etc.).
// Returns the smallest (prefix, period) pair with >= kMinRepeats repetitions.
static std::optional<RepeatPattern>
detectRepeatPeriod(const llvm::SmallVector<std::string> &shapes) {
  size_t N = shapes.size();
  constexpr size_t kMinRepeats = 3;

  for (size_t P = 1; P <= N / kMinRepeats; ++P) {
    for (size_t prefix = 0; prefix + kMinRepeats * P <= N; ++prefix) {
      // Try suffix=0 (exact), then suffix=P (last block may differ, e.g.
      // lm_head replacing a layer weight in the final transformer block).
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

// Build groups based on position within the repeating layer pattern.
// Each position in the repeat unit becomes a group containing all args
// at that position across every layer. Prefix args form individual groups.
// This allows same-shaped tensors at different positions (e.g., Q-proj vs
// O-proj) to receive different shardings.
static llvm::SmallVector<ShapeClass> buildLayerPositionGroups(
    ModuleOp &module,
    const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
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
    auto tensorType = dyn_cast<RankedTensorType>(funcOp.getArgument(i).getType());
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

  // Suffix args: fold into matching layer-position groups when the shape
  // matches the corresponding position in the repeat unit.  Only create
  // individual groups for mismatched args (e.g., lm_head replacing a layer
  // weight in the final block).
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

static llvm::SmallVector<ShapeClass> buildShapeClasses(
    ModuleOp &module,
    const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
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
// per-arg ShardingConfig. Each arg gets the sharding of its shape class;
// pruned/non-tensor args get all-replicated.
static ShardingConfig expandShapeClassConfig(
    const ShardingConfig &classConfig,
    const llvm::SmallVector<ShapeClass> &shapeClasses,
    const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
        &perArgOptions) {
  ShardingConfig config;
  size_t numArgs = perArgOptions.size();

  for (size_t a = 0; a < numArgs; ++a) {
    config.argDimSharded.push_back(perArgOptions[a][0]);
  }

  for (size_t c = 0; c < shapeClasses.size(); ++c) {
    for (size_t argIdx : shapeClasses[c].argIndices) {
      config.argDimSharded[argIdx] = classConfig.argDimSharded[c];
    }
  }

  return config;
}

// Recursive helper for Cartesian product of per-arg sharding options.
static void cartesianProductHelper(
    const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
        &perArgOptions,
    size_t argIdx, ShardingConfig &current,
    llvm::SmallVector<ShardingConfig> &result) {
  if (argIdx == perArgOptions.size()) {
    result.push_back(current);
    return;
  }
  for (const auto &option : perArgOptions[argIdx]) {
    current.argDimSharded.push_back(option);
    cartesianProductHelper(perArgOptions, argIdx + 1, current, result);
    current.argDimSharded.pop_back();
  }
}

static llvm::SmallVector<ShardingConfig> cartesianProduct(
    const llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
        &perArgOptions) {
  llvm::SmallVector<ShardingConfig> result;
  ShardingConfig current;
  cartesianProductHelper(perArgOptions, 0, current, result);
  return result;
}

// Tier 1: enumerate arg-level sharding configs.
// Uses proper Cartesian product of per-arg options (with shape-aware pruning).
// When the per-arg space is too large, tries shape-class grouping: args with
// identical tensor shapes are assigned the same sharding, collapsing the
// search space dramatically for repeated transformer architectures.
// Returns empty if even the shape-class space exceeds the budget, signaling
// the caller to use hierarchical search.
static llvm::SmallVector<ShardingConfig> enumerateTier1Configs(
    ModuleOp &module,
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
        &perArgOptions,
    llvm::SmallVector<size_t> &prunedArgs,
    llvm::SmallVector<ShapeClass> &shapeClasses) {
  perArgOptions = enumeratePerArgOptions(module, prunedArgs);
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

  // Per-arg space too large; try layer-position grouping first, then
  // fall back to shape-class grouping.
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
        llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
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

  // No repeating pattern found; fall back to shape-class grouping.
  shapeClasses = buildShapeClasses(module, perArgOptions);
  if (shapeClasses.empty()) {
    llvm::errs() << "AutoSharding: no shape classes found, "
                 << "will use hierarchical search\n";
    return {};
  }

  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>> perClassOptions;
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

// Cross-product Tier 1 configs with Tier 2 constraint choices.
// Per constraint candidate: absent OR each valid dim sharding.
static llvm::SmallVector<ShardingConfig> expandWithConstraints(
    const llvm::SmallVector<ShardingConfig> &tier1Configs,
    const llvm::SmallVector<ConstraintCandidate> &candidates) {
  if (candidates.empty()) {
    return tier1Configs;
  }

  // Valid sharding options per candidate (not including "absent").
  llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
      perCandidateOptions;
  for (const auto &cand : candidates) {
    perCandidateOptions.push_back(enumerateValidDimShardings(cand.rank));
  }

  // Total constraint combinations: product of (1 + numValidOptions) per
  // candidate, where the +1 accounts for "absent" (no constraint).
  size_t constraintCombinations = 1;
  for (const auto &opts : perCandidateOptions) {
    constraintCombinations *= (opts.size() + 1);
  }

  size_t totalConfigs = tier1Configs.size() * constraintCombinations;
  constexpr size_t kMaxConfigs = 50000;
  if (totalConfigs > kMaxConfigs) {
    llvm::errs() << "AutoSharding: Tier1*Tier2 would produce " << totalConfigs
                 << " configs (cap=" << kMaxConfigs
                 << "), skipping Tier 2 constraints\n";
    return tier1Configs;
  }

  llvm::errs() << "AutoSharding: Tier 2 expands " << tier1Configs.size()
               << " Tier 1 configs x " << constraintCombinations
               << " constraint combos = " << totalConfigs << " total\n";

  llvm::SmallVector<ShardingConfig> combined;
  combined.reserve(totalConfigs);

  for (const auto &t1 : tier1Configs) {
    for (size_t ci = 0; ci < constraintCombinations; ++ci) {
      ShardingConfig config;
      config.argDimSharded = t1.argDimSharded;

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

// Apply Tier 1 arg shardings and Tier 2 sharding constraints to a module.
static void
applyShardingHints(ModuleOp module, const ShardingConfig &config,
                   StringRef meshName, StringRef shardAxisName,
                   const llvm::SmallVector<ConstraintCandidate> &candidates) {
  MLIRContext *context = module.getContext();

  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return;
  }
  func::FuncOp funcOp = *funcOps.begin();

  // --- Tier 1: set sdy.sharding on function arguments ---
  constexpr bool isClosed = false;
  for (size_t argIdx = 0; argIdx < config.argDimSharded.size(); ++argIdx) {
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    for (bool sharded : config.argDimSharded[argIdx]) {
      if (sharded) {
        auto axisRef = mlir::sdy::AxisRefAttr::get(context, shardAxisName);
        dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
            context, {axisRef}, isClosed));
      } else {
        dimShardings.push_back(
            mlir::sdy::DimensionShardingAttr::get(context, {}, isClosed));
      }
    }

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

  // --- Tier 2: insert sdy.sharding_constraint on intermediates ---
  if (config.constraintTargets.empty() || candidates.empty()) {
    return;
  }

  // Snapshot body ops by index for positional lookup in the clone.
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

    // Build TensorShardingAttr for the constraint target.
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
    for (bool sharded : target) {
      if (sharded) {
        auto axisRef = mlir::sdy::AxisRefAttr::get(context, shardAxisName);
        dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
            context, {axisRef}, isClosed));
      } else {
        dimShardings.push_back(
            mlir::sdy::DimensionShardingAttr::get(context, {}, isClosed));
      }
    }

    auto sharding = mlir::sdy::TensorShardingAttr::get(
        context, meshName, dimShardings, /*replicatedAxes=*/{},
        /*unreducedAxes=*/{});

    // Snapshot existing uses before creating the constraint op, so we only
    // redirect pre-existing uses (not the constraint's own operand).
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

// Remove stablehlo.custom_call @tt.mark_argument ops from a module.
// These tt-xla-specific identity ops block Shardy from propagating shardings
// through the graph, preventing CCL insertion during cost evaluation.
// Each call is replaced by forwarding its input directly to all users.
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

// Build a sub-pipeline with the remaining StableHLO passes (everything that
// normally runs after AutoSharding in the stablehlo-pipeline).
static void addRemainingStableHLOPasses(OpPassManager &pm) {
  pm.addPass(createDecoupleConstFanoutPass());
  pm.addPass(createDecomposeCustomCallTuplesPass());
  pm.addPass(createFlattenCompositePass());
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

static std::string formatConfig(const ShardingConfig &config) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "[";
  for (size_t a = 0; a < config.argDimSharded.size(); ++a) {
    if (a > 0) {
      os << ", ";
    }
    os << "arg" << a << ":[";
    for (size_t d = 0; d < config.argDimSharded[a].size(); ++d) {
      if (d > 0) {
        os << ",";
      }
      os << (config.argDimSharded[a][d] ? "S" : "R");
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
        os << ((*config.constraintTargets[c])[d] ? "S" : "R");
      }
      os << "]";
    }
  }
  os << "]";
  return result;
}

static std::string configDirName(size_t idx, const ShardingConfig &config) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << llvm::format("variant_%02zu", idx);
  for (size_t a = 0; a < config.argDimSharded.size(); ++a) {
    os << "_arg" << a << "-";
    for (bool s : config.argDimSharded[a]) {
      os << (s ? "S" : "R");
    }
  }
  for (size_t c = 0; c < config.constraintTargets.size(); ++c) {
    os << "_c" << c << "-";
    if (!config.constraintTargets[c]) {
      os << "none";
    } else {
      for (bool s : *config.constraintTargets[c]) {
        os << (s ? "S" : "R");
      }
    }
  }
  return result;
}

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
// AutoShardingPass implementation.
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

    auto analysisOpt = analyzeModule(rootModule);
    if (!analysisOpt) {
      return;
    }
    auto &analysis = *analysisOpt;

    std::optional<SearchResult> searchResult;
    if (analysis.useSubgraphWise) {
      searchResult = evaluateConfigsSubgraphWise(rootModule, analysis);
    } else if (analysis.useHierarchical) {
      searchResult = evaluateConfigsHierarchical(rootModule, analysis);
    } else {
      searchResult = evaluateConfigs(rootModule, analysis);
    }
    if (!searchResult) {
      rootModule.emitError(
          "AutoSharding: all sharding configurations failed to lower");
      signalPassFailure();
      return;
    }

    applyBestConfig(rootModule, analysis, *searchResult);
  }

private:
  struct AnalysisResult {
    MeshInfo meshInfo;
    std::string shardAxisName;
    int64_t meshAxisSize;
    func::FuncOp originalFuncOp;
    llvm::SmallVector<ShardingConfig> configs;
    llvm::SmallVector<ShardingConfig> tier1Configs;
    llvm::SmallVector<ConstraintCandidate> candidates;
    std::string dumpRoot;

    // Hierarchical / shape-class / layer-position search fields.
    bool useHierarchical = false;
    bool useSubgraphWise = false;
    bool usesLayerPositionGroups = false;
    std::optional<RepeatPattern> repeatPattern;
    std::optional<SubgraphTemplate> subgraphTemplate;
    llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>> perArgOptions;
    llvm::SmallVector<size_t> prunedArgs;
    llvm::SmallVector<ShapeClass> shapeClasses;
    std::string searchStrategy;
    std::string manualRefPath;
  };

  struct VariantResult {
    size_t idx;
    std::string label;
    bool succeeded;
    double cost;
    double commCost;
    double memBenefit;
  };

  struct SearchResult {
    size_t bestIdx;
    double bestCost;
    llvm::SmallVector<VariantResult> results;
  };

  struct EvalResult {
    bool succeeded;
    double netCost;
    double commCost;
    double memBenefit;
  };

  EvalResult evaluateSingleConfig(ModuleOp rootModule,
                                  const AnalysisResult &analysis,
                                  const ShardingConfig &config,
                                  int64_t maxElementsOverride = 0) {
    MLIRContext *context = rootModule.getContext();
    ModuleOp clonedModule = cast<ModuleOp>(rootModule->clone());
    auto cleanup =
        llvm::make_scope_exit([&clonedModule] { clonedModule->erase(); });

    stripMarkArgumentCalls(clonedModule);
    applyShardingHints(clonedModule, config, analysis.meshInfo.meshName,
                       analysis.shardAxisName, analysis.candidates);

    PassManager pm(context, ModuleOp::getOperationName(),
                   PassManager::Nesting::Implicit);
    addRemainingStableHLOPasses(pm);

    if (failed(pm.run(clonedModule))) {
      return {false, 0.0, 0.0, 0.0};
    }

    ShardingCostModel::Options costOpts;
    costOpts.maxElementsOverride = maxElementsOverride;
    ShardingCostModel costModel(costOpts);
    ShardingResult sr = costModel.evaluate(clonedModule, config,
                                           analysis.originalFuncOp,
                                           analysis.meshAxisSize);
    return {true, sr.netCost, sr.communicationCost, sr.memoryBenefit};
  }

  std::optional<AnalysisResult> analyzeModule(ModuleOp rootModule) {
    auto meshInfoOpt = extractMeshInfo(rootModule);
    if (!meshInfoOpt) {
      rootModule.emitWarning("AutoSharding: no mesh found in module, skipping");
      return std::nullopt;
    }

    AnalysisResult analysis;
    analysis.meshInfo = *meshInfoOpt;

    auto shardableAxes = analysis.meshInfo.getShardableAxes();
    if (shardableAxes.empty()) {
      llvm::errs() << "AutoSharding: no shardable axes (all size 1), "
                   << "applying all-replicated config\n";
      return std::nullopt;
    }

    analysis.shardAxisName = shardableAxes[0];
    analysis.meshAxisSize = 1;
    for (const auto &[name, size] : analysis.meshInfo.axes) {
      if (name == analysis.shardAxisName) {
        analysis.meshAxisSize = size;
        break;
      }
    }
    llvm::errs() << "AutoSharding: mesh='" << analysis.meshInfo.meshName
                 << "', sharding axis='" << analysis.shardAxisName
                 << "' (size=" << analysis.meshAxisSize << ")\n";

    auto funcOps = rootModule.getOps<func::FuncOp>();
    if (funcOps.empty()) {
      rootModule.emitWarning("AutoSharding: no FuncOp found, skipping");
      return std::nullopt;
    }
    analysis.originalFuncOp = *funcOps.begin();

    // 1. Enumerate Tier 1 configs (arg-level shardings) with pruning.
    //    Also tries shape-class grouping when per-arg space is too large.
    analysis.tier1Configs = enumerateTier1Configs(
        rootModule, analysis.perArgOptions, analysis.prunedArgs,
        analysis.shapeClasses);

    // Detect whether layer-position grouping was used (keys start with
    // "layer_pos" or "prefix_").
    if (!analysis.shapeClasses.empty()) {
      for (const auto &sc : analysis.shapeClasses) {
        if (sc.key.find("layer_pos") == 0 || sc.key.find("prefix_") == 0 ||
            sc.key.find("suffix_") == 0) {
          analysis.usesLayerPositionGroups = true;
          break;
        }
      }
    }

    // Recover the repeat pattern for logging if layer-position was used.
    if (analysis.usesLayerPositionGroups) {
      auto funcOps2 = rootModule.getOps<func::FuncOp>();
      if (!funcOps2.empty()) {
        func::FuncOp fOp = *funcOps2.begin();
        llvm::SmallVector<std::string> shapeSeq;
        for (auto arg : fOp.getArguments()) {
          auto tt = dyn_cast<RankedTensorType>(arg.getType());
          shapeSeq.push_back(tt ? getShapeKey(tt) : "_non_tensor_");
        }
        analysis.repeatPattern = detectRepeatPeriod(shapeSeq);
      }
    }

    if (analysis.tier1Configs.empty() && !analysis.perArgOptions.empty()) {
      // Try subgraph-wise search first when layer-position groups and a
      // matching subgraph template are available.
      if (analysis.usesLayerPositionGroups && analysis.repeatPattern) {
        analysis.subgraphTemplate = matchSubgraphTemplate(
            *analysis.repeatPattern, analysis.shapeClasses);
      }

      if (analysis.subgraphTemplate) {
        analysis.useSubgraphWise = true;
        analysis.searchStrategy =
            "subgraph-wise exhaustive (template: " +
            analysis.subgraphTemplate->archName + ")";
      } else {
        analysis.useHierarchical = true;
        if (analysis.usesLayerPositionGroups) {
          analysis.searchStrategy =
              "hierarchical over layer-position groups (greedy + pairwise)";
        } else if (!analysis.shapeClasses.empty()) {
          analysis.searchStrategy =
              "hierarchical over shape classes (greedy + pairwise)";
        } else {
          analysis.searchStrategy = "hierarchical (greedy + pairwise)";
        }
      }
      llvm::errs() << "AutoSharding: using " << analysis.searchStrategy << "\n";
    } else if (analysis.tier1Configs.empty()) {
      rootModule.emitWarning("AutoSharding: no configs enumerated, skipping");
      return std::nullopt;
    } else {
      if (analysis.usesLayerPositionGroups) {
        analysis.searchStrategy = "layer-position exhaustive";
      } else if (!analysis.shapeClasses.empty()) {
        analysis.searchStrategy = "shape-class exhaustive";
      } else {
        analysis.searchStrategy = "exhaustive";
      }
      llvm::errs() << "AutoSharding: " << analysis.tier1Configs.size()
                   << " Tier 1 configs (" << analysis.searchStrategy << ")\n";
    }

    // 2. Collect Tier 2 constraint candidates (intermediate op results).
    analysis.candidates =
        collectConstraintCandidates(rootModule, maxConstraintCandidates);
    llvm::errs() << "AutoSharding: " << analysis.candidates.size()
                 << " Tier 2 constraint candidate(s)\n";

    if (!analysis.useHierarchical && !analysis.useSubgraphWise) {
      // 3. Cross-product Tier 1 with Tier 2 constraint options.
      analysis.configs =
          expandWithConstraints(analysis.tier1Configs, analysis.candidates);

      llvm::errs() << "AutoSharding: evaluating " << analysis.configs.size()
                   << " total configurations (Tier 1 x Tier 2)\n";
    }

    // Set up dump directory for summary and (optionally) per-variant MLIR.
    if (!dumpDir.empty()) {
      analysis.dumpRoot = createDumpRoot(dumpDir);
      if (!analysis.dumpRoot.empty()) {
        llvm::errs() << "AutoSharding: dumping to " << analysis.dumpRoot
                     << "\n";
      }
    }

    analysis.manualRefPath = manualRef;

    return analysis;
  }

  std::optional<SearchResult> evaluateConfigs(ModuleOp rootModule,
                                              const AnalysisResult &analysis) {
    MLIRContext *context = rootModule.getContext();
    ShardingCostModel costModel;

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
                         analysis.meshInfo.meshName, analysis.shardAxisName,
                         analysis.candidates);

      std::string variantDir;
      if (dumpVariants && collectResults) {
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
          costModel.evaluate(clonedModule, analysis.configs[i],
                             analysis.originalFuncOp, analysis.meshAxisSize);
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

  std::optional<SearchResult>
  evaluateConfigsHierarchical(ModuleOp rootModule, AnalysisResult &analysis) {
    const auto &perArgOptions = analysis.perArgOptions;
    size_t numArgs = perArgOptions.size();
    bool useShapeClasses = !analysis.shapeClasses.empty();

    // Build all-replicated baseline (first option for each arg is always R).
    ShardingConfig bestConfig;
    for (size_t a = 0; a < numArgs; ++a) {
      bestConfig.argDimSharded.push_back(perArgOptions[a][0]);
    }

    llvm::SmallVector<ShardingConfig> allConfigs;
    llvm::SmallVector<VariantResult> allResults;
    size_t bestIdx = 0;
    double bestCost = std::numeric_limits<double>::infinity();

    auto tryConfig = [&](const ShardingConfig &config) -> bool {
      size_t idx = allConfigs.size();
      allConfigs.push_back(config);

      auto er = evaluateSingleConfig(rootModule, analysis, config);
      allResults.push_back({idx, formatConfig(config), er.succeeded,
                            er.netCost, er.commCost, er.memBenefit});

      llvm::errs() << "AutoSharding: eval " << idx;
      if (er.succeeded) {
        llvm::errs() << " comm=" << er.commCost << " benefit=" << er.memBenefit
                     << " net=" << er.netCost << "\n";
      } else {
        llvm::errs() << " FAILED\n";
      }

      if (er.succeeded && er.netCost < bestCost) {
        bestCost = er.netCost;
        bestIdx = idx;
        return true;
      }
      return false;
    };

    // Evaluate all-replicated baseline.
    tryConfig(bestConfig);

    constexpr int kMaxIterations = 5;

    if (useShapeClasses) {
      const auto &classes = analysis.shapeClasses;
      size_t numClasses = classes.size();

      // ---- Phase 1: Greedy Coordinate Descent over shape classes ----
      llvm::errs() << "AutoSharding: Phase 1 - Greedy Coordinate Descent over "
                   << numClasses << " shape classes\n";

      llvm::SmallVector<size_t> changedClasses;

      for (int iter = 0; iter < kMaxIterations; ++iter) {
        bool improved = false;
        for (size_t ci = 0; ci < numClasses; ++ci) {
          auto currentSharding =
              bestConfig.argDimSharded[classes[ci].argIndices[0]];

          for (size_t optIdx = 0; optIdx < classes[ci].options.size();
               ++optIdx) {
            if (classes[ci].options[optIdx] == currentSharding) {
              continue;
            }

            ShardingConfig candidate = bestConfig;
            for (size_t argIdx : classes[ci].argIndices) {
              candidate.argDimSharded[argIdx] = classes[ci].options[optIdx];
            }
            tryConfig(candidate);
          }

          bestConfig = allConfigs[bestIdx];

          if (bestConfig.argDimSharded[classes[ci].argIndices[0]] !=
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

      // ---- Phase 2: Pairwise Refinement over shape classes ----
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
                  bestConfig.argDimSharded[classes[classA].argIndices[0]];
              auto currentB =
                  bestConfig.argDimSharded[classes[classB].argIndices[0]];
              if (classes[classA].options[optA] == currentA &&
                  classes[classB].options[optB] == currentB) {
                continue;
              }

              ShardingConfig candidate = bestConfig;
              for (size_t argIdx : classes[classA].argIndices) {
                candidate.argDimSharded[argIdx] =
                    classes[classA].options[optA];
              }
              for (size_t argIdx : classes[classB].argIndices) {
                candidate.argDimSharded[argIdx] =
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
      // ---- Original per-arg hierarchical search ----
      llvm::errs() << "AutoSharding: Phase 1 - Greedy Coordinate Descent\n";

      llvm::SmallVector<size_t> changedArgs;

      for (int iter = 0; iter < kMaxIterations; ++iter) {
        bool improved = false;
        for (size_t argIdx = 0; argIdx < numArgs; ++argIdx) {
          if (perArgOptions[argIdx].size() <= 1) {
            continue;
          }

          auto currentArgSharding = bestConfig.argDimSharded[argIdx];

          for (size_t optIdx = 0; optIdx < perArgOptions[argIdx].size();
               ++optIdx) {
            if (perArgOptions[argIdx][optIdx] == currentArgSharding) {
              continue;
            }

            ShardingConfig candidate = bestConfig;
            candidate.argDimSharded[argIdx] = perArgOptions[argIdx][optIdx];
            tryConfig(candidate);
          }

          bestConfig = allConfigs[bestIdx];

          if (bestConfig.argDimSharded[argIdx] != currentArgSharding) {
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

      // ---- Phase 2: Pairwise Refinement ----
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
        size_t argIdx = arg.getArgNumber();
        if (perArgOptions[argIdx].size() <= 1) {
          continue;
        }
        if (llvm::find(changedArgs, argIdx) != changedArgs.end()) {
          continue;
        }

        int64_t numElements = 1;
        for (auto dim : tensorType.getShape()) {
          numElements *= dim;
        }
        argSizes.push_back({numElements, argIdx});
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
                   << interestingArgs.size() << " [";
      for (size_t i = 0; i < interestingArgs.size(); ++i) {
        if (i > 0) {
          llvm::errs() << ", ";
        }
        llvm::errs() << interestingArgs[i];
      }
      llvm::errs() << "]\n";

      size_t phase2Evals = 0;
      for (size_t i = 0; i < interestingArgs.size(); ++i) {
        for (size_t j = i + 1; j < interestingArgs.size(); ++j) {
          size_t argA = interestingArgs[i];
          size_t argB = interestingArgs[j];

          for (size_t optA = 0; optA < perArgOptions[argA].size(); ++optA) {
            for (size_t optB = 0; optB < perArgOptions[argB].size(); ++optB) {
              if (perArgOptions[argA][optA] ==
                      bestConfig.argDimSharded[argA] &&
                  perArgOptions[argB][optB] ==
                      bestConfig.argDimSharded[argB]) {
                continue;
              }

              ShardingConfig candidate = bestConfig;
              candidate.argDimSharded[argA] = perArgOptions[argA][optA];
              candidate.argDimSharded[argB] = perArgOptions[argB][optB];

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

  //===--------------------------------------------------------------------===//
  // Subgraph-wise exhaustive search.
  //===--------------------------------------------------------------------===//

  std::optional<SearchResult>
  evaluateConfigsSubgraphWise(ModuleOp rootModule, AnalysisResult &analysis) {
    const auto &perArgOptions = analysis.perArgOptions;
    const auto &groups = analysis.shapeClasses;
    const auto &tmpl = *analysis.subgraphTemplate;
    size_t numArgs = perArgOptions.size();

    constexpr int64_t kMaxExhaustiveConfigs = 50000;

    // Compute layer-local maxElements from repeating period args only,
    // excluding prefix/suffix outliers (e.g., embedding tables) that would
    // inflate the normalization constant and suppress layer-weight benefits.
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
      layerMaxElements = ShardingCostModel::computeMaxElements(
          analysis.originalFuncOp);
    }
    llvm::errs() << "AutoSharding: subgraph-wise using layer-local "
                 << "maxElements=" << layerMaxElements << " (vs global="
                 << ShardingCostModel::computeMaxElements(
                        analysis.originalFuncOp)
                 << ")\n";

    // Build all-replicated baseline.
    ShardingConfig bestConfig;
    for (size_t a = 0; a < numArgs; ++a) {
      bestConfig.argDimSharded.push_back(perArgOptions[a][0]);
    }

    llvm::SmallVector<ShardingConfig> allConfigs;
    llvm::SmallVector<VariantResult> allResults;
    size_t bestIdx = 0;
    double bestCost = std::numeric_limits<double>::infinity();

    auto tryConfig = [&](const ShardingConfig &config) -> bool {
      size_t idx = allConfigs.size();
      allConfigs.push_back(config);

      auto er = evaluateSingleConfig(rootModule, analysis, config,
                                     layerMaxElements);
      allResults.push_back({idx, formatConfig(config), er.succeeded,
                            er.netCost, er.commCost, er.memBenefit});

      llvm::errs() << "AutoSharding: eval " << idx;
      if (er.succeeded) {
        llvm::errs() << " comm=" << er.commCost << " benefit=" << er.memBenefit
                     << " net=" << er.netCost << "\n";
      } else {
        llvm::errs() << " FAILED\n";
      }

      if (er.succeeded && er.netCost < bestCost) {
        bestCost = er.netCost;
        bestIdx = idx;
        return true;
      }
      return false;
    };

    // Evaluate all-replicated baseline.
    tryConfig(bestConfig);

    // Helper: find the group index for a given period position.
    auto findGroupForPosition = [&](size_t pos) -> std::optional<size_t> {
      std::string prefix = "layer_pos" + std::to_string(pos) + "_";
      for (size_t gi = 0; gi < groups.size(); ++gi) {
        if (groups[gi].key.find(prefix) == 0) {
          return gi;
        }
      }
      return std::nullopt;
    };

    // Collect prefix/suffix groups not covered by any template family.
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

    // Build the sequence of families to search: template families + "other".
    struct FamilySearch {
      std::string name;
      llvm::SmallVector<size_t> groupIndices;
    };

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

    // Search each family exhaustively.
    for (const auto &fs : familySearches) {
      llvm::errs() << "AutoSharding: searching subgraph family '" << fs.name
                   << "' (" << fs.groupIndices.size() << " groups)\n";

      // Build per-group options for this family.
      llvm::SmallVector<llvm::SmallVector<llvm::SmallVector<bool>>>
          familyGroupOptions;
      for (size_t gi : fs.groupIndices) {
        familyGroupOptions.push_back(groups[gi].options);
        llvm::errs() << "  group '" << groups[gi].key << "': "
                     << groups[gi].options.size() << " options, "
                     << groups[gi].argIndices.size() << " args\n";
      }

      // Check search space size.
      int64_t familySpace = 1;
      bool overflow = false;
      for (const auto &opts : familyGroupOptions) {
        familySpace *= static_cast<int64_t>(opts.size());
        if (familySpace > kMaxExhaustiveConfigs) {
          overflow = true;
          break;
        }
      }

      if (overflow) {
        llvm::errs() << "AutoSharding: family '" << fs.name
                     << "' search space too large (>" << kMaxExhaustiveConfigs
                     << "), using greedy within family\n";

        // Fall back to greedy coordinate descent within this family.
        constexpr int kMaxIterations = 5;
        for (int iter = 0; iter < kMaxIterations; ++iter) {
          bool improved = false;
          for (size_t fi = 0; fi < fs.groupIndices.size(); ++fi) {
            size_t gi = fs.groupIndices[fi];
            auto currentSharding =
                bestConfig.argDimSharded[groups[gi].argIndices[0]];

            for (size_t optIdx = 0; optIdx < groups[gi].options.size();
                 ++optIdx) {
              if (groups[gi].options[optIdx] == currentSharding) {
                continue;
              }

              ShardingConfig candidate = bestConfig;
              for (size_t argIdx : groups[gi].argIndices) {
                candidate.argDimSharded[argIdx] = groups[gi].options[optIdx];
              }
              tryConfig(candidate);
            }

            bestConfig = allConfigs[bestIdx];
            if (bestConfig.argDimSharded[groups[gi].argIndices[0]] !=
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

        // Exhaustive enumeration over this family's groups.
        auto familyConfigs = cartesianProduct(familyGroupOptions);
        for (const auto &fc : familyConfigs) {
          ShardingConfig candidate = bestConfig;

          // Apply this family config to the corresponding args.
          for (size_t fi = 0; fi < fs.groupIndices.size(); ++fi) {
            size_t gi = fs.groupIndices[fi];
            for (size_t argIdx : groups[gi].argIndices) {
              candidate.argDimSharded[argIdx] = fc.argDimSharded[fi];
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

  void applyBestConfig(ModuleOp rootModule, const AnalysisResult &analysis,
                       const SearchResult &search) {
    MLIRContext *context = rootModule.getContext();

    llvm::errs() << "AutoSharding: selected config " << search.bestIdx << " "
                 << formatConfig(analysis.configs[search.bestIdx])
                 << " with net cost=" << search.bestCost << "\n";
    applyShardingHints(rootModule, analysis.configs[search.bestIdx],
                       analysis.meshInfo.meshName, analysis.shardAxisName,
                       analysis.candidates);

    // Save the winning config's MLIR graphs.
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
    }

    // Write summary file.
    if (!analysis.dumpRoot.empty()) {
      writeSummary(analysis, search);
    }
  }

  void writeSummary(const AnalysisResult &analysis,
                    const SearchResult &search) {
    llvm::SmallString<256> summaryPath(analysis.dumpRoot);
    llvm::sys::path::append(summaryPath, "summary.txt");
    std::error_code ec;
    llvm::raw_fd_ostream fos(summaryPath, ec);
    if (ec) {
      return;
    }

    ShardingCostModel::Options costOpts;

    fos << "Auto Sharding Summary\n";
    fos << "=======================\n";
    fos << "Mesh: " << analysis.meshInfo.meshName << "\n";
    fos << "Sharding axis: " << analysis.shardAxisName << "\n";
    fos << "Search strategy: " << analysis.searchStrategy << "\n";

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

    if (!analysis.shapeClasses.empty()) {
      if (analysis.usesLayerPositionGroups) {
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

    if (!analysis.useHierarchical && !analysis.useSubgraphWise) {
      fos << "Tier 1 configs: " << analysis.tier1Configs.size() << "\n";
      fos << "Tier 2 constraint candidates: " << analysis.candidates.size()
          << "\n";
    }
    fos << "Total configs evaluated: " << search.results.size() << "\n";
    fos << "Cost model: per-CCL latency + volume-weighted bandwidth + "
        << "critical-path penalty + output-gather cost, "
        << "parameter multiplier="
        << llvm::format("%.1f", costOpts.parameterMultiplier)
        << " compute-benefit-weight="
        << llvm::format("%.1f", costOpts.computeBenefitWeight) << "\n\n";

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

    fos << "\nSelected: config " << search.bestIdx << " "
        << formatConfig(analysis.configs[search.bestIdx])
        << " with net cost=" << llvm::format("%.3f", search.bestCost) << "\n";

    // Winner sharding by layer position for easy comparison with manual.
    if (analysis.useSubgraphWise && analysis.repeatPattern &&
        analysis.subgraphTemplate) {
      const auto &winConfig = analysis.configs[search.bestIdx];
      const auto &rp = *analysis.repeatPattern;
      const auto &tmpl = *analysis.subgraphTemplate;
      const auto &groups = analysis.shapeClasses;

      llvm::SmallVector<llvm::SmallVector<bool>> manualShardings;
      bool hasManual = false;
      if (!analysis.manualRefPath.empty()) {
        manualShardings = parseManualShardings(analysis.manualRefPath);
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

      auto formatSharding = [](const llvm::SmallVector<bool> &dims) {
        std::string s = "[";
        for (size_t d = 0; d < dims.size(); ++d) {
          s += dims[d] ? "S" : "R";
          if (d + 1 < dims.size()) {
            s += ",";
          }
        }
        s += "]";
        return s;
      };

      auto findGroupForPos = [&](size_t pos) -> std::optional<size_t> {
        std::string pfx = "layer_pos" + std::to_string(pos) + "_";
        for (size_t gi = 0; gi < groups.size(); ++gi) {
          if (groups[gi].key.find(pfx) == 0) {
            return gi;
          }
        }
        return std::nullopt;
      };

      auto getFamilyForPos = [&](size_t pos) -> std::string {
        for (const auto &fam : tmpl.families) {
          for (size_t fi = 0; fi < fam.periodPositions.size(); ++fi) {
            if (fam.periodPositions[fi] == pos) {
              return fam.name;
            }
          }
        }
        return "other";
      };

      for (size_t pos = 0; pos < rp.period; ++pos) {
        auto gi = findGroupForPos(pos);
        std::string famName = getFamilyForPos(pos);

        fos << "  pos " << llvm::format("%-3zu", pos);
        if (gi) {
          size_t repArgIdx = groups[*gi].argIndices[0];
          std::string autoSharding =
              formatSharding(winConfig.argDimSharded[repArgIdx]);
          fos << llvm::format("%-28s", groups[*gi].key.c_str())
              << llvm::format("%-12s", autoSharding.c_str());

          if (hasManual) {
            size_t manualArgIdx = rp.prefixLen + pos;
            std::string manualSharding =
                (manualArgIdx < manualShardings.size())
                    ? formatSharding(manualShardings[manualArgIdx])
                    : "?";
            bool match = (manualArgIdx < manualShardings.size()) &&
                         (winConfig.argDimSharded[repArgIdx] ==
                          manualShardings[manualArgIdx]);
            fos << llvm::format("%-12s", manualSharding.c_str())
                << (match ? " " : "*") << "(" << famName << ")";
          } else {
            fos << "(" << famName << ")";
          }
        } else {
          fos << "(pruned)";
          if (hasManual) {
            size_t manualArgIdx = rp.prefixLen + pos;
            if (manualArgIdx < manualShardings.size()) {
              std::string manualSharding =
                  formatSharding(manualShardings[manualArgIdx]);
              fos << std::string(28, ' ')
                  << std::string(12, ' ')
                  << llvm::format("%-12s", manualSharding.c_str());
            }
          }
        }
        fos << "\n";
      }

      // Show prefix/suffix args.
      if (rp.prefixLen > 0 || rp.suffixLen > 0) {
        fos << "\n  Prefix/Suffix args:\n";
        for (size_t gi = 0; gi < groups.size(); ++gi) {
          bool isLayerGroup = groups[gi].key.find("layer_pos") == 0;
          if (!isLayerGroup && !groups[gi].argIndices.empty()) {
            size_t argIdx = groups[gi].argIndices[0];
            std::string autoSharding =
                formatSharding(winConfig.argDimSharded[argIdx]);
            fos << "  " << llvm::format("%-34s", groups[gi].key.c_str())
                << llvm::format("%-12s", autoSharding.c_str());

            if (hasManual && argIdx < manualShardings.size()) {
              std::string manualSharding =
                  formatSharding(manualShardings[argIdx]);
              bool match =
                  (winConfig.argDimSharded[argIdx] == manualShardings[argIdx]);
              fos << llvm::format("%-12s", manualSharding.c_str())
                  << (match ? " " : "*");
            }
            fos << "(" << groups[gi].argIndices.size() << " args)\n";
          }
        }
      }

      if (hasManual) {
        fos << "\n  (* = auto differs from manual)\n";
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::stablehlo
