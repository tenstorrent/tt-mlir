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

#include <cmath>

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

// Tier 1: enumerate arg-level sharding configs.
static llvm::SmallVector<ShardingConfig>
enumerateTier1Configs(ModuleOp &module) {
  llvm::SmallVector<ShardingConfig> configs;
  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return configs;
  }
  func::FuncOp funcOp = *funcOps.begin();

  llvm::SmallVector<int64_t> argRanks;
  for (auto arg : funcOp.getArguments()) {
    auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
    if (!tensorType) {
      return configs;
    }
    argRanks.push_back(tensorType.getRank());
  }

  int64_t totalDims = 0;
  for (auto rank : argRanks) {
    totalDims += rank;
  }

  if (totalDims > 20) {
    llvm::errs() << "AutoSharding: totalDims=" << totalDims
                 << " too large, capping at first 1024 configs\n";
  }

  int64_t numConfigs = 1LL << std::min(totalDims, static_cast<int64_t>(20));

  for (int64_t bits = 0; bits < numConfigs; ++bits) {
    ShardingConfig config;
    int64_t bitIdx = 0;
    bool valid = true;
    for (size_t argIdx = 0; argIdx < argRanks.size(); ++argIdx) {
      llvm::SmallVector<bool> dimChoices;
      int shardedCount = 0;
      for (int64_t d = 0; d < argRanks[argIdx]; ++d) {
        bool sharded = (bits >> bitIdx) & 1;
        dimChoices.push_back(sharded);
        if (sharded) {
          ++shardedCount;
        }
        ++bitIdx;
      }
      // With a single shardable axis, at most one dim per tensor can use it.
      if (shardedCount > 1) {
        valid = false;
        break;
      }
      config.argDimSharded.push_back(std::move(dimChoices));
    }
    if (valid) {
      configs.push_back(std::move(config));
    }
  }
  return configs;
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

    auto searchResult = evaluateConfigs(rootModule, analysis);
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

    // 1. Enumerate Tier 1 configs (arg-level shardings).
    analysis.tier1Configs = enumerateTier1Configs(rootModule);
    if (analysis.tier1Configs.empty()) {
      rootModule.emitWarning("AutoSharding: no configs enumerated, skipping");
      return std::nullopt;
    }
    llvm::errs() << "AutoSharding: " << analysis.tier1Configs.size()
                 << " Tier 1 configs\n";

    // 2. Collect Tier 2 constraint candidates (intermediate op results).
    analysis.candidates =
        collectConstraintCandidates(rootModule, maxConstraintCandidates);
    llvm::errs() << "AutoSharding: " << analysis.candidates.size()
                 << " Tier 2 constraint candidate(s)\n";

    // 3. Cross-product Tier 1 with Tier 2 constraint options.
    analysis.configs =
        expandWithConstraints(analysis.tier1Configs, analysis.candidates);

    llvm::errs() << "AutoSharding: evaluating " << analysis.configs.size()
                 << " total configurations (Tier 1 x Tier 2)\n";

    // Set up dump directory for summary and (optionally) per-variant MLIR.
    if (!dumpDir.empty()) {
      analysis.dumpRoot = createDumpRoot(dumpDir);
      if (!analysis.dumpRoot.empty()) {
        llvm::errs() << "AutoSharding: dumping to " << analysis.dumpRoot
                     << "\n";
      }
    }

    return analysis;
  }

  std::optional<SearchResult> evaluateConfigs(ModuleOp rootModule,
                                              const AnalysisResult &analysis) {
    MLIRContext *context = rootModule.getContext();
    ShardingCostModel costModel;

    double bestCost = INFINITY;
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
    fos << "Tier 1 configs: " << analysis.tier1Configs.size() << "\n";
    fos << "Tier 2 constraint candidates: " << analysis.candidates.size()
        << "\n";
    fos << "Total configs evaluated: " << analysis.configs.size() << "\n";
    fos << "Cost model: per-CCL latency + volume-weighted bandwidth, "
        << "parameter multiplier="
        << llvm::format("%.1f", costOpts.parameterMultiplier) << "\n\n";

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
  }
};

} // namespace
} // namespace mlir::tt::stablehlo
