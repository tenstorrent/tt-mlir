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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

#define DEBUG_TYPE "auto-sharding"

// =============================================================================
// AutoSharding Pass
// =============================================================================
//
// Algorithm overview:
//   1. Extract the mesh from the module and identify the sharding axis.
//   2. Enumerate all valid per-argument sharding configurations. Each argument
//      of rank r has (r+1) choices (shard on one dim or replicate), giving
//      ∏(rank_i + 1) total configs.
//   3. For each configuration: clone the module, apply sharding hints, run the
//      full remaining StableHLO pipeline (propagation, reshard-to-collectives,
//      canonicalization), and score the result using a cost model.
//   4. The cost model computes:
//        net_cost = communication_cost - memory_benefit
//      where communication_cost sums weighted CCL ops and memory_benefit
//      estimates savings from distributing tensors across devices.
//   5. Select the configuration with the lowest net_cost and apply it to the
//      original module.
//
// Known limitations:
//   - Only searches single-axis sharding (first mesh axis with size > 1).
//   - Compile time scales with ∏(rank_i + 1) configs, capped at 2^20.
//   - Cost model uses heuristic weights — not calibrated to specific hardware.
//   - Does not consider operator-level sharding, only argument-level.
//
// =============================================================================

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_AUTOSHARDINGPASS // move the include and define to the top
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Data structures.
//===----------------------------------------------------------------------===//

// Per-argument, per-dimension sharding decisions. Each inner vector
// corresponds to one function argument; each bool indicates whether that
// dimension is sharded on the mesh axis or not
//
// Example:
//   func @main(%arg0: tensor<32x32xf32>, %arg1: tensor<64xf32>)
//   config = [[true, false], [false]]
//     -> %arg0 sharded on dim 0, %arg1 replicated.
using ArgDimSharded = llvm::SmallVector<llvm::SmallVector<bool>>;

// Everything needed to run the sharding search. Constructed once after
// validation, then passed by const-ref to the evaluation and application steps.
struct AnalysisResult {
  std::string meshName;
  // First shardable axis (size > 1) from the mesh — the axis we shard on.
  std::string shardAxisName;
  int64_t meshAxisSize;
  func::FuncOp originalFuncOp;
  llvm::SmallVector<ArgDimSharded> configs;
  std::string dumpRoot; // Empty if dumping is disabled.
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

// Result of the exhaustive sharding search.
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

// Find the first shardable axis (size > 1) in the module's mesh.
// Returns the mesh name, axis name, and axis size via out-parameters.
// Returns false if no mesh is found or no axis has size > 1.
// Only searches single-axis sharding
// (next version will support multi-axis 2D meshes).
static bool extractShardableAxis(ModuleOp &module, std::string &meshName,
                                 std::string &axisName, int64_t &axisSize) {
  llvm::SmallVector<mlir::sdy::MeshOp> meshOps =
      shardy_utils::getMeshOps(module);
  if (meshOps.empty()) {
    return false;
  }

  meshName = meshOps[0].getSymName().str();
  for (auto axisAttr : meshOps[0].getMeshAttr().getAxes()) {
    if (axisAttr.getSize() > 1) {
      axisName = axisAttr.getName().str();
      axisSize = axisAttr.getSize();
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Configuration enumeration.
//===----------------------------------------------------------------------===//

// Enumerate all valid per-argument sharding configurations.
// Constraint: at most one dimension per tensor can be sharded (we only have a
// single shardable mesh axis).
//
// Each argument of rank r has (r + 1) valid choices: shard on exactly one of
// the r dimensions, or replicate (no sharding). The total search space is
// ∏(rank_i + 1), which is much smaller than 2^totalDims because invalid
// multi-shard-per-tensor configs are never generated.
static llvm::SmallVector<ArgDimSharded>
enumerateArgShardings(ModuleOp &module) {
  llvm::SmallVector<ArgDimSharded> configs;
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

  // Compute total number of valid configs: ∏(rank_i + 1).
  // Each factor is (rank + 1) choices: shard on one dim, or replicate.
  int64_t numConfigs = 1;
  for (auto rank : argRanks) {
    numConfigs *= (rank + 1);
    if (numConfigs > (1LL << 20)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "AutoSharding: search space exceeds 2^20, capping\n");
      numConfigs = 1LL << 20;
      break;
    }
  }

  // Enumerate configs using mixed-radix indexing.
  // For each config index, decompose it into per-arg choices where
  // choice == rank means replicate, and choice < rank means shard that dim.
  for (int64_t idx = 0; idx < numConfigs; ++idx) {
    ArgDimSharded config;
    int64_t remainder = idx;
    for (size_t argIdx = 0; argIdx < argRanks.size(); ++argIdx) {
      int64_t base = argRanks[argIdx] + 1;
      int64_t choice = remainder % base;
      remainder /= base;

      llvm::SmallVector<bool> dimChoices(argRanks[argIdx], false);
      if (choice < argRanks[argIdx]) {
        dimChoices[choice] = true;
      }
      config.push_back(std::move(dimChoices));
    }
    configs.push_back(std::move(config));
  }
  return configs;
}

//===----------------------------------------------------------------------===//
// Sharding application.
//===----------------------------------------------------------------------===//

// Apply arg-level sharding hints to a module by setting sdy.sharding
// annotations on each function argument.
static void applyShardingHints(ModuleOp module, const ArgDimSharded &config,
                               StringRef meshName, StringRef shardAxisName) {
  MLIRContext *context = module.getContext();

  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return;
  }
  func::FuncOp funcOp = *funcOps.begin();

  constexpr bool isClosed = false;
  for (size_t argIdx = 0; argIdx < config.size(); ++argIdx) {
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    for (bool sharded : config[argIdx]) {
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
    // Remove any existing sdy.sharding before adding the new one. The
    // addDictionaryAttrSdyShardingAnnotation API expects the caller to
    // handle pre-existing annotations explicitly.
    std::optional<mlir::DictionaryAttr> optDict = std::nullopt;
    if (existingDict) {
      optDict = shardy_utils::removeDictionaryAttrSdyShardingAnnotations(
          context, existingDict);
    }
    auto newDict = shardy_utils::addDictionaryAttrSdyShardingAnnotation(
        context, sharding, optDict);
    funcOp.setArgAttrs(argIdx, newDict);
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

// Format a sharding config as a human-readable string, e.g.
// "[arg0:[S,R], arg1:[R]]".
static std::string formatConfig(const ArgDimSharded &config) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "[";
  for (size_t a = 0; a < config.size(); ++a) {
    if (a > 0) {
      os << ", ";
    }
    os << "arg" << a << ":[";
    for (size_t d = 0; d < config[a].size(); ++d) {
      if (d > 0) {
        os << ",";
      }
      os << (config[a][d] ? "S" : "R");
    }
    os << "]";
  }
  os << "]";
  return result;
}

// Generate a filesystem-safe directory name for a variant, e.g.
// "variant_03_arg0-SR_arg1-R".
static std::string configDirName(size_t idx, const ArgDimSharded &config) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << llvm::format("variant_%02zu", idx);
  for (size_t a = 0; a < config.size(); ++a) {
    os << "_arg" << a << "-";
    for (bool s : config[a]) {
      os << (s ? "S" : "R");
    }
  }
  return result;
}

// Create (or verify) the dump root directory. Returns the path on success,
// or an empty string on failure.
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
// Core algorithm: evaluate configs and apply the best one.
//===----------------------------------------------------------------------===//

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
    applyShardingHints(clonedModule, analysis.configs[i], analysis.meshName,
                       analysis.shardAxisName);

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
                   << formatConfig(analysis.configs[i]) << " failed to lower\n";
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
        evaluate(clonedModule, analysis.configs[i], analysis.originalFuncOp,
                 analysis.meshAxisSize);
    LLVM_DEBUG(llvm::dbgs() << "AutoSharding: config " << i << " "
                            << formatConfig(analysis.configs[i])
                            << " comm=" << sr.communicationCost
                            << " benefit=" << sr.memoryBenefit
                            << " net=" << sr.netCost << "\n");

    if (collectResults) {
      results.push_back({i, formatConfig(analysis.configs[i]), true, sr.netCost,
                         sr.communicationCost, sr.memoryBenefit});
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

  fos << "Auto Sharding Summary\n";
  fos << "=======================\n";
  fos << "Mesh: " << analysis.meshName << "\n";
  fos << "Sharding axis: " << analysis.shardAxisName << "\n";
  fos << "Configs evaluated: " << analysis.configs.size() << "\n";
  fos << "Cost model: per-CCL latency + volume-weighted bandwidth, "
      << "parameter multiplier=3.0\n\n";

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

// Apply the winning sharding config to the original module and optionally
// dump the winning MLIR + summary.
static void applyBestConfig(ModuleOp rootModule, const AnalysisResult &analysis,
                            const SearchResult &search) {
  MLIRContext *context = rootModule.getContext();

  LLVM_DEBUG(llvm::dbgs() << "AutoSharding: selected config " << search.bestIdx
                          << " "
                          << formatConfig(analysis.configs[search.bestIdx])
                          << " with net cost=" << search.bestCost << "\n");
  applyShardingHints(rootModule, analysis.configs[search.bestIdx],
                     analysis.meshName, analysis.shardAxisName);

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
      llvm::sys::path::append(winnerCCLPath, "winner_stablehlo_with_ccls.mlir");
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

    // Validate: module must have a mesh with at least one shardable axis and
    // at least one FuncOp.
    std::string meshName, shardAxisName;
    int64_t meshAxisSize = 0;
    if (!extractShardableAxis(rootModule, meshName, shardAxisName,
                              meshAxisSize)) {
      rootModule.emitWarning(
          "AutoSharding: no mesh or no shardable axes found, skipping");
      return;
    }
    auto funcOps = rootModule.getOps<func::FuncOp>();
    if (funcOps.empty()) {
      rootModule.emitWarning("AutoSharding: no FuncOp found, skipping");
      return;
    }
    if (std::distance(funcOps.begin(), funcOps.end()) > 1) {
      rootModule.emitWarning(
          "AutoSharding: multiple FuncOps found (e.g. from composite ops); "
          "only the first will be used for sharding enumeration");
    }

    LLVM_DEBUG(llvm::dbgs()
               << "AutoSharding: mesh='" << meshName << "', sharding axis='"
               << shardAxisName << "' (size=" << meshAxisSize << ")\n");

    // Enumerate and validate configs.
    auto configs = enumerateArgShardings(rootModule);
    if (configs.empty()) {
      rootModule.emitWarning("AutoSharding: no configs enumerated, skipping");
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "AutoSharding: evaluating " << configs.size()
                            << " configurations\n");

    // Set up dump directory.
    std::string dumpRoot;
    if (!dumpDir.empty()) {
      dumpRoot = createDumpRoot(dumpDir);
      if (dumpRoot.empty()) {
        rootModule.emitWarning(
            "AutoSharding: failed to create dump directory '")
            << dumpDir << "', continuing without dumping";
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "AutoSharding: dumping to " << dumpRoot << "\n");
      }
    }

    AnalysisResult analysis{
        std::move(meshName), std::move(shardAxisName), meshAxisSize,
        *funcOps.begin(),    std::move(configs),       std::move(dumpRoot),
    };

    // Run search.
    context->exitMultiThreadedExecution();
    auto restoreGuard = llvm::make_scope_exit(
        [context] { context->enterMultiThreadedExecution(); });

    auto searchResult = evaluateConfigs(rootModule, analysis, dumpVariants);
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
