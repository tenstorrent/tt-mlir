// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/user_priority_propagation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <limits>

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_AUTOSHARDINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// For Dimentions
struct ShardingConfig {
  llvm::SmallVector<llvm::SmallVector<bool>> argDimSharded;
};

// For Axes
struct MeshInfo {
  llvm::StringRef meshName;
  llvm::SmallVector<std::pair<std::string, int64_t>> axes;

  llvm::SmallVector<std::string> getShardableAxes() const {
    llvm::SmallVector<std::string> results;
    for (const auto &[name, size] : axes) {
      // Already checked in extractMeshInfo
      if (size > 1) {
        results.push_back(name);
      }
    }
    return results;
  }
};

static std::optional<MeshInfo> extractMeshInfo(ModuleOp& module) {
  llvm::SmallVector<mlir::sdy::MeshOp> meshOps = shardy_utils::getMeshOps(module);
  if (meshOps.empty()) {
    return std::nullopt;
  }

  MeshInfo info;
  info.meshName = meshOps[0].getSymName();
  for (auto axisAttr : meshOps[0].getMeshAttr().getAxes()) {
    if (axisAttr.getSize() > 1) { // What if there is no size argument? Could it be a case?
      info.axes.emplace_back(axisAttr.getName().str(), axisAttr.getSize());
    }
  }
  return info;
}

// Direct get value to apply to dims
static llvm::SmallVector<ShardingConfig> enumerateShardingConfigs(ModuleOp& module) {
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

  // As each dimension can be sharded or not, there are 2^totalDims configs
  int64_t numConfigs = 1LL << std::min(totalDims, static_cast<int64_t>(20));

  for (int64_t bits = 0; bits < numConfigs; ++bits) {
    ShardingConfig config;
    int64_t bitIdx = 0;
    bool valid = true;
    for (size_t argIdx = 0; argIdx < argRanks.size(); ++argIdx) {
      llvm::SmallVector<bool> dimChoices; // for that argument, which dimensions are sharded?
      int shardedCount = 0;
      for (int64_t d = 0; d < argRanks[argIdx]; ++d) {
        bool sharded = (bits >> bitIdx) & 1; // ???
        dimChoices.push_back(sharded);
        if (sharded) {
          ++shardedCount;
        }
        ++bitIdx;
      }
      // With a single shardable axis, at most one dim per tensor can be
      // sharded on it. Skip configs that shard multiple dims on the same axis. ???
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

// Apply sharding hints to a module's function arguments based on a config.
static void applyShardingHints(ModuleOp module, const ShardingConfig &config,
                               StringRef meshName, StringRef shardAxisName) {
  MLIRContext *context = module.getContext();

  auto funcOps = module.getOps<func::FuncOp>();
  if (funcOps.empty()) {
    return;
  }
  func::FuncOp funcOp = *funcOps.begin();

  for (size_t argIdx = 0; argIdx < config.argDimSharded.size(); ++argIdx) {
    llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;

    for (bool sharded : config.argDimSharded[argIdx]) {
      if (sharded) {
        auto axisRef = mlir::sdy::AxisRefAttr::get(context, shardAxisName);
        dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
            context, {axisRef}, /*isClosed=*/false));
      } else {
        dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
            context, {}, /*isClosed=*/false));
      }
    }

    auto sharding = mlir::sdy::TensorShardingAttr::get(
        context, meshName, dimShardings, /*replicatedAxes=*/{},
        /*unreducedAxes=*/{});

    // Build new dict, guarding against null existing attrs.
    auto existingDict = funcOp.getArgAttrDict(argIdx);
    std::optional<mlir::DictionaryAttr> optDict =
        existingDict ? std::optional(existingDict) : std::nullopt;
    auto newDict = shardy_utils::addDictionaryAttrSdyShardingAnnotation(
        context, sharding, optDict);
    funcOp.setArgAttrs(argIdx, newDict);
  }
}


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

  pm.nest<func::FuncOp>().addPass(
      mlir::sdy::createReshardToCollectivesPass());

  pm.addPass(createShardyCCLCanonicalizationPass());
  pm.addPass(createUpdateGlobalToLocalShapesPass());
  pm.addPass(createReoutlineCompositePass());
  pm.addPass(mlir::sdy::createCloseShardingsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

// Evaluate cost of a lowered TTNN module by counting and weighting CCL ops.
// Uses string-based op name matching to avoid compile-time dependency on TTNN.
static double evaluateCost(ModuleOp module) {
  double totalCost = 0.0;

  module.walk([&](Operation *op) {
    StringRef opName = op->getName().getStringRef();
    if (opName == "ttnn.all_gather") {
      totalCost += 1.0;
    } else if (opName == "ttnn.reduce_scatter") {
      totalCost += 1.5;
    } else if (opName == "ttnn.all_reduce") {
      totalCost += 2.0;
    }
  });

  return totalCost;
}


// Format a ShardingConfig for log output.
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
  os << "]";
  return result;
}

// Build a filesystem-friendly config name like "arg0-RR_arg1-SR".
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
  return result;
}


// Create a timestamped dump directory and return its path.
// Returns empty string on failure.
static std::string createDumpRoot(StringRef baseDir) {
  auto now = std::chrono::system_clock::now();
  auto timeT = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  ::localtime_r(&timeT, &tm);

  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

  llvm::SmallString<256> path;
  if (baseDir.empty()) {
    path = ".";
  } else {
    path = baseDir;
  }
  llvm::sys::path::append(path, llvm::Twine("auto_sharding_") + buf);

  if (auto ec = llvm::sys::fs::create_directories(path)) {
    llvm::errs() << "AutoSharding: failed to create dump directory " << path
                 << ": " << ec.message() << "\n";
    return "";
  }
  return path.str().str();
}

// Dump a module to a file. Returns true on success.
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


class AutoShardingPass
    : public impl::AutoShardingPassBase<AutoShardingPass> {
public:
  using impl::AutoShardingPassBase<AutoShardingPass>::AutoShardingPassBase;

  void runOnOperation() final {
    ModuleOp rootModule = getOperation();
    MLIRContext *context = rootModule.getContext();

    // The outer PassManager::run() holds a multi-threaded execution guard on
    // the context. Inner PassManager::run() calls appendDialectRegistry which
    // asserts single-threaded access. Temporarily exit the outer guard.
    context->exitMultiThreadedExecution();
    auto restoreGuard = llvm::make_scope_exit(
        [context] { context->enterMultiThreadedExecution(); });

    // Extract mesh info from the module.
    auto meshInfoOpt = extractMeshInfo(rootModule);
    if (!meshInfoOpt) {
      rootModule.emitWarning(
          "AutoSharding: no mesh found in module, skipping");
      return;
    }
    const MeshInfo &meshInfo = *meshInfoOpt;

    // Find axes worth sharding on (size > 1).
    auto shardableAxes = meshInfo.getShardableAxes();
    if (shardableAxes.empty()) {
      llvm::errs() << "AutoSharding: no shardable axes (all size 1), "
                   << "applying all-replicated config\n";
      return;
    }

    // Phase 0: use only the first shardable axis.
    StringRef shardAxisName = shardableAxes[0];
    llvm::errs() << "AutoSharding: mesh='" << meshInfo.meshName
                 << "', sharding axis='" << shardAxisName << "'\n";

    // 1. Enumerate sharding configs.
    auto configs = enumerateShardingConfigs(rootModule);
    if (configs.empty()) {
      rootModule.emitWarning(
          "AutoSharding: no configs enumerated, skipping");
      return;
    }

    llvm::errs() << "AutoSharding: evaluating " << configs.size()
                 << " sharding configurations\n";

    // Set up dump directory if requested.
    std::string dumpRoot;
    if (dumpVariants) {
      dumpRoot = createDumpRoot(dumpDir);
      if (!dumpRoot.empty()) {
        llvm::errs() << "AutoSharding: dumping variants to " << dumpRoot
                     << "\n";
      }
    }

    // Lookup the downstream pipelines (runtime, no compile-time dep on TTNN).
    const auto *ttirPipeline =
        PassPipelineInfo::lookup("stablehlo-to-ttir-pipeline");
    const auto *ttnnPipeline =
        PassPipelineInfo::lookup("ttir-to-ttnn-backend-pipeline");

    if (!ttirPipeline || !ttnnPipeline) {
      rootModule.emitError(
          "AutoSharding: cannot find required pipelines "
          "(stablehlo-to-ttir-pipeline / ttir-to-ttnn-backend-pipeline)");
      signalPassFailure();
      return;
    }

    // Build ttnn pipeline options string.
    std::string ttnnOptions;
    if (!systemDescPath.empty()) {
      ttnnOptions = "system-desc-path=" + systemDescPath;
    }
    if (!meshShape.empty()) {
      if (!ttnnOptions.empty()) {
        ttnnOptions += " ";
      }
      ttnnOptions += "mesh-shape=";
      for (size_t j = 0; j < meshShape.size(); ++j) {
        if (j > 0) {
          ttnnOptions += ",";
        }
        ttnnOptions += std::to_string(meshShape[j]);
      }
    }

    double bestCost = std::numeric_limits<double>::max();
    size_t bestIdx = 0;
    bool anySucceeded = false;


    struct VariantResult {
      size_t idx;
      std::string label;
      bool succeeded;
      double cost;
    };
    llvm::SmallVector<VariantResult> results;
    bool collectResults = !dumpRoot.empty();

    std::function<LogicalResult(const Twine &)> errHandler =
        [](const Twine &) { return failure(); };

    for (size_t i = 0; i < configs.size(); ++i) {
      // 2. Clone the module.
      ModuleOp clonedModule = cast<ModuleOp>(rootModule->clone());
      auto cleanup =
          llvm::make_scope_exit([&clonedModule] { clonedModule->erase(); });

      // 3. Apply sharding hints to the clone.
      applyShardingHints(clonedModule, configs[i], meshInfo.meshName,
                         shardAxisName);

      // Dump the StableHLO IR with hints applied (before lowering).
      std::string variantDir;
      if (collectResults) {
        std::string varName = configDirName(i, configs[i]);
        llvm::SmallString<256> vdir(dumpRoot);
        llvm::sys::path::append(vdir, varName);
        llvm::sys::fs::create_directories(vdir);
        variantDir = vdir.str().str();

        llvm::SmallString<256> hloPath(vdir);
        llvm::sys::path::append(hloPath, "01_stablehlo_with_hints.mlir");
        dumpModuleToFile(clonedModule, hloPath);
      }

      // 4. Build and run the sub-pipeline on the clone.
      PassManager pm(context, ModuleOp::getOperationName(),
                     PassManager::Nesting::Implicit);
      addRemainingStableHLOPasses(pm);

      if (failed(ttirPipeline->addToPipeline(pm, "", errHandler))) {
        llvm::errs() << "AutoSharding: failed to build ttir pipeline\n";
        if (collectResults) {
          results.push_back(
              {i, formatConfig(configs[i]), false, 0.0});
        }
        continue;
      }

      if (failed(ttnnPipeline->addToPipeline(pm, ttnnOptions, errHandler))) {
        llvm::errs() << "AutoSharding: failed to build ttnn pipeline\n";
        if (collectResults) {
          results.push_back(
              {i, formatConfig(configs[i]), false, 0.0});
        }
        continue;
      }

      if (failed(pm.run(clonedModule))) {
        llvm::errs() << "AutoSharding: config " << i << " "
                     << formatConfig(configs[i]) << " failed to lower\n";
        if (collectResults) {
          results.push_back(
              {i, formatConfig(configs[i]), false, 0.0});
        }
        continue;
      }

      // Dump the lowered TTNN IR.
      if (!variantDir.empty()) {
        llvm::SmallString<256> ttnnPath(variantDir);
        llvm::sys::path::append(ttnnPath, "02_lowered_ttnn.mlir");
        dumpModuleToFile(clonedModule, ttnnPath);
      }


      // 5. Evaluate cost.
      double cost = evaluateCost(clonedModule);
      llvm::errs() << "AutoSharding: config " << i << " "
                   << formatConfig(configs[i]) << " cost=" << cost << "\n";

      if (collectResults) {
        results.push_back({i, formatConfig(configs[i]), true, cost});
      }
      anySucceeded = true;
      if (cost < bestCost) {
        bestCost = cost;
        bestIdx = i;
      }
    }

    if (!anySucceeded) {
      rootModule.emitError(
          "AutoSharding: all sharding configurations failed to lower");
      signalPassFailure();
      return;
    }

    // 6. Apply the winning config to the original module.
    llvm::errs() << "AutoSharding: selected config " << bestIdx << " "
                 << formatConfig(configs[bestIdx])
                 << " with cost=" << bestCost << "\n";
    applyShardingHints(rootModule, configs[bestIdx], meshInfo.meshName,
                       shardAxisName);

    // Write summary file.
    if (!dumpRoot.empty()) {
      llvm::SmallString<256> summaryPath(dumpRoot);
      llvm::sys::path::append(summaryPath, "summary.txt");
      std::error_code ec;
      llvm::raw_fd_ostream fos(summaryPath, ec);
      if (!ec) {
        fos << "Auto Sharding Summary\n";
        fos << "=======================\n";
        fos << "Mesh: " << meshInfo.meshName << "\n";
        fos << "Sharding axis: " << shardAxisName << "\n";
        fos << "Total configs evaluated: " << configs.size() << "\n\n";

        fos << "Config  Sharding                                 Status  Cost\n";
        fos << std::string(72, '-') << "\n";
        for (const auto &r : results) {
          fos << r.idx;
          size_t idxLen = std::to_string(r.idx).size();
          if (idxLen < 8) {
            fos.indent(8 - idxLen);
          }

          fos << r.label;
          size_t labelLen = r.label.size();
          if (labelLen < 42) {
            fos.indent(42 - labelLen);
          }

          if (r.succeeded) {
            fos << "OK      " << llvm::format("%.6f", r.cost);
          } else {
            fos << "FAILED  -";
          }
          if (r.succeeded && r.idx == bestIdx) {
            fos << "  <-- WINNER";
          }
          fos << "\n";
        }

        fos << "\nSelected: config " << bestIdx << " "
            << formatConfig(configs[bestIdx]) << " with cost=" << bestCost
            << "\n";
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::stablehlo
