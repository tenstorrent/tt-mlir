// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TEST_UNITTESTS_OPTIMIZER_L1SPILLTESTFIXTURE_H
#define TEST_UNITTESTS_OPTIMIZER_L1SPILLTESTFIXTURE_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1SpillObserver.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"

#include "gtest/gtest.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mlir::tt::ttnn::test {

using op_constraint_validation::ValidationResult;
using op_constraint_validation::ValidationStatus;

/// Kibibyte constant (IEC binary prefix: 1 KiB = 1024 bytes). Used as a
/// readable size multiplier in tests — `1300 * kKiB ≈ 1.27 MiB`, in the
/// ball-park of one L1 bank's scratch space.
inline constexpr uint64_t kKiB = 1024;

//===----------------------------------------------------------------------===//
// RecordingObserver
//
// Captures every L1SpillObserver callback into typed vectors so tests can
// assert on the full sequence of pass decisions without inspecting the IR.
//===----------------------------------------------------------------------===//
class RecordingObserver : public L1SpillObserver {
public:
  struct EvictionRecord {
    Operation *victim;
    int64_t pos;
    uint64_t freedBytes;
  };

  struct OOMRecord {
    Operation *op;
    int64_t pos;
    uint64_t occupiedL1;
  };

  struct LiveRecord {
    Operation *op;
    int64_t pos;
    uint64_t opL1Usage;
    int64_t lastUse;
    uint64_t occupiedAfter;
  };

  struct SelfSpillRecord {
    Operation *op;
    int64_t pos;
  };

  struct DemotionRecord {
    Operation *op;
    int64_t pos;
    bool success;
  };

  // Accumulated events in callback order.
  std::vector<EvictionRecord> evictions;
  std::vector<OOMRecord> ooms;
  std::vector<LiveRecord> lives;
  std::vector<SelfSpillRecord> selfSpills;
  std::vector<DemotionRecord> demotions;
  size_t totalSpillsAtEnd = 0;
  uint64_t finalOccupied = 0;

  void onLiveAdded(Operation *op, int64_t pos, uint64_t opL1Usage,
                   int64_t lastUse, uint64_t occupiedAfter) override {
    lives.push_back({op, pos, opL1Usage, lastUse, occupiedAfter});
  }

  void onOOM(Operation *op, int64_t pos, uint64_t occupiedL1) override {
    ooms.push_back({op, pos, occupiedL1});
  }

  void onEviction(Operation *victim, int64_t pos,
                  uint64_t freedBytes) override {
    evictions.push_back({victim, pos, freedBytes});
  }

  void onSelfSpill(Operation *op, int64_t pos) override {
    selfSpills.push_back({op, pos});
  }

  void onDemotion(Operation *op, int64_t pos, bool success,
                  uint64_t /*newL1Usage*/) override {
    demotions.push_back({op, pos, success});
  }

  void onSpillEnd(size_t totalSpills, uint64_t finalOcc,
                  size_t /*liveTensors*/) override {
    totalSpillsAtEnd = totalSpills;
    finalOccupied = finalOcc;
  }
};

//===----------------------------------------------------------------------===//
// L1SpillTestFixture
//
// gtest base class. Provides:
//  - MLIR context with TTCore + TTNN + Func dialects loaded.
//  - Helpers to build a func::FuncOp with typed TTNN op graphs.
//  - Per-op test configuration (setL1Usage, forceOOM, forceNotImplemented).
//  - run() to construct and drive L1SpillManagement<SumL1MemoryTracker> with
//    a backend validator lambda built from perOpConfigs.
//  - Post-run assertion helpers that inspect the mutated IR.
//===----------------------------------------------------------------------===//
class L1SpillTestFixture : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;
  uint64_t l1BudgetPerCore = 1300 * kKiB;

  struct PerOpConfig {
    std::optional<ValidationStatus> forceStatus;
    uint64_t outputL1Usage = 0;
    uint64_t cbPeakUsage = 0;     // CB peak when output is L1-sharded
    uint64_t dramCBPeakUsage = 0; // CB peak when output is DRAM-interleaved
                                  // (used by evictForDramCBGrowth probe)
    // Hard sharded-input constraint (e.g. paged_update_cache): when set, the op
    // returns MetalBackendError if any input is DRAM. Drives the
    // constraint-driven reshard path once a producer is spilled.
    bool requiresShardedInput = false;
  };
  llvm::DenseMap<mlir::Operation *, PerOpConfig> perOpConfigs;

  void SetUp() override {
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<TTNNDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
  }

  void TearDown() override {}

  // --- Per-op configuration helpers ---

  void setL1Usage(mlir::Operation *op, uint64_t l1, uint64_t cb = 0) {
    perOpConfigs[op].outputL1Usage = l1;
    perOpConfigs[op].cbPeakUsage = cb;
  }

  void setDramCBPeak(mlir::Operation *op, uint64_t cb) {
    perOpConfigs[op].dramCBPeakUsage = cb;
  }

  void forceOOM(mlir::Operation *op) {
    perOpConfigs[op].forceStatus = ValidationStatus::OutOfMemoryError;
  }

  void forceNotImplemented(mlir::Operation *op) {
    perOpConfigs[op].forceStatus = ValidationStatus::NotImplemented;
  }

  /// Mark `op` as requiring sharded (L1) inputs: it fails validation once any
  /// input is DRAM (i.e. after its producer is spilled).
  void setRequiresShardedInput(mlir::Operation *op) {
    perOpConfigs[op].requiresShardedInput = true;
  }

  // --- Layout helpers ---

  /// Default grid {8, 1} is the canonical HeightSharded layout (M, 1).
  TTNNLayoutAttr makeL1Sharded(llvm::ArrayRef<int64_t> shape,
                               llvm::ArrayRef<int64_t> grid = {8, 1}) {
    auto elemType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());
    auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
    return TTNNLayoutAttr::Builder(&context, shape, elemType)
        .setBufferType(BufferType::L1)
        .setMemoryLayout(TensorMemoryLayout::HeightSharded)
        .setGridShape(grid)
        .buildWithCanonicalCorePlacement(deviceAttr);
  }

  /// Per-core L1 bytes for `layout` (matches how the pass sizes reshards).
  uint64_t perCoreL1Usage(TTNNLayoutAttr layout) const {
    return utils::getPerCoreL1Usage(
        layout, ttmlir::utils::volume(layout.getGridShape()));
  }

  TTNNLayoutAttr makeDRAM(llvm::ArrayRef<int64_t> shape) {
    auto elemType = mlir::tt::ttcore::TileType::get(builder.getBF16Type());
    auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
    return TTNNLayoutAttr::Builder(&context, shape, elemType)
        .setBufferType(BufferType::DRAM)
        .setMemoryLayout(TensorMemoryLayout::Interleaved)
        .buildWithCanonicalCorePlacement(deviceAttr);
  }

  mlir::RankedTensorType tensorType(llvm::ArrayRef<int64_t> shape,
                                    TTNNLayoutAttr layout) {
    return mlir::RankedTensorType::get(shape, builder.getBF16Type(), layout);
  }

  // --- Func builder ---

  llvm::SmallVector<mlir::Value>
  beginFunc(llvm::ArrayRef<mlir::RankedTensorType> argTypes,
            llvm::StringRef name = "test") {
    llvm::SmallVector<mlir::Type> types(argTypes.begin(), argTypes.end());
    auto funcType = builder.getType<mlir::FunctionType>(mlir::TypeRange(types),
                                                        mlir::TypeRange({}));
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), name,
                                              funcType);
    mlir::Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);

    llvm::SmallVector<mlir::Value> args;
    for (unsigned i = 0; i < argTypes.size(); ++i) {
      args.push_back(block->getArgument(i));
    }
    return args;
  }

  mlir::Operation *addUnary(mlir::Value input, mlir::RankedTensorType outType,
                            uint64_t l1UsageBytes) {
    auto op = builder.create<ReluOp>(builder.getUnknownLoc(), outType, input);
    setL1Usage(op.getOperation(), l1UsageBytes);
    return op.getOperation();
  }

  mlir::Operation *addBinary(mlir::Value lhs, mlir::Value rhs,
                             mlir::RankedTensorType outType,
                             uint64_t l1UsageBytes) {
    auto op = builder.create<AddOp>(builder.getUnknownLoc(), outType, lhs, rhs);
    setL1Usage(op.getOperation(), l1UsageBytes);
    return op.getOperation();
  }

  /// Add a ReshapeOp whose output shape qualifies for `canReshapeBeView`
  /// when the caller picks a tile-aligned shape with matching last dim.
  mlir::Operation *addReshape(mlir::Value input, mlir::RankedTensorType outType,
                              uint64_t l1UsageBytes) {
    auto outShape = outType.getShape();
    llvm::SmallVector<int32_t> shapeI32(outShape.begin(), outShape.end());
    auto shapeAttr = builder.getI32ArrayAttr(shapeI32);
    auto op = builder.create<ReshapeOp>(builder.getUnknownLoc(), outType, input,
                                        shapeAttr);
    setL1Usage(op.getOperation(), l1UsageBytes);
    return op.getOperation();
  }

  void finishFunc(llvm::ArrayRef<mlir::Value> returnVals) {
    // Update the FunctionType FIRST so the ReturnOp's operands are
    // consistent with the function signature at creation time; otherwise
    // any intermediate verification would see a mismatch.
    llvm::SmallVector<mlir::Type> retTypes;
    for (auto v : returnVals) {
      retTypes.push_back(v.getType());
    }
    auto argTypes = func.getFunctionType().getInputs();
    func.setType(builder.getType<mlir::FunctionType>(argTypes, retTypes));
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), returnVals);
  }

  // --- Backend validator builder ---

  /// Build the lambda installed into SumL1MemoryTracker::backendValidator.
  /// Decision logic:
  ///   1. If perOpConfigs[op].forceStatus is set → return that status.
  ///   2. Else if additionalL1 + outputL1Usage > l1BudgetPerCore → OOM.
  ///   3. Else → Success with (outputL1Usage, cbPeakUsage) and the op's
  ///      current result-type layout.
  SumL1MemoryTracker::BackendValidatorFn makeValidator() const {
    uint64_t budget = l1BudgetPerCore;
    return [this, budget](mlir::Operation *op,
                          llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                          const OpConfig &config,
                          uint64_t additionalL1) -> ValidationResult {
      auto it = perOpConfigs.find(op);
      uint64_t outL1 = 0;
      if (it != perOpConfigs.end()) {
        outL1 = it->second.outputL1Usage;
      } else if (op->getNumResults() > 0) {
        // Pass-inserted reshard (no explicit setL1Usage): default to its
        // layout's per-core L1 so it occupies a real address-sim slot. DRAM
        // spills contribute 0 via the configIsDRAM branch below.
        if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(
                op->getResult(0).getType())) {
          if (auto enc =
                  mlir::dyn_cast_or_null<TTNNLayoutAttr>(rt.getEncoding())) {
            if (enc.hasL1BufferType()) {
              outL1 = perCoreL1Usage(enc);
            }
          }
        }
      }

      // CB peak depends on whether the probed config is L1 or DRAM. The pass
      // queries DRAM CB after demotion via evictForDramCBGrowth — read
      // dramCBPeakUsage in that case so the cushion math behaves like real
      // tt-metal (DRAM CB can be larger because it's locally_allocated).
      bool configIsDRAM = false;
      if (config.outputLayout) {
        configIsDRAM = !config.outputLayout.hasL1BufferType();
      }
      uint64_t cbP = 0;
      if (it != perOpConfigs.end()) {
        cbP =
            configIsDRAM ? it->second.dramCBPeakUsage : it->second.cbPeakUsage;
      }

      // DRAM probes (additionalL1=0 by convention) never report OOM in our
      // fake; they only carry CB info back to the pass.
      uint64_t effectiveOutL1 = configIsDRAM ? 0 : outL1;

      if (it != perOpConfigs.end() && it->second.forceStatus.has_value()) {
        ValidationStatus s = *it->second.forceStatus;
        if (s == ValidationStatus::Success) {
          return ValidationResult::success(0, layoutFromOp(op, config),
                                           effectiveOutL1, cbP);
        }
        if (s == ValidationStatus::NotImplemented) {
          return ValidationResult::notImplemented("injected NotImplemented");
        }
        if (s == ValidationStatus::OutOfMemoryError) {
          return ValidationResult::outOfMemoryError("injected OOM");
        }
        return ValidationResult::metalBackendError("injected backend error");
      }

      // Hard sharded-input constraint: reject a DRAM input (triggers the
      // constraint-driven reshard once a producer is spilled).
      if (it != perOpConfigs.end() && it->second.requiresShardedInput) {
        for (TTNNLayoutAttr inLayout : inputLayouts) {
          if (inLayout && !inLayout.hasL1BufferType()) {
            return ValidationResult::metalBackendError(
                "requires sharded input");
          }
        }
      }

      if (additionalL1 + effectiveOutL1 > budget) {
        return ValidationResult::outOfMemoryError("budget exceeded");
      }
      return ValidationResult::success(0, layoutFromOp(op, config),
                                       effectiveOutL1, cbP);
    };
  }

  // --- Pass execution ---

  /// Pass instance kept alive as a fixture member so the observer (owned by
  /// the pass via unique_ptr) outlives run() and stays accessible through
  /// the returned raw pointer.
  std::unique_ptr<L1SpillManagement<SumL1MemoryTracker>> pass;

  struct RunResult {
    RecordingObserver *observer;
  };

  /// Run L1SpillManagement<SumL1MemoryTracker> on `func` with the test
  /// validator installed.
  RunResult run() {
    auto obs = std::make_unique<RecordingObserver>();
    auto *rawObs = obs.get();

    auto deviceAttr = mlir::tt::ttcore::lookupDevice(module.get());
    ttcore::GridAttr deviceGrid = deviceAttr.getWorkerGrid();

    pass = std::make_unique<L1SpillManagement<SumL1MemoryTracker>>(
        func, deviceGrid, l1BudgetPerCore, std::move(obs));
    pass->getMemoryTracker().backendValidator = makeValidator();
    pass->run();
    return {rawObs};
  }

  // --- IR inspection helpers ---

  /// Count ToMemoryConfigOp that write to DRAM (i.e. spills inserted by pass).
  size_t countSpills() {
    size_t n = 0;
    func.walk([&](ToMemoryConfigOp op) {
      auto rt =
          mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
      if (!rt) {
        return;
      }
      auto enc = mlir::dyn_cast_or_null<TTNNLayoutAttr>(rt.getEncoding());
      if (enc && enc.getBufferType() == BufferType::DRAM) {
        ++n;
      }
    });
    return n;
  }

  /// Return true if `v`'s direct user is a spill-to-DRAM ToMemoryConfigOp.
  bool wasSpilled(mlir::Value v) {
    for (auto *user : v.getUsers()) {
      auto mc = mlir::dyn_cast<ToMemoryConfigOp>(user);
      if (!mc) {
        continue;
      }
      auto rt =
          mlir::dyn_cast<mlir::RankedTensorType>(mc.getResult().getType());
      if (!rt) {
        continue;
      }
      auto enc = mlir::dyn_cast_or_null<TTNNLayoutAttr>(rt.getEncoding());
      if (enc && enc.getBufferType() == BufferType::DRAM) {
        return true;
      }
    }
    return false;
  }

  /// Return true if `v`'s defining op still has an L1 result layout (i.e. it
  /// was NOT demoted in place to DRAM). Distinct from wasSpilled, which looks
  /// for an inserted ToMemoryConfigOp user — demoteToDram mutates the result
  /// type directly without inserting a spill op.
  bool resultIsL1(mlir::Value v) {
    auto rt = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!rt) {
      return false;
    }
    auto enc = mlir::dyn_cast_or_null<TTNNLayoutAttr>(rt.getEncoding());
    return enc && enc.hasL1BufferType();
  }

  /// Count all ops of a given kind in func.
  template <typename OpT>
  size_t countOps() {
    size_t n = 0;
    func.walk([&](OpT /*op*/) { ++n; });
    return n;
  }

private:
  /// Return the TTNNLayoutAttr encoded in op->getResult(0)'s type, or fall
  /// back to config.outputLayout. The pass uses this to write the layout
  /// back via applyOutputConfig; returning the same layout means the IR is
  /// not modified by validation.
  static TTNNLayoutAttr layoutFromOp(mlir::Operation *op,
                                     const OpConfig &config) {
    if (op->getNumResults() > 0) {
      if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(
              op->getResult(0).getType())) {
        if (auto enc =
                mlir::dyn_cast_or_null<TTNNLayoutAttr>(rt.getEncoding())) {
          return enc;
        }
      }
    }
    return config.outputLayout;
  }
};

} // namespace mlir::tt::ttnn::test

#endif // TEST_UNITTESTS_OPTIMIZER_L1SPILLTESTFIXTURE_H
