// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/GreedyL1SpillManagement.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1SpillManagement.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::ttnn {

namespace impl {

std::unique_ptr<::mlir::Pass> createTTNNGreedyL1SpillManagement();
std::unique_ptr<::mlir::Pass>
createTTNNGreedyL1SpillManagement(TTNNGreedyL1SpillManagementOptions options);

template <typename DerivedT>
class TTNNGreedyL1SpillManagementBase
    : public ::mlir::OperationPass<::mlir::ModuleOp> {
public:
  using Base = TTNNGreedyL1SpillManagementBase;

  TTNNGreedyL1SpillManagementBase()
      : ::mlir::OperationPass<::mlir::ModuleOp>(
            ::mlir::TypeID::get<DerivedT>()) {}
  TTNNGreedyL1SpillManagementBase(const TTNNGreedyL1SpillManagementBase &other)
      : ::mlir::OperationPass<::mlir::ModuleOp>(other) {}
  TTNNGreedyL1SpillManagementBase &
  operator=(const TTNNGreedyL1SpillManagementBase &) = delete;
  TTNNGreedyL1SpillManagementBase(TTNNGreedyL1SpillManagementBase &&) = delete;
  TTNNGreedyL1SpillManagementBase &
  operator=(TTNNGreedyL1SpillManagementBase &&) = delete;
  ~TTNNGreedyL1SpillManagementBase() override = default;

  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("ttnn-greedy-l1-spill-management");
  }
  ::llvm::StringRef getArgument() const override {
    return "ttnn-greedy-l1-spill-management";
  }

  ::llvm::StringRef getDescription() const override {
    return "Belady's algorithm for L1 budget enforcement.";
  }

  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("TTNNGreedyL1SpillManagement");
  }
  ::llvm::StringRef getName() const override {
    return "TTNNGreedyL1SpillManagement";
  }

  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TTNNGreedyL1SpillManagementBase<DerivedT>)

  TTNNGreedyL1SpillManagementBase(TTNNGreedyL1SpillManagementOptions options)
      : TTNNGreedyL1SpillManagementBase() {
    // Future: initialize options from the struct.
  }

protected:
  // Future: add pass options here.

private:
  friend std::unique_ptr<::mlir::Pass> createTTNNGreedyL1SpillManagement() {
    return std::make_unique<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass> createTTNNGreedyL1SpillManagement(
      TTNNGreedyL1SpillManagementOptions options) {
    return std::make_unique<DerivedT>(std::move(options));
  }
};
} // namespace impl

std::unique_ptr<::mlir::Pass> createTTNNGreedyL1SpillManagement() {
  return impl::createTTNNGreedyL1SpillManagement();
}

std::unique_ptr<::mlir::Pass>
createTTNNGreedyL1SpillManagement(TTNNGreedyL1SpillManagementOptions options) {
  return impl::createTTNNGreedyL1SpillManagement(std::move(options));
}

class TTNNGreedyL1SpillManagement
    : public impl::TTNNGreedyL1SpillManagementBase<
          TTNNGreedyL1SpillManagement> {
public:
  using impl::TTNNGreedyL1SpillManagementBase<
      TTNNGreedyL1SpillManagement>::TTNNGreedyL1SpillManagementBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Get L1 budget from system description and usage cap.
    ttcore::GridAttr deviceGrid =
        ttcore::lookupDevice(moduleOp).getWorkerGrid();
    ttcore::ChipDescAttr chipDesc = ttcore::getOpChipDescAttr(moduleOp);
    float tensorL1UsageCap = utils::getTensorL1UsageCap(moduleOp);
    uint64_t l1BudgetPerCore =
        static_cast<uint64_t>(tensorL1UsageCap * chipDesc.getUsableL1Size());

    TTMLIR_TRACE(ttmlir::LogComponent::GreedyOptimizer,
                 "L1 spill management: budget per core = {0} bytes "
                 "(usable = {1}, cap = {2})",
                 l1BudgetPerCore, chipDesc.getUsableL1Size(), tensorL1UsageCap);

    moduleOp->walk([&](func::FuncOp func) {
      if (!ttmlir::utils::isForwardDeviceFunc(func)) {
        return;
      }

      L1SpillManagement<SumL1MemoryTracker> spill(func, deviceGrid,
                                                  l1BudgetPerCore);
      spill.run();
    });
  }
};

} // namespace mlir::tt::ttnn
