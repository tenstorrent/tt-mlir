// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <mlir/Pass/PassRegistry.h>

#ifdef GEN_PASS_DECL
#define GEN_PASS_DECL_TTNNOPTIMIZER
#undef GEN_PASS_DECL
#endif // GEN_PASS_DECL

//===----------------------------------------------------------------------===//
// TTNNOptimizer
//===----------------------------------------------------------------------===//
#ifdef GEN_PASS_DECL_TTNNOPTIMIZER
struct TTNNOptimizerOptions {
  llvm::StringMap<InputLayoutOverrideParams> overrideInputLayout =
      llvm::StringMap<InputLayoutOverrideParams>();
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout =
      llvm::StringMap<OutputLayoutOverrideParams>();
  bool memoryLayoutAnalysisEnabled = false;
  MemoryLayoutAnalysisPolicyType memoryLayoutAnalysisPolicy =
      MemoryLayoutAnalysisPolicyType::DFSharding;
  bool memReconfigEnabled = false;
  int64_t maxLegalLayouts = 64;
};
std::unique_ptr<::mlir::Pass> createTTNNOptimizer();
std::unique_ptr<::mlir::Pass> createTTNNOptimizer(TTNNOptimizerOptions options);
#undef GEN_PASS_DECL_TTNNOPTIMIZER
#endif // GEN_PASS_DECL_TTNNOPTIMIZER
#ifdef GEN_PASS_DEF_TTNNOPTIMIZER

namespace impl {
std::unique_ptr<::mlir::Pass> createTTNNOptimizer();
} // namespace impl

namespace impl {
std::unique_ptr<::mlir::Pass> createTTNNOptimizer(TTNNOptimizerOptions options);
} // namespace impl
namespace impl {

// Go through the ops, set sharding specs for each op based on sharding
// analysis, by updating layout attribute of each op.
//
template <typename DerivedT>
class TTNNOptimizerBase : public ::mlir::OperationPass<::mlir::ModuleOp> {
public:
  using Base = TTNNOptimizerBase;

  TTNNOptimizerBase()
      : ::mlir::OperationPass<::mlir::ModuleOp>(
            ::mlir::TypeID::get<DerivedT>()) {}
  TTNNOptimizerBase(const TTNNOptimizerBase &other)
      : ::mlir::OperationPass<::mlir::ModuleOp>(other) {}
  TTNNOptimizerBase &operator=(const TTNNOptimizerBase &) = delete;
  TTNNOptimizerBase(TTNNOptimizerBase &&) = delete;
  TTNNOptimizerBase &operator=(TTNNOptimizerBase &&) = delete;
  ~TTNNOptimizerBase() = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("ttnn-optimizer");
  }
  ::llvm::StringRef getArgument() const override { return "ttnn-optimizer"; }

  ::llvm::StringRef getDescription() const override {
    return "Determine op configurations for maximum performance.";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("TTNNOptimizer");
  }
  ::llvm::StringRef getName() const override { return "TTNNOptimizer"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTNNOptimizerBase<DerivedT>)

  TTNNOptimizerBase(TTNNOptimizerOptions options) : TTNNOptimizerBase() {
    overrideInputLayout = std::move(options.overrideInputLayout);
    overrideOutputLayout = std::move(options.overrideOutputLayout);
    memoryLayoutAnalysisEnabled =
        std::move(options.memoryLayoutAnalysisEnabled);
    memReconfigEnabled = std::move(options.memReconfigEnabled);
    memoryLayoutAnalysisPolicy = std::move(options.memoryLayoutAnalysisPolicy);
    maxLegalLayouts = std::move(options.maxLegalLayouts);
  }

protected:
  ::mlir::Pass::Option<llvm::StringMap<InputLayoutOverrideParams>,
                       mlir::tt::ttnn::InputLayoutOverrideParser>
      overrideInputLayout{
          *this, "insert-memreconfig",
          ::llvm::cl::desc(
              "Manually insert memory reconfig op for specific op's operand."),
          ::llvm::cl::init(llvm::StringMap<InputLayoutOverrideParams>())};
  ::mlir::Pass::Option<llvm::StringMap<OutputLayoutOverrideParams>,
                       mlir::tt::ttnn::OutputLayoutOverrideParser>
      overrideOutputLayout{
          *this, "override-output-layout",
          ::llvm::cl::desc("Override output tensor layout for specific ops."),
          ::llvm::cl::init(llvm::StringMap<OutputLayoutOverrideParams>())};
  ::mlir::Pass::Option<bool> memoryLayoutAnalysisEnabled{
      *this, "memory-layout-analysis-enabled",
      ::llvm::cl::desc("Enable memory layout optimization."),
      ::llvm::cl::init(false)};
  ::mlir::Pass::Option<bool> memReconfigEnabled{
      *this, "memreconfig-enabled",
      ::llvm::cl::desc("Memory layout reconfiguration pass."),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<mlir::tt::MemoryLayoutAnalysisPolicyType,
                       mlir::tt::MemoryLayoutAnalysisPolicyTypeParser>
      memoryLayoutAnalysisPolicy{
          *this, "memory-layout-analysis-policy",
          llvm::cl::desc("Specify policy for memory layout analysis."),
          llvm::cl::init(MemoryLayoutAnalysisPolicyType::DFSharding)};
  ::mlir::Pass::Option<int64_t> maxLegalLayouts{
      *this, "max-legal-layouts",
      ::llvm::cl::desc(
          "Override maximum number of legal layouts for grid analysis."),
      ::llvm::cl::init(64)};

private:
  friend std::unique_ptr<::mlir::Pass> createTTNNOptimizer() {
    return std::make_unique<DerivedT>();
  }

  friend std::unique_ptr<::mlir::Pass>
  createTTNNOptimizer(TTNNOptimizerOptions options) {
    return std::make_unique<DerivedT>(std::move(options));
  }
};
} // namespace impl

std::unique_ptr<::mlir::Pass> createTTNNOptimizer() {
  return impl::createTTNNOptimizer();
}

std::unique_ptr<::mlir::Pass>
createTTNNOptimizer(TTNNOptimizerOptions options) {
  return impl::createTTNNOptimizer(std::move(options));
}
#undef GEN_PASS_DEF_TTNNOPTIMIZER
#endif // GEN_PASS_DEF_TTNNOPTIMIZER

//===----------------------------------------------------------------------===//
// TTNNOptimizer Registration
//===----------------------------------------------------------------------===//
#ifdef GEN_PASS_REGISTRATION
inline void registerTTNNOptimizer() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createTTNNOptimizer(); });
}
#undef GEN_PASS_REGISTRATION
#endif // GEN_PASS_REGISTRATION
