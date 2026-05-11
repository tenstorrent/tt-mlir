#include "engine/compile.hpp"

#include <memory>
#include <utility>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>
#include <ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h>
#include <ttmlir/RegisterAll.h>
#include <ttmlir/Target/TTNN/TTNNToFlatbuffer.h>

namespace tt::kurbla {

namespace {

// Process-wide MLIR state. Constructed once on first compile() call.
// registerAllPasses() writes into LLVM's global registry, so we keep this
// in one place to avoid duplicate registrations.
struct EngineState {
    mlir::MLIRContext context;

    EngineState() {
        mlir::tt::registerAllPasses();
        mlir::DialectRegistry registry;
        mlir::tt::registerAllDialects(registry);
        mlir::tt::registerAllExtensions(registry);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
    }
};

EngineState &engine_state() {
    static EngineState state;
    return state;
}

mlir::tt::ttcore::Arch to_ttcore_arch(CompileOptions::MockArch arch) {
    switch (arch) {
        case CompileOptions::MockArch::WormholeB0:
            return mlir::tt::ttcore::Arch::WormholeB0;
        case CompileOptions::MockArch::Blackhole:
            return mlir::tt::ttcore::Arch::Blackhole;
    }
    return mlir::tt::ttcore::Arch::WormholeB0;
}

std::string make_error_message(std::string_view fallback, const std::string &captured) {
    if (captured.empty()) {
        return std::string(fallback);
    }
    return captured;
}

// Runs the TTIR-to-TTNN runtime pipeline on `module` and emits a flatbuffer.
// Assumes the caller has already installed a ScopedDiagnosticHandler that
// writes captured diagnostics into `diag_buffer`. The module is mutated in
// place; on success it contains TTNN ops.
CompiledProgram run_ttir_to_ttnn_and_emit(mlir::ModuleOp module, const CompileOptions &options,
                                          const std::string &diag_buffer) {
    // Pre-attach lets TTCoreRegisterDevicePass take the mockArch branch and
    // keep our attr; the path overload would overwrite it.
    if (options.system_desc.has_value()) {
        auto diag_fn = [&]() -> mlir::InFlightDiagnostic { return module->emitOpError(); };
        auto attr_or = mlir::tt::ttcore::SystemDescAttr::getFromBuffer(module.getContext(),
                                                                       options.system_desc->handle.get(), diag_fn);
        if (mlir::failed(attr_or)) {
            throw PipelineError(make_error_message("failed to attach in-memory system desc", diag_buffer));
        }
        // FailureOr hides has_value()/operator bool, so clang-tidy can't see the gate above.
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        module->setAttr(mlir::tt::ttcore::SystemDescAttr::name, attr_or.value());
    }

    mlir::tt::ttnn::TTIRToTTNNRuntimePipelineOptions pm_opts;
    pm_opts.optimizationLevel = options.optimization_level;
    pm_opts.systemDescPath = options.system_desc.has_value() ? std::string{} : options.system_desc_path;
    pm_opts.mockSystemDescArch = to_ttcore_arch(options.mock_arch);

    mlir::PassManager pm(module.getContext(), mlir::ModuleOp::getOperationName());
    mlir::tt::ttnn::createTTIRToTTNNRuntimePipeline(pm, pm_opts);

    if (mlir::failed(pm.run(module))) {
        throw PipelineError(make_error_message("ttir-to-ttnn pipeline failed", diag_buffer));
    }

    std::shared_ptr<void> fb = mlir::tt::ttnn::ttnnToFlatbuffer(module);
    if (!fb) {
        throw PipelineError(make_error_message("ttnnToFlatbuffer returned null", diag_buffer));
    }

    return CompiledProgram{tt::runtime::Binary(std::move(fb))};
}

} // namespace

mlir::MLIRContext &mlir_context() {
    return engine_state().context;
}

CompiledProgram compile_ttir_to_ttnn_flatbuffer(mlir::ModuleOp module, const CompileOptions &options) {
    mlir::MLIRContext *ctx = module.getContext();

    std::string diag_buffer;
    llvm::raw_string_ostream diag_stream(diag_buffer);
    mlir::ScopedDiagnosticHandler diag_handler(ctx, [&](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        diag.print(diag_stream);
        diag_stream << "\n";
        return mlir::success();
    });

    return run_ttir_to_ttnn_and_emit(module, options, diag_buffer);
}

CompiledProgram compile_ttir_to_ttnn_flatbuffer(std::string_view ttir, const CompileOptions &options) {
    mlir::MLIRContext &ctx = engine_state().context;

    std::string diag_buffer;
    llvm::raw_string_ostream diag_stream(diag_buffer);
    mlir::ScopedDiagnosticHandler diag_handler(&ctx, [&](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        diag.print(diag_stream);
        diag_stream << "\n";
        return mlir::success();
    });

    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(ttir, &ctx);
    if (!module) {
        throw ParseError(make_error_message("failed to parse TTIR module", diag_buffer));
    }

    return run_ttir_to_ttnn_and_emit(module.get(), options, diag_buffer);
}

} // namespace tt::kurbla
