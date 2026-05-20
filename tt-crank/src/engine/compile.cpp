#include "engine/compile.hpp"

#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>

#include "assert.hpp"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>

#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>
#include <ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h>
#include <ttmlir/RegisterAll.h>
#include <ttmlir/Target/TTNN/TTNNToFlatbuffer.h>

namespace tt::kurbla {

namespace {

// Cheap insurance against concurrent MLIRContext use. The context (and the
// pass-pipeline registry it relies on) is not thread-safe — see compile.hpp's
// note. Until a real mutex is added around compile, the first caller's thread
// becomes the only allowed thread; any other thread entering compile throws
// loudly rather than silently corrupting MLIR's interning tables. Remove this
// the moment compile is properly serialized.
void assert_single_threaded_mlir_access() {
    static const std::thread::id allowed_thread = std::this_thread::get_id();
    if (std::this_thread::get_id() != allowed_thread) {
        throw CompileError("tt-kurbla compile: MLIRContext accessed from a different thread than the first caller. "
                           "The context is not yet thread-safe; serialize calls or add a mutex around compile.");
    }
}

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
    assert_single_threaded_mlir_access();
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
CompiledProgram run_ttir_to_ttnn_and_emit(mlir::ModuleOp module_op, const CompileOptions &options,
                                          const std::string &diag_buffer) {
    // Catches the ModuleOp-overload path of compile_ttir_to_ttnn_flatbuffer:
    // engine_state() isn't called there (the module brings its own context),
    // so the thread check would otherwise be skipped on that path.
    assert_single_threaded_mlir_access();

    // Pre-attach lets TTCoreRegisterDevicePass take the mockArch branch and
    // keep our attr; the path overload would overwrite it.
    if (options.system_desc.has_value()) {
        auto diag_fn = [&]() -> mlir::InFlightDiagnostic { return module_op->emitOpError(); };
        auto attr_or = mlir::tt::ttcore::SystemDescAttr::getFromBuffer(module_op.getContext(),
                                                                       options.system_desc->handle.get(), diag_fn);
        if (mlir::failed(attr_or)) {
            throw PipelineError(make_error_message("failed to attach in-memory system desc", diag_buffer));
        }
        // FailureOr hides has_value()/operator bool, so clang-tidy can't see the gate above.
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        module_op->setAttr(mlir::tt::ttcore::SystemDescAttr::name, attr_or.value());
    }

    mlir::tt::ttnn::TTIRToTTNNRuntimePipelineOptions pm_opts;
    pm_opts.optimizationLevel = options.optimization_level;
    pm_opts.systemDescPath = options.system_desc.has_value() ? std::string{} : options.system_desc_path;
    pm_opts.mockSystemDescArch = to_ttcore_arch(options.mock_arch);

    mlir::PassManager pm(module_op.getContext(), mlir::ModuleOp::getOperationName());
    mlir::tt::ttnn::createTTIRToTTNNRuntimePipeline(pm, pm_opts);

    {
        ZoneScopedN("tt_kurbla::ttir_to_ttnn_pipeline");
        if (mlir::failed(pm.run(module_op))) {
            throw PipelineError(make_error_message("ttir-to-ttnn pipeline failed", diag_buffer));
        }
    }

    // Opt-in dump of the post-pipeline TTNN IR — useful when debugging op
    // lowerings from the torch frontend without rebuilding with verbose passes.
    if (std::getenv("TT_KURBLA_PRINT_TTNN_IR") != nullptr) {
        llvm::errs() << "[tt_kurbla] ===== TTNN module =====\n";
        module_op.print(llvm::errs());
        llvm::errs() << "\n[tt_kurbla] ========================\n";
    }

    std::shared_ptr<void> fb;
    {
        ZoneScopedN("tt_kurbla::ttnn_to_flatbuffer");
        fb = mlir::tt::ttnn::ttnnToFlatbuffer(module_op);
    }
    if (!fb) {
        throw PipelineError(make_error_message("ttnnToFlatbuffer returned null", diag_buffer));
    }

    return CompiledProgram{tt::runtime::Binary(std::move(fb))};
}

} // namespace

mlir::MLIRContext &mlir_context() {
    return engine_state().context;
}

CompiledProgram compile_ttir_to_ttnn_flatbuffer(mlir::ModuleOp module_op, const CompileOptions &options) {
    mlir::MLIRContext *ctx = module_op.getContext();

    std::string diag_buffer;
    llvm::raw_string_ostream diag_stream(diag_buffer);
    mlir::ScopedDiagnosticHandler diag_handler(ctx, [&](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        diag.print(diag_stream);
        diag_stream << "\n";
        return mlir::success();
    });

    return run_ttir_to_ttnn_and_emit(module_op, options, diag_buffer);
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

    mlir::OwningOpRef<mlir::ModuleOp> module_op = mlir::parseSourceString<mlir::ModuleOp>(ttir, &ctx);
    if (!module_op) {
        throw ParseError(make_error_message("failed to parse TTIR module", diag_buffer));
    }

    return run_ttir_to_ttnn_and_emit(module_op.get(), options, diag_buffer);
}

} // namespace tt::kurbla
