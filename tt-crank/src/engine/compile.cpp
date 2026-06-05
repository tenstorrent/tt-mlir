#include "engine/compile.hpp"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>

#include "assert.hpp"
#include "config.hpp"
#include "misc.hpp"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <ttmlir/Support/IRHasher.h>

#include <tracy/Tracy.hpp>

#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>
#include <ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h>
#include <ttmlir/RegisterAll.h>
#include <ttmlir/Target/TTNN/TTNNToFlatbuffer.h>

namespace tt::kurbla {

namespace {

// Serializes compile across threads and rejects same-thread re-entry. The
// process-wide MLIRContext is not thread-safe. The mutex is non-recursive, so
// re-entry would deadlock; the per-thread `active_` flag is checked before
// locking to report it instead.
class MLIRCompileGuard {
public:
    MLIRCompileGuard() {
        TT_FATAL(!m_active, "compile must not be re-entered on the same thread");
        m_lock = std::unique_lock<std::mutex>(m_mutex);
        m_active = true;
    }
    ~MLIRCompileGuard() { m_active = false; }
    MLIRCompileGuard(const MLIRCompileGuard &) = delete;
    MLIRCompileGuard &operator=(const MLIRCompileGuard &) = delete;

private:
    inline static std::mutex m_mutex;
    inline static thread_local bool m_active = false;
    std::unique_lock<std::mutex> m_lock;
};

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

// Hash module by hashing all functions in it.
std::string hash_module(mlir::ModuleOp module_op) {
    llvm::SHA256 sha;
    module_op->walk([&](mlir::func::FuncOp func) { sha.update(llvm::StringRef(mlir::tt::hashFuncOp(func))); });
    return llvm::toHex(sha.final());
}

// Compiler cache.
class CompilerCache {
public:
    CompiledProgram *operator[](const std::string &key) {
        if (!comp_cache_enabled()) {
            return nullptr;
        }

        if (monitor<mc_comp_cache>(m_cache.contains(key))) {
            return &m_cache.at(key);
        }

        return nullptr;
    }

    void insert(const std::string &key, CompiledProgram cp) {
        if (comp_cache_enabled()) {
            m_cache.emplace(key, std::move(cp));
        }
    }

private:
    std::unordered_map<std::string, CompiledProgram> m_cache;
};

CompilerCache cache; // NOLINT

// Runs the TTIR-to-TTNN runtime pipeline on `module` and emits a flatbuffer.
// Assumes the caller has already installed a ScopedDiagnosticHandler that
// writes captured diagnostics into `diag_buffer`. The module is mutated in
// place; on success it contains TTNN ops.
CompiledProgram run_ttir_to_ttnn_and_emit(mlir::ModuleOp module_op, const CompileOptions &options,
                                          const std::string &diag_buffer) {
    MLIRCompileGuard guard;

    // Hash before any IR mutations so the key reflects the original TTIR.
    std::string key = hash_module(module_op);
    if (auto *entry = cache[key]) {
        return *entry;
    }

    // Pre-attach lets TTCoreRegisterDevicePass take the mockArch branch and
    // keep our attr; the path overload would overwrite it.
    if (options.system_desc.has_value()) {
        auto diag_fn = [&]() -> mlir::InFlightDiagnostic { return module_op->emitOpError(); };
        auto attr_or = mlir::tt::ttcore::SystemDescAttr::getFromBuffer(module_op.getContext(),
                                                                       options.system_desc->handle.get(), diag_fn);
        TT_FATAL(mlir::succeeded(attr_or), "{}",
                 make_error_message("failed to attach in-memory system desc", diag_buffer));
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
        TT_FATAL(mlir::succeeded(pm.run(module_op)), "{}",
                 make_error_message("ttir-to-ttnn pipeline failed", diag_buffer));
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
        TT_FATAL(fb != nullptr, "{}", make_error_message("ttnnToFlatbuffer returned null", diag_buffer));
    }

    CompiledProgram prog{std::move(fb)};
    cache.insert(key, prog);

    return prog;
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
    TT_FATAL(module_op, "{}", make_error_message("failed to parse TTIR module", diag_buffer));

    return run_ttir_to_ttnn_and_emit(module_op.get(), options, diag_buffer);
}

} // namespace tt::kurbla
