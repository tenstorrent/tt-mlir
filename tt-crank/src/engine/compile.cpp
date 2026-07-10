#include "engine/compile.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <tt-logger/tt-logger.hpp>
#include <tt/runtime/types.h>
#include <utility>
#include <vector>

#include "assert.hpp"
#include "config.hpp"
#include "engine/compile_options.hpp"
#include "engine/device.hpp"
#include "misc.hpp"
#include "version.hpp"
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

#include <tt/runtime/runtime.h>

#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>
#include <ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h>
#include <ttmlir/RegisterAll.h>
#include <ttmlir/Target/TTNN/TTNNToFlatbuffer.h>
#include <ttmlir/Target/TTNN/Target.h>

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

std::string make_error_message(std::string_view fallback, const std::string &captured) {
    if (captured.empty()) {
        return std::string(fallback);
    }
    return captured;
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

        if (std::optional<CompiledProgram> program = load_from_disk(key)) {
            return &m_cache.emplace(key, std::move(*program)).first->second;
        }

        return nullptr;
    }

    CompiledProgram &insert(const std::string &key, CompiledProgram cp) {
        CompiledProgram &program = m_cache.emplace(key, std::move(cp)).first->second;
        store_on_disk(key, program);

        return program;
    }

    static void store_on_disk(const std::string &key, const CompiledProgram &cp) {
        if (!comp_cache_on_disk_enabled()) {
            return;
        }

        const char *cache_dir = compile_cache_dir_config();
        if (!std::filesystem::exists(cache_dir)) {
            std::filesystem::create_directories(cache_dir);
        }

        std::string path = cache_dir + key;
        TT_FATAL(!std::filesystem::exists(path), "Binary already stored on disk: {}", path);

        log_info(tt::LogAlways, "Storing binary on disk: {}", path);
        cp.binary.store(path.c_str());
    }

    static std::optional<CompiledProgram> load_from_disk(const std::string &key) {
        if (!comp_cache_on_disk_enabled()) {
            return std::nullopt;
        }

        std::string path = compile_cache_dir_config() + key;
        if (!monitor<mc_comp_cache_on_disk>(std::filesystem::exists(path))) {
            return std::nullopt;
        }

        log_info(tt::LogAlways, "Loading binary from disk: {}", path);
        return CompiledProgram{tt::runtime::Binary::loadFromPath(path.c_str())};
    }

private:
    std::unordered_map<std::string, CompiledProgram> m_cache;
};

CompilerCache cache; // NOLINT

// Calculates sha256 compilation key.
// Key is computed by hashing all functions in module, hashing all pipeline options, and hashing mlir git worktree.
std::string calc_compilation_key(mlir::ModuleOp module_op,
                                 const mlir::tt::ttnn::TTIRToTTNNRuntimePipelineOptions &pm_opts) {
    llvm::SHA256 sha;
    module_op->walk([&](mlir::func::FuncOp func) { sha.update(llvm::StringRef(mlir::tt::hashFuncOp(func))); });

    std::string opts;
    llvm::raw_string_ostream os(opts);
    pm_opts.print(os);
    sha.update(llvm::StringRef(opts));

    sha.update(ttmlir_git_worktree_hash());
    return llvm::toHex(sha.final());
}

void set_pipeline_options(const CompileOptions &options, mlir::tt::ttnn::TTIRToTTNNRuntimePipelineOptions &pm_opts) {
    options.set_options_on(pm_opts);

    const auto mesh_shape = ::tt::kurbla::runtime_device_mesh_shape();
    const auto &mesh_fabric = ::tt::kurbla::runtime_mesh_fabric_config(mesh_shape);

    // Pass in the currently opened device mesh shape - otherwise the CCL ops will hit issues during compilation.
    pm_opts.meshShape = std::vector<std::int64_t>(mesh_shape.begin(), mesh_shape.end());

    // Match CCL topology to what the fabric actually supports per mesh axis.
    std::vector<mlir::tt::ttcore::Topology> mesh_topology;
    mesh_topology.reserve(mesh_fabric.perAxisConfig.size());
    for (const auto axis : mesh_fabric.perAxisConfig) {
        mesh_topology.push_back(axis == ::tt::runtime::FabricConfig::FABRIC_1D_RING
                                    ? mlir::tt::ttcore::Topology::Ring
                                    : mlir::tt::ttcore::Topology::Linear);
    }

    pm_opts.meshTopology = mesh_topology;
}

// Opt-in dump of the TTIR — useful when debugging.
void print_tt_ir(mlir::ModuleOp module_op) {
    if (print_tt_ir_enabled()) {
        llvm::errs() << "[tt_kurbla] ===== TTIR module =====\n";
        module_op.print(llvm::errs());
        llvm::errs() << "\n[tt_kurbla] ========================\n";
    }
}

// Opt-in dump of the post-pipeline TTNN IR.
void print_ttnn_ir(const CompiledProgram &prog) {
    if (!print_ttnn_ir_enabled()) {
        return;
    }

    auto ir = prog.ttnn_ir();
    llvm::errs() << "[tt_kurbla] ===== TTNN module =====\n";
    llvm::errs() << (ir.empty() ? "[tt_kurbla] <no TTNN IR embedded in cached binary>" : ir);
    llvm::errs() << "\n[tt_kurbla] ========================\n";
}

// Prints compile options.
void print_compile_options(const CompileOptions &options) {
    if (print_compile_options_enabled()) {
        log_info(tt::LogAlways, "Compile options: {}", options.to_string());
    }
}

// Runs the TTIR-to-TTNN runtime pipeline on `module` and emits a flatbuffer.
// Assumes the caller has already installed a ScopedDiagnosticHandler that
// writes captured diagnostics into `diag_buffer`. The module is mutated in
// place; on success it contains TTNN ops.
CompiledProgram &run_ttir_to_ttnn_and_emit(mlir::ModuleOp module_op, const CompileOptions &options,
                                           const std::string &diag_buffer) {
    MLIRCompileGuard guard;

    print_tt_ir(module_op);
    print_compile_options(options);

    mlir::tt::ttnn::TTIRToTTNNRuntimePipelineOptions pm_opts;
    set_pipeline_options(options, pm_opts);

    auto key = calc_compilation_key(module_op, pm_opts);
    if (auto *entry = cache[key]) {
        print_ttnn_ir(*entry);
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

    mlir::PassManager pm(module_op.getContext(), mlir::ModuleOp::getOperationName());
    mlir::tt::ttnn::createTTIRToTTNNRuntimePipeline(pm, pm_opts);

    {
        ZoneScopedN("tt_kurbla::ttir_to_ttnn_pipeline");
        TT_FATAL(mlir::succeeded(pm.run(module_op)), "{}",
                 make_error_message("ttir-to-ttnn pipeline failed", diag_buffer));
    }

    std::shared_ptr<void> fb;
    {
        ZoneScopedN("tt_kurbla::ttnn_to_flatbuffer");
        fb = mlir::tt::ttnn::ttnnToFlatbuffer(module_op);
        TT_FATAL(fb != nullptr, "{}", make_error_message("ttnnToFlatbuffer returned null", diag_buffer));
    }

    CompiledProgram &prog = cache.insert(key, CompiledProgram(std::move(fb)));
    print_ttnn_ir(prog);

    return prog;
}

} // namespace

mlir::MLIRContext &mlir_context() {
    return engine_state().context;
}

CompiledProgram::CompiledProgram(tt::runtime::Binary bin)
    : binary(std::move(bin)), input_descs(binary.getProgramInputs(0)), output_descs(binary.getProgramOutputs(0)),
      num_inputs(input_descs.size()) {
    input_layouts.reserve(num_inputs);
    for (std::uint32_t i = 0; i < num_inputs; ++i) {
        input_layouts.push_back(tt::runtime::getLayout(binary, /*program_index=*/0, i));
    }
}

std::string_view CompiledProgram::ttnn_ir() const {
    const void *handle = binary.handle.get();
    if (handle == nullptr || !::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(handle)) {
        return {};
    }

    const auto *fb = ::tt::target::ttnn::GetSizePrefixedTTNNBinary(handle);
    if (fb == nullptr || fb->mlir() == nullptr || fb->mlir()->source() == nullptr) {
        return {};
    }

    return fb->mlir()->source()->c_str();
}

CompiledProgram &compile_ttir_to_ttnn_flatbuffer(mlir::ModuleOp module_op, const CompileOptions &options) {
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

CompiledProgram &compile_ttir_to_ttnn_flatbuffer(std::string_view ttir, const CompileOptions &options) {
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
