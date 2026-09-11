#pragma once

#include "assert.hpp"
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <tt/runtime/types.h>

#include "compile_options.hpp"
#include "tt_kurbla_export.hpp"

namespace tt::kurbla {

struct TT_KURBLA_API CompiledProgram {
    CompiledProgram(tt::runtime::Binary binary);

    tt::runtime::Binary binary;

    // Input tensors descriptors.
    std::vector<tt::runtime::TensorDesc> input_descs;

    // Output tensors descriptors.
    std::vector<tt::runtime::TensorDesc> output_descs;

    // Number of program inputs.
    std::size_t num_inputs;

    // Input layouts for @main function.
    std::vector<tt::runtime::Layout> input_layouts;

    const tt::runtime::Layout &input_layout_at(std::size_t idx) const {
        TT_FATAL(idx < input_layouts.size(), "Index out of bounds.");
        return input_layouts[idx];
    }

    std::uint32_t num_programs() const { return binary.getNumPrograms(); }
    std::string program_name(std::uint32_t program_index) const { return binary.getProgramName(program_index); }

    // Extract the IR from the binary.
    // NOTE: this is the final IR associated in this binary, it isn't necessarily
    // TTNN IR, but since we only lower to TTNN, in our case it is (for now).
    std::string_view ttnn_ir() const;
};

// `CompiledProgram` with additional metadata produced by the compilation.
struct TT_KURBLA_API CompileResult {
    // Borrowed: the program is owned by the process-wide compile cache and
    // outlives this result. Never null.
    CompiledProgram *program;

    // The TTIR the program was compiled from. Populated when the caller passes
    // `capture_ttir`, empty otherwise.
    std::string ttir;

    // True when the program came straight out of the compile cache, i.e. no
    // pipeline ran.
    bool cache_hit;

    // Wall-clock time the engine spent producing this result. On a cache hit
    // only the lookup ran, so this is near zero.
    std::chrono::duration<double, std::milli> compile_duration{};
};

// Returns the process-wide MLIRContext used by tt-kurbla's compile pipelines.
// Use this when building TTIR modules programmatically (e.g. lowering from
// framework IR like PyTorch FX) — the resulting ModuleOp must live in this
// context for the ModuleOp-based compile overload to work.
TT_KURBLA_API mlir::MLIRContext &mlir_context();

// Compile TTIR text through the ttir-to-ttnn runtime pipeline and emit a TTNN
// flatbuffer. `capture_ttir` copies `ttir` into the result — no re-print needed,
// the caller already handed us the text.
//
// Not safe to call from multiple threads concurrently in v1 — the underlying
// MLIRContext is a process-wide singleton without internal locking. Callers
// must serialize.
TT_KURBLA_API CompileResult compile_ttir_to_ttnn_flatbuffer(std::string_view ttir, const CompileOptions &options = {},
                                                            bool capture_ttir = false);

// Compile a pre-built TTIR ModuleOp. The module must live in mlir_context()
// and is mutated in place — on success it holds TTNN ops, not TTIR.
//
// When `capture_ttir` is set, the TTIR text will be created from the module
// and stored in the `CompileResult`.
//
// Same threading caveat as the string overload: serialize calls.
TT_KURBLA_API CompileResult compile_ttir_to_ttnn_flatbuffer(mlir::ModuleOp module_op,
                                                            const CompileOptions &options = {},
                                                            bool capture_ttir = false);

} // namespace tt::kurbla
