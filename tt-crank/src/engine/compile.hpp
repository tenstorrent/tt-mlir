#pragma once

#include "assert.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <tt/runtime/types.h>

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
};

// Options for TTIR-starting pipelines. Shared across all current pipelines
// because every field maps to tt-mlir's TTIRToTTNNCommonPipelineOptions base.
// Per-pipeline option splits arrive only when a future pipeline grows a knob
// the others don't have.
struct TT_KURBLA_API CompileOptions {
    enum class MockArch { WormholeB0, Blackhole };

    // 0=all optimizer passes off (fastest), 1=optimizer on without sharding,
    // 2=optimizer on with memory layout analysis (sharding). See tt-mlir's
    // TTIRToTTNNCommonPipelineOptions for the precise mapping.
    int optimization_level = 0;

    // Precedence: system_desc > system_desc_path > mock_arch.
    std::optional<tt::runtime::SystemDesc> system_desc;
    std::string system_desc_path;
    MockArch mock_arch = MockArch::WormholeB0;
};

// Returns the process-wide MLIRContext used by tt-kurbla's compile pipelines.
// Use this when building TTIR modules programmatically (e.g. lowering from
// framework IR like PyTorch FX) — the resulting ModuleOp must live in this
// context for the ModuleOp-based compile overload to work.
TT_KURBLA_API mlir::MLIRContext &mlir_context();

// Compile TTIR text through the ttir-to-ttnn runtime pipeline and emit a TTNN
// flatbuffer.
//
// Not safe to call from multiple threads concurrently in v1 — the underlying
// MLIRContext is a process-wide singleton without internal locking. Callers
// must serialize.
TT_KURBLA_API CompiledProgram &compile_ttir_to_ttnn_flatbuffer(std::string_view ttir,
                                                               const CompileOptions &options = {});

// Compile a pre-built TTIR ModuleOp. The module must live in mlir_context()
// and is mutated in place — on success it holds TTNN ops, not TTIR.
//
// This is the entry point for callers that construct TTIR in memory rather
// than serializing it to text first (e.g. a PyTorch FX → TTIR lowering).
//
// Same threading caveat as the string overload: serialize calls.
TT_KURBLA_API CompiledProgram &compile_ttir_to_ttnn_flatbuffer(mlir::ModuleOp module_op,
                                                               const CompileOptions &options = {});

} // namespace tt::kurbla
