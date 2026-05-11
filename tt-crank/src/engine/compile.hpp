#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <tt/runtime/types.h>

namespace tt::kurbla {

// Output of a successful compile. Wraps tt::runtime::Binary so we can attach
// tt-kurbla-specific metadata later (compile stats, source identity, cache
// key) without changing the public function signature.
struct CompiledProgram {
    tt::runtime::Binary binary;
};

// Options for TTIR-starting pipelines. Shared across all current pipelines
// because every field maps to tt-mlir's TTIRToTTNNCommonPipelineOptions base.
// Per-pipeline option splits arrive only when a future pipeline grows a knob
// the others don't have.
struct CompileOptions {
    // Mock arch used when system_desc_path is empty.
    enum class MockArch { WormholeB0, Blackhole };

    // 0=all optimizer passes off (fastest), 1=optimizer on without sharding,
    // 2=optimizer on with memory layout analysis (sharding). See tt-mlir's
    // TTIRToTTNNCommonPipelineOptions for the precise mapping.
    int optimization_level = 0;

    // Path to a system descriptor flatbuffer. If empty, mock_arch is used.
    std::string system_desc_path;

    MockArch mock_arch = MockArch::WormholeB0;
};

// Returns the process-wide MLIRContext used by tt-kurbla's compile pipelines.
// Use this when building TTIR modules programmatically (e.g. lowering from
// framework IR like PyTorch FX) — the resulting ModuleOp must live in this
// context for the ModuleOp-based compile overload to work.
mlir::MLIRContext& mlir_context();

// Compile TTIR text through the ttir-to-ttnn runtime pipeline and emit a TTNN
// flatbuffer. Throws CompileError (ParseError / PipelineError) on failure with
// captured MLIR diagnostics in what().
//
// Not safe to call from multiple threads concurrently in v1 — the underlying
// MLIRContext is a process-wide singleton without internal locking. Callers
// must serialize.
CompiledProgram compile_ttir_to_ttnn_flatbuffer(std::string_view ttir, const CompileOptions& options = {});

// Compile a pre-built TTIR ModuleOp. The module must live in mlir_context()
// and is mutated in place — on success it holds TTNN ops, not TTIR.
//
// This is the entry point for callers that construct TTIR in memory rather
// than serializing it to text first (e.g. a PyTorch FX → TTIR lowering).
//
// Same threading caveat as the string overload: serialize calls.
CompiledProgram compile_ttir_to_ttnn_flatbuffer(mlir::ModuleOp module, const CompileOptions& options = {});

class CompileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ParseError : public CompileError {
public:
    using CompileError::CompileError;
};

class PipelineError : public CompileError {
public:
    using CompileError::CompileError;
};

} // namespace tt::kurbla
