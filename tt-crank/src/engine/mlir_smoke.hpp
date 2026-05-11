#pragma once

namespace tt::kurbla {

// Smoke test for the tt-mlir integration: calls an exported entry point from
// libTTMLIRCompiler.so. Exists only to prove that tt-mlir headers are reachable
// at compile time and the shared library is wired up at link time.
bool mlir_smoke();

} // namespace tt::kurbla
