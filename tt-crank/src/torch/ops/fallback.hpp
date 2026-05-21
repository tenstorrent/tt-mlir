#pragma once

namespace tt::kurbla::torch_backend {

// Strict mode: when true, the global CPU fallback raises TORCH_CHECK instead
// of running. Used by tests to assert that a code path uses native tt kernels
// for every op. Idempotent and process-wide.
void set_fallback_strict(bool strict);
bool fallback_strict();

} // namespace tt::kurbla::torch_backend
