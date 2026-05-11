#include "engine/mlir_smoke.hpp"

#include <ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h>

namespace tt::kurbla {

bool mlir_smoke() {
    // Touches an entry point exported by libTTMLIRCompiler.so to prove link.
    // The TTIR pipeline registration is idempotent, so calling it on every
    // invocation is fine for a smoke test.
    mlir::tt::ttir::registerTTIRPipelines();
    return true;
}

} // namespace tt::kurbla
