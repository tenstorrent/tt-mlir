
#ifndef TTMLIR_CONVERSION_PASSES_H
#define TTMLIR_CONVERSION_PASSES_H

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_REGISTRATION
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttnn

#endif // TTNN_CONVERSION_PASSES_H
