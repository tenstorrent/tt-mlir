#ifndef TTNN_INITPASSES_H
#define TTNN_INITPASSES_H

#include "ttmlir/Conversion/Passes.h"
// #include "ttmlir/Dialect/EmitC/Pipelines.h"
// #include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Passes.h"

#include <cstdlib>

namespace mlir::tt::ttnn {

// This function may be called to register the MLIR passes with the
// global registry.
// If you're building a compiler, you likely don't need this: you would build a
// pipeline programmatically without the need to register with the global
// registry, since it would already be calling the creation routine of the
// individual passes.
// The global registry is interesting to interact with the command-line tools.

inline void registerAllEmitCPasses() {
  // #ifdef EMITC_BUILD_HLO
  //   registerConvertStablehloRegionOpsToEmitCPass();
  //   registerConvertStablehloToEmitCPass();
  //   registerInsertEmitCStablehloIncludePass();
  //   registerStablehloToEmitCPipeline();
  // #endif // EMITC_BUILD_HLO
  registerConvertTTNNToEmitCPass();
  //   registerTTNNToEmitCPass();
  //   registerConvertArithToEmitCPass();
  //   registerConvertTensorToEmitCPass();
  //   registerConvertTosaToEmitCPass();
  //   registerInsertEmitCArithIncludePass();
  //   registerInsertEmitCTensorIncludePass();
  //   registerInsertEmitCTosaIncludePass();
  //   registerArithToEmitCPipeline();
  //   registerTensorToEmitCPipeline();
  //   registerTosaToEmitCPipeline();
}

} // namespace mlir::tt::ttnn

#endif // TTNN_INITPASSES_H
