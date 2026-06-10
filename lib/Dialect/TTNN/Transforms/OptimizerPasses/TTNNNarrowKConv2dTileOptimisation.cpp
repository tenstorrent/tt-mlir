// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TTNNNarrowKConv2dTileOptimisation — stub pass.
//
// This pass is disabled and does nothing. The symbol must exist because the
// system forge _C.so was compiled against a version of libTTMLIRCompiler that
// exported createTTNNNarrowKConv2dTileOptimisation.  Removing it causes an
// ImportError at Python import time.
//
// The actual NarrowK optimization (K=3→16 tile padding reduction) requires the
// TTNN tilize device kernel to support TILE_WIDTH != 32, which it currently
// does not.  When that constraint is lifted this stub can be replaced.
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNNARROWKCONV2DTILEOPTIMISATION
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNNarrowKConv2dTileOptimisationPass
    : public impl::TTNNNarrowKConv2dTileOptimisationBase<
          TTNNNarrowKConv2dTileOptimisationPass> {
public:
  using impl::TTNNNarrowKConv2dTileOptimisationBase<
      TTNNNarrowKConv2dTileOptimisationPass>::
      TTNNNarrowKConv2dTileOptimisationBase;
  void runOnOperation() final {} // no-op
};
} // namespace mlir::tt::ttnn
