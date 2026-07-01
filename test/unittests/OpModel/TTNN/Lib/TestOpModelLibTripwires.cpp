// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tripwire tests: pass today, expected to fail when a specific tt-metal
// change lands.  Each test's comment should name the metal issue/PR it
// tracks.  Kept in a separate target so it compiles fast and the file can be
// removed (or individual tripwires dropped) once the upstream change arrives.
//
// This file is currently an empty placeholder -- add a TEST_F here when a new
// tt-metal behavior needs a tripwire.  The PagedUpdateCacheOpWrongGrid
// tripwire lived here until tt-metal #45016 landed; see git history for an
// example of the pattern.

#include "OpModelFixture.h"

namespace mlir::tt::ttnn::op_model {

class OpModelTripwireTest : public OpModelFixture {};

} // namespace mlir::tt::ttnn::op_model
