#pragma once

#include <tt/runtime/types.h>

#include "tt_kurbla_export.hpp"

namespace tt::kurbla {

// Process-wide MeshDevice, opened lazily on first call. Lifted out of
// execution_payload.cpp so the torch backend (and any future frontend) can
// call tt::runtime::toLayout/toHost without going through ExecutionPayload.
// One open per process; closed at static-destruction time.
TT_KURBLA_API ::tt::runtime::Device &runtime_device();

// Process-wide SystemDesc, probed once against `runtime_device()` and cached.
// Used as the compile target; matches the actually-open mesh's topology so
// downstream lowering and the runtime agree on chip count / layout.
// First access opens the device (via runtime_device()) before probing — both
// state items share the same DeviceState singleton, so init order is fixed.
TT_KURBLA_API const ::tt::runtime::SystemDesc &runtime_system_desc();

} // namespace tt::kurbla
