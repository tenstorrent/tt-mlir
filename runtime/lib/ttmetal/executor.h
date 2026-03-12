// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_EXECUTOR_H
#define RUNTIME_LIB_TTMETAL_EXECUTOR_H

#include "executor_utils.h"

#define FMT_HEADER_ONLY
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTMetal/Target.h"

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

std::vector<Tensor>
executeMeshDeviceProgram(::tt::tt_metal::distributed::MeshDevice *meshDevice,
                         const ::tt::target::metal::DeviceProgram *program,
                         const std::vector<Tensor> &inputs,
                         common::DylibManager &&dylibs);

} // namespace tt::runtime::ttmetal

#endif // RUNTIME_LIB_TTMETAL_EXECUTOR_H
