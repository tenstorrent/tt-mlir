// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_EXECUTOR_H
#define RUNTIME_LIB_TTMETAL_EXECUTOR_H

#define FMT_HEADER_ONLY
#include "tt-metalium/mesh_device.hpp"

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTMetal/Target.h"

namespace tt::runtime::ttmetal {

std::vector<Tensor>
executeDeviceProgram(::tt::tt_metal::IDevice *device,
                     const ::tt::target::metal::DeviceProgram *program,
                     const std::vector<Tensor> &inputs,
                     common::DylibManager &&dylibs);

} // namespace tt::runtime::ttmetal

#endif // RUNTIME_LIB_TTMETAL_EXECUTOR_H
