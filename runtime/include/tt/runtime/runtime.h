// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_RUNTIME_H
#define TT_RUNTIME_RUNTIME_H

#include <cstdint>
#include <functional>
#include <vector>

#include "tt/runtime/types.h"
#include "ttmlir/Target/Common/types_generated.h"

namespace tt::runtime {

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc();

Tensor createTensor(void *ptr, std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

Device openDevice(std::vector<int> deviceIds = {0});

void closeDevice(Device device);

Event submit(Device device, Binary executable, std::uint32_t programIndex,
             std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs);

void wait(Event event);

} // namespace tt::runtime

#endif
