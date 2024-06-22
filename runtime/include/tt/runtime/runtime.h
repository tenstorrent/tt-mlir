// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_RUNTIME_H
#define TT_RUNTIME_RUNTIME_H

#include <cstdint>
#include <functional>
#include <vector>

#include "tt/runtime/types.h"

namespace tt::runtime {

SystemDesc getCurrentSystemDesc();

Device openDevice(std::vector<int> deviceIds = {0});

void closeDevice(Device device);

Event submit(Device device, Binary executable,
             std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs,
             std::function<void()> completedCallback = nullptr);

void wait(Event event);

} // namespace tt::runtime

#endif
