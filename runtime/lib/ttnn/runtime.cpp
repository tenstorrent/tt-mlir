// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/utils.h"

namespace tt::runtime {
Device wrap(ttnn::Device &device) {
  return Device{utils::unsafe_borrow_shared(&device)};
}

ttnn::Device& unwrap(Device device) {
  return *static_cast<ttnn::Device *>(device.handle.get());
}

SystemDesc getCurrentSystemDesc() { return SystemDesc{}; }

Device openDevice(SystemDesc const &sysDesc,
                  std::vector<std::uint32_t> /*deviceIds*/) {
  auto &device = ttnn::open_device(0);
  return wrap(device);
}

void closeDevice(Device device) {
  auto &ttnn_device = unwrap(device);
  ttnn::close_device(ttnn_device);
}
}
