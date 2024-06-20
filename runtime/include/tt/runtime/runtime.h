// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_RUNTIME_H
#define TT_RUNTIME_RUNTIME_H

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace tt::runtime {

struct SystemDesc {
  std::vector<std::uint8_t> fbb;
  void saveToFile(const char *filepath) const;
};

struct Executable {
  std::shared_ptr<void> handle;
  void saveToFile(const char *filepath) const;
};

struct Device {
  void *handle;
};

struct Event {
  void *handle;
};

struct Tensor {
  void *handle;
};

SystemDesc getCurrentSystemDesc();

Device openDevice(SystemDesc const &sysDesc,
                  std::vector<std::uint32_t> deviceIds = {0});

Event submit(Device device, Executable executable,
             std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs,
             std::function<void()> completedCallback = nullptr);

void wait(Event event);

void closeDevice(Device device);

} // namespace tt::runtime

#endif
