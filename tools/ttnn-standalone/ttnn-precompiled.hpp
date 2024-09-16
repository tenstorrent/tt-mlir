// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/bfloat16.hpp"
#include "core.hpp"
#include "device.hpp"
#include "operations/core/core.hpp"
#include "operations/creation.hpp"
#include "operations/eltwise/binary/binary.hpp"
#include "tensor/tensor.hpp"
#include "tensor/types.hpp"
#include "types.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

namespace ttnn {

class DeviceGetter {
public:
  static ttnn::Device *getInstance() {
    // ttnn::Device& device = open_device(0);
    static ttnn::Device *instance;

    return instance;
  }

private:
  ~DeviceGetter() { close_device(*device); }

public:
  DeviceGetter(DeviceGetter const &) = delete;
  void operator=(DeviceGetter const &) = delete;

  Device *device;
};

} // namespace ttnn
