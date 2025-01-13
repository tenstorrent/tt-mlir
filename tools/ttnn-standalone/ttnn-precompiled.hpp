// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
#define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

#include "common/bfloat16.hpp"
#include "core.hpp"
#include "device.hpp"
#include "operations/core/core.hpp"
#include "operations/creation.hpp"
#include "operations/eltwise/binary/binary.hpp"
#include "operations/embedding/embedding.hpp"
#include "operations/embedding_backward/embedding_backward.hpp"
#include "operations/matmul/matmul.hpp"
#include "tensor/tensor.hpp"
#include "tensor/types.hpp"
#include "types.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

namespace ttnn {

// DeviceGetter class
//
// Singleton implementation for Device
//
class DeviceGetter {
public:
  static ttnn::IDevice *getInstance() {
    static ttnn::IDevice *instance = &ttnn::open_device(0);

    return instance;
  }

private:
  ~DeviceGetter() { ttnn::close_device(*device); }

public:
  DeviceGetter(DeviceGetter const &) = delete;
  void operator=(DeviceGetter const &) = delete;

  ttnn::IDevice *device;
};

} // namespace ttnn

#endif // TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
