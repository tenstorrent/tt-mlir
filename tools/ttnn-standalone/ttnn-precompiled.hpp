// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
#define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

// ANCHOR: standalone_includes
#include "core.hpp"
#include "device.hpp"
#include "operations/copy.hpp"
#include "operations/core/core.hpp"
#include "operations/creation.hpp"
#include "operations/data_movement/concat/concat.hpp"
#include "operations/data_movement/repeat/repeat.hpp"
#include "operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "operations/data_movement/transpose/transpose.hpp"
#include "operations/eltwise/binary/binary.hpp"
#include "operations/eltwise/binary/binary_composite.hpp"
#include "operations/eltwise/unary/unary_composite.hpp"
#include "operations/embedding/embedding.hpp"
#include "operations/embedding_backward/embedding_backward.hpp"
#include "operations/matmul/matmul.hpp"
#include "operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "operations/normalization/softmax/softmax.hpp"
#include "operations/reduction/generic/generic_reductions.hpp"
#include "tensor/tensor.hpp"
#include "tensor/types.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/small_vector.hpp"
#include "types.hpp"
// ANCHOR_END: standalone_includes

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
