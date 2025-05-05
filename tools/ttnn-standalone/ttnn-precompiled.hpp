// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
#define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

// ANCHOR: standalone_includes
#include "core.hpp"
#include "device.hpp"
#include "operations/ccl/all_gather/all_gather.hpp"
#include "operations/ccl/ccl_host_types.hpp"
#include "operations/ccl/mesh_shard_impl.h"
#include "operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "operations/conv/conv2d/conv2d.hpp"
#include "operations/conv/conv2d/prepare_conv2d_weights.cpp"
#include "operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "operations/copy.hpp"
#include "operations/core/core.hpp"
#include "operations/creation.hpp"
#include "operations/data_movement/concat/concat.hpp"
#include "operations/data_movement/permute/permute.hpp"
#include "operations/data_movement/repeat/repeat.hpp"
#include "operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "operations/data_movement/slice/slice.hpp"
#include "operations/data_movement/transpose/transpose.hpp"
#include "operations/eltwise/binary/binary.hpp"
#include "operations/eltwise/binary/binary_composite.hpp"
#include "operations/eltwise/quantization/quantization.hpp"
#include "operations/eltwise/unary/unary_composite.hpp"
#include "operations/embedding/embedding.hpp"
#include "operations/embedding_backward/embedding_backward.hpp"
#include "operations/matmul/matmul.hpp"
#include "operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "operations/normalization/softmax/softmax.hpp"
#include "operations/pool/generic/generic_pools.hpp"
#include "operations/pool/upsample/upsample.hpp"
#include "operations/reduction/argmax/argmax.hpp"
#include "operations/reduction/generic/generic_reductions.hpp"
#include "operations/reduction/prod/prod.hpp"
#include "tensor/tensor.hpp"
#include "tensor/types.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/small_vector.hpp"
#include "types.hpp"
// ANCHOR_END: standalone_includes

#include <cassert>
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
    static std::shared_ptr<ttnn::MeshDevice> instance =
        ::ttnn::MeshDevice::create_unit_mesh(0);

    return instance.get();
  }

private:
  ~DeviceGetter() { ttnn::close_device(*device); }

public:
  DeviceGetter(const DeviceGetter &) = delete;
  void operator=(const DeviceGetter &) = delete;

  ttnn::IDevice *device;
};

// Wrapper to abstract const-eval logic out of runtime funcs to keep them
// cleaner.  Invokes constEvalFunc iff outputs is empty.
void constEvalFuncWrapper(
    std::function<std::vector<ttnn::Tensor>(std::vector<ttnn::Tensor>)>
        constEvalFunc,
    const std::vector<ttnn::Tensor> &inputs,
    std::vector<ttnn::Tensor> *outputs) {
  if (outputs->empty()) {
    *outputs = constEvalFunc(inputs);
  }
}

} // namespace ttnn

#endif // TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
