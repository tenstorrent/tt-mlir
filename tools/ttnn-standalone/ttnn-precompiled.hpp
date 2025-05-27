// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
#define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

// ANCHOR: standalone_includes
#include "operations/ccl/all_gather/all_gather.hpp"
#include "operations/ccl/ccl_host_types.hpp"
#include "operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "operations/conv/conv2d/conv2d.hpp"
#include "operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "operations/conv/conv_transpose2d/conv_transpose2d.hpp"
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
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/small_vector.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "workarounds.hpp"
// ANCHOR_END: standalone_includes

#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace ttnn {

// DeviceGetter class
//
// Singleton implementation for Device
//
class DeviceGetter {
public:
  static constexpr std::size_t l1SmallSize = 1 << 15;

  static ttnn::MeshDevice *getInstance() {
    // If we have an external device, use it.
    if (externalDevice) {
      assert(ownedDevice == nullptr);
      return externalDevice;
    }

    // Otherwise, create and use our own device.
    if (!ownedDevice) {
      ownedDevice = ::ttnn::MeshDevice::create_unit_mesh(0, l1SmallSize);
    }
    return ownedDevice.get();
  }

  // Set an external device (we don't own it)
  static void setInstance(ttnn::MeshDevice *newInstance) {
    // We don't want to mix and match owned/external devices.
    assert(ownedDevice == nullptr);

    // Store the external device pointer.
    externalDevice = newInstance;
  }

private:
  DeviceGetter() = default;

  DeviceGetter(const DeviceGetter &) = delete;
  DeviceGetter &operator=(const DeviceGetter &) = delete;

  // External device (not owned by us).
  static ttnn::MeshDevice *externalDevice;

  // Our owned device (only used if no external device is set).
  static std::shared_ptr<ttnn::MeshDevice> ownedDevice;
};

inline ttnn::MeshDevice *DeviceGetter::externalDevice = nullptr;
inline std::shared_ptr<ttnn::MeshDevice> DeviceGetter::ownedDevice;

// Function to be exported from the dylib that can be called to set the
// device--extern to avoid mangling.
extern "C" {
void setDevice(ttnn::MeshDevice *device) { DeviceGetter::setInstance(device); }
}

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
