// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
#define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

// ANCHOR: standalone_includes
#include "tt-metalium/bfloat16.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/quantization/quantization.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/rand/rand.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp"
#include "ttnn/tensor/serialization.hpp"
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

template <typename... T>
std::vector<ttnn::Tensor> util_create_vec(T &&...t) {
  return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}

namespace ttnn {

// DeviceGetter class
//
// Singleton implementation for Device
//
class DeviceGetter {
public:
  static constexpr std::size_t l1SmallSize = 1 << 15;     // 32kB
  static constexpr std::size_t traceRegionSize = 1 << 20; // 1MB

  static ttnn::MeshDevice *getInstance() {
    // If we have an external device, use it.
    if (externalDevice) {
      assert(!hasOwnedDevice);
      return externalDevice;
    }

    static std::shared_ptr<ttnn::MeshDevice> ownedInstance =
        ::ttnn::MeshDevice::create_unit_mesh(0, l1SmallSize, traceRegionSize);
    hasOwnedDevice = true;
    return ownedInstance.get();
  }

  // Set an external device (we don't own it)
  static void setInstance(ttnn::MeshDevice *newInstance) {
    // We don't want to mix and match owned/external devices.
    assert(!hasOwnedDevice);

    // Store the external device pointer.
    externalDevice = newInstance;
  }

private:
  DeviceGetter() = default;

  DeviceGetter(const DeviceGetter &) = delete;
  DeviceGetter &operator=(const DeviceGetter &) = delete;

  // External device (not owned by us).
  static ttnn::MeshDevice *externalDevice;
  // Flag to track if we've set local ownedInstance or not.
  static bool hasOwnedDevice;
};

inline ttnn::MeshDevice *DeviceGetter::externalDevice = nullptr;
inline bool DeviceGetter::hasOwnedDevice = false;

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
