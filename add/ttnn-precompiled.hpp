// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
#define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

// ANCHOR: standalone_includes
// #include "ttnn/core.hpp"
// #include "ttnn/device.hpp"
// #include "ttnn/tensor/tensor.hpp"
// #include "ttnn/tensor/types.hpp"
// #include "ttnn/types.hpp"
// #include "ttnn/operations/core/core.hpp"
// #include "ttnn/operations/creation.hpp"
// #include "tt-metalium/bfloat16.hpp"
// #include "workarounds.hpp"
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
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
      assert(!hasOwnedDevice);
      return externalDevice;
    }

    static std::shared_ptr<ttnn::MeshDevice> ownedInstance =
        ::ttnn::MeshDevice::create_unit_mesh(0, l1SmallSize);
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
