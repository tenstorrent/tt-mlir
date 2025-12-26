// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/detail/python/nanobind_headers.h"

namespace nb = nanobind;

namespace tt::runtime::python {

void registerRuntimeUtilsBindings(nb::module_ &m) {
  m.def(
      "create_runtime_tensor_from_ttnn",
      [](const ::ttnn::Tensor &tensor, bool retain) -> ::tt::runtime::Tensor {
        return ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
            tensor, std::nullopt, retain);
      },
      "Create a tt::runtime tensor from a TTNN tensor", nb::arg("tensor"),
      nb::arg("retain") = false);

  m.def(
      "create_runtime_device_from_ttnn",
      [](::ttnn::MeshDevice *mesh_device) -> ::tt::runtime::Device {
        return ::tt::runtime::ttnn::utils::createRuntimeDeviceFromTTNN(
            mesh_device);
      },
      "Create a tt::runtime device from a TTNN mesh device",
      nb::arg("mesh_device"));

  m.def(
      "get_ttnn_tensor_from_runtime_tensor",
      [](tt::runtime::Tensor tensor) -> ::ttnn::Tensor {
        return ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(
            tensor);
      },
      "Get a TTNN tensor from a tt::runtime tensor");
}
} // namespace tt::runtime::python
