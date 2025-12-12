// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/detail/python/nanobind_headers.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#pragma clang diagnostic pop

namespace nb = nanobind;
namespace py = pybind11;

namespace tt::runtime::python {

void registerRuntimeUtilsBindings(nb::module_ &m) {
  m.def(
      "create_runtime_tensor_from_ttnn",
      [](nb::object tensor_obj, bool retain) -> ::tt::runtime::Tensor {
        py::handle tensor_pybind_obj(tensor_obj.ptr());
        const ::ttnn::Tensor &tensor =
            py::cast<const ::ttnn::Tensor &>(tensor_pybind_obj);
        return ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
            tensor, std::nullopt, retain);
      },
      "Create a tt::runtime tensor from a TTNN tensor", nb::arg("tensor"),
      nb::arg("retain") = false);

  m.def(
      "create_runtime_device_from_ttnn",
      [](nb::object mesh_device_obj) -> ::tt::runtime::Device {
        py::handle mesh_device_pybind_obj(mesh_device_obj.ptr());
        ::ttnn::MeshDevice *mesh_device =
            py::cast<::ttnn::MeshDevice *>(mesh_device_pybind_obj);
        return ::tt::runtime::ttnn::utils::createRuntimeDeviceFromTTNN(
            mesh_device);
      },
      "Create a tt::runtime device from a TTNN mesh device",
      nb::arg("mesh_device"));

  m.def(
      "get_ttnn_tensor_from_runtime_tensor",
      [](tt::runtime::Tensor tensor) -> nb::object {
        ::ttnn::Tensor &ttnn_tensor =
            ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(tensor);
        py::object py_tensor = py::cast(std::move(ttnn_tensor),
                                        py::return_value_policy::reference);
        return nb::borrow(py_tensor.ptr());
      },
      "Get a TTNN tensor from a tt::runtime tensor");
}
} // namespace tt::runtime::python
