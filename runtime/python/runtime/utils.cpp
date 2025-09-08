// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include "tt/runtime/detail/ttnn/utils.h"

#include "tt/runtime/detail/python/nanobind_headers.h"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace nb = nanobind;
namespace py = pybind11;

namespace tt::runtime::python {

void registerRuntimeUtilsBindings(nb::module_ &m) {
  m.def(
      "create_runtime_tensor_from_ttnn",
      [](nb::object tensor_obj, bool retain) {
        py::handle tensor_pybind_obj(tensor_obj.ptr());
        const ::ttnn::Tensor &tensor =
            py::cast<const ::ttnn::Tensor &>(tensor_pybind_obj);
        std::cout << "tensor: " << tensor.is_allocated() << " storage_type: "
                  << static_cast<int>(tensor.storage_type()) << std::endl;
        return ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
            tensor, std::nullopt, retain);
      },
      "Create a tt::runtime tensor from a TTNN tensor", nb::arg("tensor"),
      nb::arg("retain") = false);
  m.def("get_ttnn_tensor_from_runtime_tensor",
        &tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor,
        "Get a TTNN tensor from a tt::runtime tensor");
}
} // namespace tt::runtime::python
