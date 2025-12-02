// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/detail/python/nanobind_headers.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/utils.h"

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

  // TTMetal-compatible wrappers for TTNN objects
  // These allow executing TTMetal binaries with TTNN device/tensors
  m.def(
      "create_ttmetal_device_from_ttnn",
      [](nb::object mesh_device_obj) -> ::tt::runtime::Device {
        py::handle mesh_device_pybind_obj(mesh_device_obj.ptr());
        ::ttnn::MeshDevice *mesh_device =
            py::cast<::ttnn::MeshDevice *>(mesh_device_pybind_obj);

        // Create a non-owning shared_ptr to the MeshDevice
        std::shared_ptr<void> deviceSharedPtr =
            ::tt::runtime::utils::unsafeBorrowShared(mesh_device);

        // Create Device with TTMetal runtime tag (no trace cache for now)
        return ::tt::runtime::Device(deviceSharedPtr, nullptr,
                                     ::tt::runtime::DeviceRuntime::TTMetal);
      },
      "Create a TTMetal-compatible runtime device from a TTNN mesh device",
      nb::arg("mesh_device"));

  m.def(
      "create_ttmetal_tensor_from_ttnn",
      [](nb::object tensor_obj, bool retain) -> ::tt::runtime::Tensor {
        py::handle tensor_pybind_obj(tensor_obj.ptr());
        const ::ttnn::Tensor &tensor =
            py::cast<const ::ttnn::Tensor &>(tensor_pybind_obj);

        // Get the underlying buffer and device
        ::tt::tt_metal::Buffer *buffer = tensor.buffer();
        LOG_ASSERT(buffer != nullptr,
                   "TTNN tensor must have a device buffer for TTMetal wrapping");

        ::ttnn::MeshDevice *meshDevice = tensor.device();
        LOG_ASSERT(meshDevice != nullptr,
                   "TTNN tensor must be on a device for TTMetal wrapping");

        // Get buffer properties
        uint64_t address = buffer->address();
        uint64_t size = buffer->size();
        uint32_t pageSize = buffer->page_size();

        // Determine buffer type from memory config
        ::tt::tt_metal::BufferType bufferType = buffer->buffer_type();

        // Create BufferShardingArgs - using std::nullopt defaults to interleaved layout
        ::tt::tt_metal::BufferShardingArgs shardingArgs(std::nullopt);

        // Create DeviceLocalBufferConfig
        ::tt::tt_metal::distributed::DeviceLocalBufferConfig localConfig{
            .page_size = pageSize,
            .buffer_type = bufferType,
            .sharding_args = std::move(shardingArgs),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };

        // Create MeshBufferConfig (replicated for single device view)
        ::tt::tt_metal::distributed::ReplicatedBufferConfig meshConfig{
            .size = size,
        };

        // Create MeshBuffer as a view over the existing address
        auto meshBuffer = ::tt::tt_metal::distributed::MeshBuffer::create(
            meshConfig, localConfig, meshDevice, address);

        // Wrap as MetalTensor and create runtime Tensor
        auto metalTensor =
            std::make_shared<::tt::runtime::ttmetal::MetalTensor>(
                std::move(meshBuffer));

        return ::tt::runtime::Tensor(
            std::static_pointer_cast<void>(metalTensor),
            /*data=*/nullptr, ::tt::runtime::DeviceRuntime::TTMetal);
      },
      "Create a TTMetal-compatible runtime tensor from a TTNN device tensor",
      nb::arg("tensor"), nb::arg("retain") = false);
}
} // namespace tt::runtime::python
