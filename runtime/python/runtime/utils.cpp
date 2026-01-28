// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/detail/python/nanobind_headers.h"

// Include for Metal buffer allocation APIs
#include "tt-metalium/allocator.hpp"

namespace nb = nanobind;

// Namespace aliases for Metal types
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

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

  // ==========================================================================
  // Metal Buffer Allocation APIs
  // ==========================================================================

  // Note: We don't bind tt_metal::BufferType enum here because it conflicts
  // with the ttnn.BufferType enum already exposed by ttnn. The functions below
  // use the buffer type directly (L1 or DRAM) without requiring enum access.

  // Wrapper class to hold the MeshBuffer and expose its properties
  // We need this because MeshBuffer is a shared_ptr internally
  struct MeshBufferWrapper {
    std::shared_ptr<distributed::MeshBuffer> buffer;

    MeshBufferWrapper(std::shared_ptr<distributed::MeshBuffer> buf)
        : buffer(std::move(buf)) {}

    uint32_t address() const { return buffer->address(); }

    size_t size() const { return buffer->size(); }

    void deallocate() { buffer->deallocate(); }

    bool is_allocated() const { return buffer != nullptr; }
  };

  nb::class_<MeshBufferWrapper>(m, "MeshBuffer")
      .def("address", &MeshBufferWrapper::address,
           "Get the starting address of the buffer")
      .def("size", &MeshBufferWrapper::size, "Get the size of the buffer")
      .def("deallocate", &MeshBufferWrapper::deallocate,
           "Deallocate the buffer from device memory")
      .def("is_allocated", &MeshBufferWrapper::is_allocated,
           "Check if the buffer is still allocated");

  // Function to allocate an L1 buffer on the device
  // This is the main API the user will call
  m.def(
      "allocate_l1_buffer",
      [](::ttnn::MeshDevice *mesh_device, size_t buffer_size,
         size_t page_size) -> MeshBufferWrapper {
        // Create device-local buffer config for L1
        tt_metal::BufferShardingArgs bufferShardingArgs(std::nullopt);

        distributed::DeviceLocalBufferConfig local_config{
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::L1,
            .sharding_args = std::move(bufferShardingArgs),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt};

        // Create replicated buffer config
        distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        // Create the MeshBuffer
        auto mesh_buffer = distributed::MeshBuffer::create(
            buffer_config, local_config, mesh_device);

        return MeshBufferWrapper(mesh_buffer);
      },
      nb::arg("mesh_device"), nb::arg("buffer_size"), nb::arg("page_size"),
      R"(
    Allocate an L1 buffer on the device using Metal's allocator.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        The TTNN mesh device to allocate on
    buffer_size : int
        Total size of the buffer in bytes
    page_size : int
        Size of each page in bytes

    Returns
    -------
    MeshBuffer
        A buffer object with address(), size(), and deallocate() methods

    Example
    -------
    >>> device = ttnn.open_device(device_id=0)
    >>> buffer = allocate_l1_buffer(device, buffer_size=1024, page_size=1024)
    >>> print(f"L1 buffer allocated at address: {buffer.address()}")
    >>> buffer.deallocate()
    )");

  // Function to allocate a DRAM buffer on the device
  m.def(
      "allocate_dram_buffer",
      [](::ttnn::MeshDevice *mesh_device, size_t buffer_size,
         size_t page_size) -> MeshBufferWrapper {
        // Create device-local buffer config for DRAM
        tt_metal::BufferShardingArgs bufferShardingArgs(std::nullopt);

        distributed::DeviceLocalBufferConfig local_config{
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::DRAM,
            .sharding_args = std::move(bufferShardingArgs),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt};

        // Create replicated buffer config
        distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

        // Create the MeshBuffer
        auto mesh_buffer = distributed::MeshBuffer::create(
            buffer_config, local_config, mesh_device);

        return MeshBufferWrapper(mesh_buffer);
      },
      nb::arg("mesh_device"), nb::arg("buffer_size"), nb::arg("page_size"),
      R"(
    Allocate a DRAM buffer on the device using Metal's allocator.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        The TTNN mesh device to allocate on
    buffer_size : int
        Total size of the buffer in bytes
    page_size : int
        Size of each page in bytes

    Returns
    -------
    MeshBuffer
        A buffer object with address(), size(), and deallocate() methods
    )");
}
} // namespace tt::runtime::python
