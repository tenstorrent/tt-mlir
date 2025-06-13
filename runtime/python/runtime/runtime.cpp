// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#include "tt/runtime/detail/debug.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace nb = nanobind;

namespace tt::runtime::python {
void registerRuntimeBindings(nb::module_ &m) {
  nb::class_<tt::runtime::MemoryView>(m, "MemoryView")
      .def_ro("num_banks", &tt::runtime::MemoryView::numBanks)
      .def_ro("total_bytes_per_bank",
              &tt::runtime::MemoryView::totalBytesPerBank)
      .def_ro("total_bytes_allocated_per_bank",
              &tt::runtime::MemoryView::totalBytesAllocatedPerBank)
      .def_ro("total_bytes_free_per_bank",
              &tt::runtime::MemoryView::totalBytesFreePerBank)
      .def_ro("largest_contiguous_bytes_free_per_bank",
              &tt::runtime::MemoryView::largestContiguousBytesFreePerBank)
      .def_ro("block_table", &tt::runtime::MemoryView::blockTable);

  nb::class_<tt::runtime::Device>(m, "Device")
      .def("get_mesh_shape", &tt::runtime::getMeshShape)
      .def("get_device_ids", &tt::runtime::getDeviceIds)
      .def("get_num_hw_cqs", &tt::runtime::getNumHwCqs)
      .def("is_program_cache_enabled", &tt::runtime::isProgramCacheEnabled)
      .def("get_l1_small_size", &tt::runtime::getL1SmallSize)
      .def("get_trace_region_size", &tt::runtime::getTraceRegionSize)
      .def("get_num_dram_channels", &tt::runtime::getNumDramChannels)
      .def("get_dram_size_per_channel", &tt::runtime::getDramSizePerChannel)
      .def("get_l1_size_per_core", &tt::runtime::getL1SizePerCore)
      .def("release_trace", &tt::runtime::releaseTrace)
      .def("deallocate_buffers", &tt::runtime::detail::deallocateBuffers)
      .def("dump_memory_report", &tt::runtime::detail::dumpMemoryReport)
      .def("dump_device_profile_results",
           &tt::runtime::detail::dumpDeviceProfileResults)
      .def("get_memory_view", &tt::runtime::detail::getMemoryView);

  nb::class_<tt::runtime::Event>(m, "Event");

  nb::class_<tt::runtime::TensorDesc>(m, "TensorDesc")
      .def_ro("shape", &tt::runtime::TensorDesc::shape)
      .def_ro("stride", &tt::runtime::TensorDesc::stride)
      .def_ro("item_size", &tt::runtime::TensorDesc::itemsize)
      .def_ro("dtype", &tt::runtime::TensorDesc::dataType);

  nb::class_<tt::runtime::MeshDeviceOptions>(m, "MeshDeviceOptions")
      .def(nb::init<>())
      .def_rw("mesh_offset", &tt::runtime::MeshDeviceOptions::meshOffset)
      .def_rw("device_ids", &tt::runtime::MeshDeviceOptions::deviceIds)
      .def_rw("num_hw_cqs", &tt::runtime::MeshDeviceOptions::numHWCQs)
      .def_rw("enable_program_cache",
              &tt::runtime::MeshDeviceOptions::enableProgramCache)
      .def_prop_rw(
          "l1_small_size",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.l1SmallSize.has_value() ? nb::cast(o.l1SmallSize.value())
                                             : nb::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, nb::handle value) {
            o.l1SmallSize = value.is_none()
                                ? std::nullopt
                                : std::make_optional(nb::cast<size_t>(value));
          })
      .def_prop_rw(
          "trace_region_size",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.traceRegionSize.has_value()
                       ? nb::cast(o.traceRegionSize.value())
                       : nb::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, nb::handle value) {
            o.traceRegionSize =
                value.is_none() ? std::nullopt
                                : std::make_optional(nb::cast<size_t>(value));
          })
      .def_prop_rw(
          "dispatch_core_type",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.dispatchCoreType.has_value()
                       ? nb::cast(o.dispatchCoreType.value())
                       : nb::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, nb::handle value) {
            o.dispatchCoreType =
                value.is_none()
                    ? std::nullopt
                    : std::make_optional(
                          nb::cast<tt::runtime::DispatchCoreType>(value));
          });

  nb::class_<tt::runtime::Tensor>(m, "Tensor")
      .def("is_allocated",
           [](tt::runtime::Tensor self) {
             return tt::runtime::isTensorAllocated(self);
           })
      .def("get_retain",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorRetain(self);
           })
      .def("set_retain",
           [](tt::runtime::Tensor self, bool retain) {
             tt::runtime::setTensorRetain(self, retain);
           })
      .def("get_shape",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorShape(self);
           })
      .def("get_stride",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorStride(self);
           })
      .def("get_volume",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorVolume(self);
           })
      .def("get_dtype",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorDataType(self);
           })
      .def("get_element_size",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorElementSize(self);
           })
      .def("get_tensor_desc",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorDesc(self);
           })
      .def(
          "get_data_buffer",
          [](tt::runtime::Tensor self) {
            std::vector<std::byte> vec = tt::runtime::getTensorDataBuffer(self);
            return nb::bytes(reinterpret_cast<const char *>(vec.data()),
                             vec.size());
          },
          nb::rv_policy::take_ownership);

  nb::class_<tt::runtime::Layout>(m, "Layout");
  nb::class_<tt::runtime::OpContext>(m, "OpContext");
  nb::class_<tt::runtime::CallbackContext>(m, "CallbackContext");

  nb::enum_<tt::runtime::MemoryBufferType>(m, "MemoryBufferType")
      .value("DRAM", tt::runtime::MemoryBufferType::DRAM)
      .value("L1", tt::runtime::MemoryBufferType::L1)
      .value("L1_SMALL", tt::runtime::MemoryBufferType::L1_SMALL)
      .value("TRACE", tt::runtime::MemoryBufferType::TRACE);

  nb::enum_<::tt::target::DataType>(m, "DataType")
      .value("Float32", ::tt::target::DataType::Float32)
      .value("Float16", ::tt::target::DataType::Float16)
      .value("BFloat16", ::tt::target::DataType::BFloat16)
      .value("BFP_Float8", ::tt::target::DataType::BFP_Float8)
      .value("BFP_BFloat8", ::tt::target::DataType::BFP_BFloat8)
      .value("BFP_Float4", ::tt::target::DataType::BFP_Float4)
      .value("BFP_BFloat4", ::tt::target::DataType::BFP_BFloat4)
      .value("BFP_Float2", ::tt::target::DataType::BFP_Float2)
      .value("BFP_BFloat2", ::tt::target::DataType::BFP_BFloat2)
      .value("UInt32", ::tt::target::DataType::UInt32)
      .value("UInt16", ::tt::target::DataType::UInt16)
      .value("UInt8", ::tt::target::DataType::UInt8)
      .value("Int32", ::tt::target::DataType::Int32);

  nb::enum_<::tt::runtime::DeviceRuntime>(m, "DeviceRuntime")
      .value("Disabled", ::tt::runtime::DeviceRuntime::Disabled)
      .value("TTNN", ::tt::runtime::DeviceRuntime::TTNN)
      .value("TTMetal", ::tt::runtime::DeviceRuntime::TTMetal);

  nb::enum_<::tt::runtime::DispatchCoreType>(m, "DispatchCoreType")
      .value("WORKER", ::tt::runtime::DispatchCoreType::WORKER)
      .value("ETH", ::tt::runtime::DispatchCoreType::ETH);

  nb::enum_<::tt::runtime::Arch>(m, "Arch")
      .value("GRAYSKULL", ::tt::runtime::Arch::GRAYSKULL)
      .value("WORMHOLE_B0", ::tt::runtime::Arch::WORMHOLE_B0)
      .value("BLACKHOLE", ::tt::runtime::Arch::BLACKHOLE)
      .value("QUASAR", ::tt::runtime::Arch::QUASAR);

  m.def("get_current_runtime", &tt::runtime::getCurrentRuntime,
        "Get the backend device runtime type");
  m.def("get_available_runtimes", &tt::runtime::getAvailableRuntimes,
        "Get the available backend device runtime types");
  m.def("set_compatible_runtime", &tt::runtime::setCompatibleRuntime,
        nb::arg("binary"),
        "Set the backend device runtime type to match the binary");
  m.def("set_current_runtime", &tt::runtime::setCurrentRuntime,
        nb::arg("runtime"), "Set the backend device runtime type");
  m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
        nb::arg("dispatch_core_type") = nb::none(),
        nb::arg("mesh_device") = nb::none(),
        "Get the current system descriptor");
  m.def(
      "create_tensor",
      [](std::uintptr_t ptr, const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
         ::tt::target::DataType dataType) {
        return tt::runtime::createTensor(
            ::tt::runtime::utils::unsafe_borrow_shared(
                reinterpret_cast<void *>(ptr)),
            shape, stride, itemsize, dataType);
      },
      "Create a host tensor with borrowed memory");
  m.def(
      "create_owned_tensor",
      [](std::uintptr_t ptr, const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
         ::tt::target::DataType dataType) {
        return tt::runtime::createOwnedHostTensor(
            reinterpret_cast<const void *>(ptr), shape, stride, itemsize,
            dataType);
      },
      "Create a tensor with owned memory");
  m.def(
      "create_empty_tensor",
      [](::tt::runtime::Device device, ::tt::runtime::Layout layout,
         const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize) {
        return tt::runtime::createEmptyTensor(device, layout, shape, stride,
                                              itemsize);
      },
      "Create an empty tensor with the specified layout");
  m.def(
      "create_multi_device_tensor",
      [](std::vector<std::uintptr_t> &ptrs,
         const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
         ::tt::target::DataType dataType,
         const std::unordered_map<std::string, std::string> &strategy) {
        std::vector<const void *> data;
        data.reserve(ptrs.size());
        std::transform(ptrs.begin(), ptrs.end(), std::back_inserter(data),
                       [](std::uintptr_t ptr) {
                         return reinterpret_cast<const void *>(ptr);
                       });
        return tt::runtime::createMultiDeviceHostTensor(
            data, shape, stride, itemsize, dataType, strategy);
      },
      "Create a multi-device host tensor with owned memory");
  m.def("get_arch", &tt::runtime::getArch,
        "Get the architecture of the device");
  m.def("get_num_available_devices", &tt::runtime::getNumAvailableDevices,
        "Get the number of available devices");
  m.def("open_mesh_device", &tt::runtime::openMeshDevice, nb::arg("mesh_shape"),
        nb::arg("options"), "Open a parent mesh of devices");
  m.def("close_mesh_device", &tt::runtime::closeMeshDevice,
        nb::arg("parent_mesh"), "Close a mesh device");
  m.def("create_sub_mesh_device", &tt::runtime::createSubMeshDevice,
        nb::arg("parent_mesh"), nb::arg("mesh_shape"),
        nb::arg("mesh_offset") = nb::none(),
        "Open a sub mesh of devices from a parent mesh");
  m.def("release_sub_mesh_device", &tt::runtime::releaseSubMeshDevice,
        nb::arg("sub_mesh"), "Release a sub mesh device from the parent");
  m.def("reshape_mesh_device", &tt::runtime::reshapeMeshDevice,
        nb::arg("mesh_device"), nb::arg("mesh_shape"), "Reshape a mesh device");
  m.def("to_host", &tt::runtime::toHost, nb::arg("tensor"),
        nb::arg("untilize") = false, "Copy the tensor to the host");
  m.def("to_layout", &tt::runtime::toLayout, nb::arg("tensor"),
        nb::arg("device"), nb::arg("layout"), nb::arg("retain") = nb::none(),
        "Create a copy of the tensor with the specified layout");
  m.def("get_layout", &tt::runtime::getLayout, nb::arg("executable"),
        nb::arg("program_index"), nb::arg("input_index"),
        "Get the layout of the input tensor");
  m.def(
      "submit",
      [](::tt::runtime::Device device, ::tt::runtime::Binary &executable,
         std::uint32_t programIndex, std::vector<::tt::runtime::Tensor> &inputs)
          -> std::vector<::tt::runtime::Tensor> {
        return ::tt::runtime::submit(device, executable, programIndex, inputs);
      },
      nb::arg("device"), nb::arg("executable"), nb::arg("program_index"),
      nb::arg("inputs"),
      "Submit a ttnn binary for execution, returns a vector of output tensors."
      "The input tensors will be moved and consumed.");
  m.def(
      "wait", [](::tt::runtime::Event event) { ::tt::runtime::wait(event); },
      nb::arg("event"));
  m.def(
      "wait", [](::tt::runtime::Tensor tensor) { ::tt::runtime::wait(tensor); },
      nb::arg("tensor"));
  m.def(
      "wait",
      [](const std::vector<::tt::runtime::Tensor> &tensors) {
        ::tt::runtime::wait(tensors);
      },
      nb::arg("tensors"));
  m.def(
      "get_op_output_tensor",
      [](tt::runtime::OpContext &opContextHandle,
         tt::runtime::CallbackContext &programContextHandle) {
        tt::runtime::Tensor tensor = tt::runtime::getOpOutputTensor(
            opContextHandle, programContextHandle);
        return tensor.handle.get() == nullptr
                   ? std::nullopt
                   : std::optional<tt::runtime::Tensor>(tensor);
      },
      "Get the output tensor of the op");
  m.def("get_op_debug_str", &tt::runtime::getOpDebugString,
        "Get the debug string of the op");
  m.def("get_op_loc_info", &tt::runtime::getOpLocInfo,
        "Get the location info of the op");
  m.def("get_debug_info_golden", &::tt::runtime::Binary::getDebugInfoGolden,
        nb::rv_policy::reference, "Get the debug info golden tensor");
  m.def(
      "memcpy",
      [](std::uintptr_t dst, ::tt::runtime::Tensor src) {
        void *dstPtr = reinterpret_cast<void *>(dst);
        ::tt::runtime::memcpy(dstPtr, src);
      },
      nb::arg("dst"), nb::arg("src"),
      "Copy the data from src tensor to dst pointer");
  m.def(
      "memcpy",
      [](::tt::runtime::Tensor dst, ::tt::runtime::Tensor src) {
        ::tt::runtime::memcpy(dst, src);
      },
      nb::arg("dst"), nb::arg("src"),
      "Copy the data from src tensor to dst tensor");
  m.def("deallocate_tensor", &tt::runtime::deallocateTensor, nb::arg("tensor"),
        nb::arg("force") = false, "Deallocate the tensor memory");

  nb::class_<tt::runtime::debug::Env>(m, "DebugEnv")
      .def_static("get", &tt::runtime::debug::Env::get)
      .def("__str__", [](const tt::runtime::debug::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  nb::class_<tt::runtime::debug::PerfEnv>(m, "DebugPerfEnv")
      .def_static("get", &tt::runtime::debug::PerfEnv::get)
      .def("__str__", [](const tt::runtime::debug::PerfEnv &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  nb::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
      .def_static(
          "get",
          [](nb::callable pre_op_func, nb::callable post_op_func) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
            tt::runtime::debug::Hooks::get(
                [pre_op_func](tt::runtime::Binary Binary,
                              tt::runtime::CallbackContext programContext,
                              tt::runtime::OpContext opContext) {
                  pre_op_func(Binary, programContext, opContext);
                },
                [post_op_func](tt::runtime::Binary Binary,
                               tt::runtime::CallbackContext programContext,
                               tt::runtime::OpContext opContext) {
                  post_op_func(Binary, programContext, opContext);
                });
#else
            tt::runtime::debug::Hooks::get();
#endif
          })
      .def("__str__", [](const tt::runtime::debug::Hooks &hooks) {
        std::stringstream os;
        os << hooks;
        return os.str();
      });

  nb::class_<tt::runtime::workaround::Env>(m, "WorkaroundEnv")
      .def_static("get", &tt::runtime::workaround::Env::get)
      .def("__str__", [](const tt::runtime::workaround::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  m.def("unregister_hooks",
        []() { ::tt::runtime::debug::Hooks::get().unregisterHooks(); });
}
} // namespace tt::runtime::python
