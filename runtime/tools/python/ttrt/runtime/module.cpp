// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#include "tt/runtime/detail/debug.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"
#if defined(TTMLIR_ENABLE_RUNTIME_TESTS) && TTMLIR_ENABLE_RUNTIME_TESTS == 1
#include "tt/runtime/detail/ttnn/test/dylib.h"
#include "tt/runtime/detail/ttnn/test/utils.h"
#endif

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "ttrt.runtime python extension for interacting with the "
            "Tenstorrent devices";
  py::class_<tt::runtime::MemoryView>(m, "MemoryView")
      .def_readonly("num_banks", &tt::runtime::MemoryView::numBanks)
      .def_readonly("total_bytes_per_bank",
                    &tt::runtime::MemoryView::totalBytesPerBank)
      .def_readonly("total_bytes_allocated_per_bank",
                    &tt::runtime::MemoryView::totalBytesAllocatedPerBank)
      .def_readonly("total_bytes_free_per_bank",
                    &tt::runtime::MemoryView::totalBytesFreePerBank)
      .def_readonly("largest_contiguous_bytes_free_per_bank",
                    &tt::runtime::MemoryView::largestContiguousBytesFreePerBank)
      .def_readonly("block_table", &tt::runtime::MemoryView::blockTable);
  py::class_<tt::runtime::Device>(m, "Device")
      .def("deallocate_buffers", &tt::runtime::detail::deallocateBuffers)
      .def("dump_memory_report", &tt::runtime::detail::dumpMemoryReport)
      .def("get_memory_view", &tt::runtime::detail::getMemoryView,
           py::arg("device_id") = 0);
  py::class_<tt::runtime::Event>(m, "Event");
  py::class_<tt::runtime::TensorDesc>(m, "TensorDesc")
      .def_readonly("shape", &tt::runtime::TensorDesc::shape)
      .def_readonly("stride", &tt::runtime::TensorDesc::stride)
      .def_readonly("item_size", &tt::runtime::TensorDesc::itemsize)
      .def_readonly("dtype", &tt::runtime::TensorDesc::dataType);
  py::class_<tt::runtime::MeshDeviceOptions>(m, "MeshDeviceOptions")
      .def(py::init<>())
      .def_readwrite("mesh_offset", &tt::runtime::MeshDeviceOptions::meshOffset)
      .def_readwrite("device_ids", &tt::runtime::MeshDeviceOptions::deviceIds)
      .def_readwrite("num_hw_cqs", &tt::runtime::MeshDeviceOptions::numHWCQs)
      .def_readwrite("enable_program_cache",
                     &tt::runtime::MeshDeviceOptions::enableProgramCache)
      .def_property(
          "l1_small_size",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.l1SmallSize.has_value() ? py::cast(o.l1SmallSize.value())
                                             : py::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, py::handle value) {
            o.l1SmallSize = py::none().is(value)
                                ? std::nullopt
                                : std::make_optional(value.cast<size_t>());
          })
      .def_property(
          "dispatch_core_type",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.dispatchCoreType.has_value()
                       ? py::cast(o.dispatchCoreType.value())
                       : py::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, py::handle value) {
            o.dispatchCoreType =
                py::none().is(value)
                    ? std::nullopt
                    : std::make_optional(
                          value.cast<tt::runtime::DispatchCoreType>());
          });
  py::class_<tt::runtime::Tensor>(m, "Tensor")
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
            return py::bytes(reinterpret_cast<const char *>(vec.data()),
                             vec.size());
          },
          py::return_value_policy::take_ownership);
  py::class_<tt::runtime::Layout>(m, "Layout");
  py::class_<tt::runtime::OpContext>(m, "OpContext");
  py::class_<tt::runtime::CallbackContext>(m, "CallbackContext");
  py::enum_<tt::runtime::MemoryBufferType>(m, "MemoryBufferType")
      .value("DRAM", tt::runtime::MemoryBufferType::DRAM)
      .value("L1", tt::runtime::MemoryBufferType::L1)
      .value("L1_SMALL", tt::runtime::MemoryBufferType::L1_SMALL)
      .value("TRACE", tt::runtime::MemoryBufferType::TRACE);
  py::enum_<::tt::target::DataType>(m, "DataType")
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
  py::enum_<::tt::runtime::DeviceRuntime>(m, "DeviceRuntime")
      .value("Disabled", ::tt::runtime::DeviceRuntime::Disabled)
      .value("TTNN", ::tt::runtime::DeviceRuntime::TTNN)
      .value("TTMetal", ::tt::runtime::DeviceRuntime::TTMetal);
  py::enum_<::tt::runtime::DispatchCoreType>(m, "DispatchCoreType")
      .value("WORKER", ::tt::runtime::DispatchCoreType::WORKER)
      .value("ETH", ::tt::runtime::DispatchCoreType::ETH);
  m.def("get_current_runtime", &tt::runtime::getCurrentRuntime,
        "Get the backend device runtime type");
  m.def("get_available_runtimes", &tt::runtime::getAvailableRuntimes,
        "Get the available backend device runtime types");
  m.def("set_compatible_runtime", &tt::runtime::setCompatibleRuntime,
        py::arg("binary"),
        "Set the backend device runtime type to match the binary");
  m.def("set_current_runtime", &tt::runtime::setCurrentRuntime,
        py::arg("runtime"), "Set the backend device runtime type");
  m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
        py::arg("dispatch_core_type") = py::none(),
        py::arg("mesh_device") = py::none(),
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
  m.def("get_num_available_devices", &tt::runtime::getNumAvailableDevices,
        "Get the number of available devices");
  m.def("open_mesh_device", &tt::runtime::openMeshDevice, py::arg("mesh_shape"),
        py::arg("options"), "Open a parent mesh of devices");
  m.def("close_mesh_device", &tt::runtime::closeMeshDevice,
        py::arg("parent_mesh"), "Close a mesh device");
  m.def("create_sub_mesh_device", &tt::runtime::createSubMeshDevice,
        py::arg("parent_mesh"), py::arg("mesh_shape"),
        py::arg("mesh_offset") = py::none(),
        "Open a sub mesh of devices from a parent mesh");
  m.def("release_sub_mesh_device", &tt::runtime::releaseSubMeshDevice,
        py::arg("sub_mesh"), "Release a sub mesh device from the parent");
  m.def("reshape_mesh_device", &tt::runtime::reshapeMeshDevice,
        py::arg("mesh_device"), py::arg("mesh_shape"), "Reshape a mesh device");
  m.def("to_host", &tt::runtime::toHost, py::arg("tensor"),
        py::arg("untilize") = false, "Copy the tensor to the host");
  m.def("to_layout", &tt::runtime::toLayout, py::arg("tensor"),
        py::arg("device"), py::arg("layout"), py::arg("retain") = py::none(),
        "Create a copy of the tensor with the specified layout");
  m.def("get_layout", &tt::runtime::getLayout, py::arg("executable"),
        py::arg("program_index"), py::arg("input_index"),
        "Get the layout of the input tensor");
  m.def(
      "submit",
      [](::tt::runtime::Device device, ::tt::runtime::Binary &executable,
         std::uint32_t programIndex, std::vector<::tt::runtime::Tensor> &inputs)
          -> std::vector<::tt::runtime::Tensor> {
        return ::tt::runtime::submit(device, executable, programIndex, inputs);
      },
      py::arg("device"), py::arg("executable"), py::arg("program_index"),
      py::arg("inputs"),
      "Submit a ttnn binary for execution, returns a vector of output tensors."
      "The input tensors will be moved and consumed.");
  m.def(
      "submit",
      [](::tt::runtime::Device device, ::tt::runtime::Binary executable,
         std::uint32_t programIndex,
         const std::vector<::tt::runtime::Tensor> &inputs,
         const std::vector<::tt::runtime::Tensor> &outputs)
          -> ::tt::runtime::Event {
        return ::tt::runtime::submit(device, executable, programIndex, inputs,
                                     outputs);
      },
      py::arg("device"), py::arg("executable"), py::arg("program_index"),
      py::arg("inputs"), py::arg("outputs"),
      "Submit a ttmetal binary for execution. returns event wrapper");
  m.def(
      "wait", [](::tt::runtime::Event event) { ::tt::runtime::wait(event); },
      py::arg("event"));
  m.def(
      "wait", [](::tt::runtime::Tensor tensor) { ::tt::runtime::wait(tensor); },
      py::arg("tensor"));
  m.def(
      "wait",
      [](const std::vector<::tt::runtime::Tensor> &tensors) {
        ::tt::runtime::wait(tensors);
      },
      py::arg("tensors"));
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
  m.def(
      "memcpy",
      [](std::uintptr_t dst, ::tt::runtime::Tensor src) {
        void *dstPtr = reinterpret_cast<void *>(dst);
        ::tt::runtime::memcpy(dstPtr, src);
      },
      py::arg("dst"), py::arg("src"),
      "Copy the data from src tensor to dst pointer");
  m.def(
      "memcpy",
      [](::tt::runtime::Tensor dst, ::tt::runtime::Tensor src) {
        ::tt::runtime::memcpy(dst, src);
      },
      py::arg("dst"), py::arg("src"),
      "Copy the data from src tensor to dst tensor");
  m.def("deallocate_tensor", &tt::runtime::deallocateTensor, py::arg("tensor"),
        py::arg("force") = false, "Deallocate the tensor memory");
  py::class_<tt::runtime::debug::Env>(m, "DebugEnv")
      .def_static("get", &tt::runtime::debug::Env::get)
      .def("__str__", [](const tt::runtime::debug::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  py::class_<tt::runtime::debug::PerfEnv>(m, "DebugPerfEnv")
      .def_static("get", &tt::runtime::debug::PerfEnv::get)
      .def("__str__", [](const tt::runtime::debug::PerfEnv &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  py::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
      .def_static(
          "get",
          [](py::function pre_op_func, py::function post_op_func) {
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

  py::class_<tt::runtime::workaround::Env>(m, "WorkaroundEnv")
      .def_static("get", &tt::runtime::workaround::Env::get)
      .def("__str__", [](const tt::runtime::workaround::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

#if defined(TTMLIR_ENABLE_RUNTIME_TESTS) && TTMLIR_ENABLE_RUNTIME_TESTS == 1
  auto testing = m.def_submodule("testing");
  testing.def("get_dram_interleaved_tile_layout",
              &tt::runtime::ttnn::test::getDramInterleavedTileLayout,
              py::arg("dtype"), "Get dram interleaved tile layout");
  testing.def("get_dram_interleaved_row_major_layout",
              &tt::runtime::ttnn::test::getDramInterleavedRowMajorLayout,
              py::arg("dtype"), "Get dram interleaved row major layout");
  testing.def("get_host_row_major_layout",
              &tt::runtime::ttnn::test::getHostRowMajorLayout, py::arg("dtype"),
              "Get host row major layout");
  testing.def("is_program_cache_enabled",
              &tt::runtime::ttnn::test::isProgramCacheEnabled,
              py::arg("device"), "Check if program cache is enabled");
  testing.def("open_so", &tt::runtime::ttnn::test::openSo, py::arg("path"),
              "Open a shared object file");
  testing.def("run_so_program", &tt::runtime::ttnn::test::runSoProgram,
              py::arg("so"), py::arg("func_name"), py::arg("inputs"),
              py::arg("device"), "Run a program from a shared object file");
  testing.def("compare_outs", &tt::runtime::ttnn::test::compareOuts,
              py::arg("lhs"), py::arg("rhs"));
#endif

  /**
   * Cleanup code to force a well ordered destruction w.r.t. the GIL
   */
  auto cleanup_callback = []() {
    ::tt::runtime::debug::Hooks::get().unregisterHooks();
  };
  m.add_object("_cleanup", py::capsule(cleanup_callback));
  m.def("unregister_hooks",
        []() { ::tt::runtime::debug::Hooks::get().unregisterHooks(); });
}
