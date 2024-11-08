// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/runtime.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "ttrt.runtime python extension for interacting with the "
            "Tenstorrent devices";
  py::class_<tt::runtime::Device>(m, "Device")
      .def("deallocate_buffers", &tt::runtime::detail::deallocateBuffers);
  py::class_<tt::runtime::Event>(m, "Event");
  py::class_<tt::runtime::Tensor>(m, "Tensor");
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
      .value("UInt8", ::tt::target::DataType::UInt8);
  py::enum_<::tt::runtime::DeviceRuntime>(m, "DeviceRuntime")
      .value("Disabled", ::tt::runtime::DeviceRuntime::Disabled)
      .value("TTNN", ::tt::runtime::DeviceRuntime::TTNN)
      .value("TTMetal", ::tt::runtime::DeviceRuntime::TTMetal);

  m.def("get_current_runtime", &tt::runtime::getCurrentRuntime,
        "Get the backend device runtime type");
  m.def("get_available_runtimes", &tt::runtime::getAvailableRuntimes,
        "Get the available backend device runtime types");
  m.def("set_compatible_runtime", &tt::runtime::setCompatibleRuntime,
        py::arg("binary"),
        "Set the backend device runtime type to match the binary");
  m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
        "Get the current system descriptor");
  m.def("register_callback", [](py::function func) {
    tt::runtime::registerCallback([func]() { func(); });
  });
  m.def("unregister_callback", &tt::runtime::unregisterCallback);
  m.def(
      "create_tensor",
      [](std::uintptr_t ptr, std::vector<std::uint32_t> const &shape,
         std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
         ::tt::target::DataType dataType) {
        return tt::runtime::createTensor(
            ::tt::runtime::utils::unsafe_borrow_shared(
                reinterpret_cast<void *>(ptr)),
            shape, stride, itemsize, dataType);
      },
      "Create a tensor with borrowed memory");
  m.def("get_num_available_devices", &tt::runtime::getNumAvailableDevices,
        "Get the number of available devices");
  m.def("open_device", &tt::runtime::openDevice, py::arg("device_ids"),
        py::arg("num_hw_cqs") = size_t{1},
        "Open a mesh of devices for execution");
  m.def("close_device", &tt::runtime::closeDevice, "Close a mesh device");
  m.def("submit", &tt::runtime::submit, py::arg("device"),
        py::arg("executable"), py::arg("program_index"), py::arg("inputs"),
        py::arg("outputs"), "Submit a binary for execution");
  m.def("wait", &tt::runtime::wait, py::arg("event"));

  py::class_<tt::runtime::debug::Env>(m, "DebugEnv")
      .def_static("get", &tt::runtime::debug::Env::get)
      .def("__str__", [](const tt::runtime::debug::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  py::class_<tt::runtime::workaround::Env>(m, "WorkaroundEnv")
      .def_static("get", &tt::runtime::workaround::Env::get)
      .def("__str__", [](const tt::runtime::workaround::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });
}
