// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "ttrt.runtime python extension for interacting with the "
            "Tenstorrent devies";

  py::class_<tt::runtime::Device>(m, "Device");
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

  m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
        "Get the current system descriptor");
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
  m.def("open_device", &tt::runtime::openDevice,
        py::arg("device_ids") = std::vector<int>{0},
        "Open a device for execution");
  m.def("close_device", &tt::runtime::closeDevice, "Close a device");
  m.def("submit", &tt::runtime::submit, py::arg("device"),
        py::arg("executable"), py::arg("program_index"), py::arg("inputs"),
        py::arg("outputs"), "Submit a binary for execution");
}
