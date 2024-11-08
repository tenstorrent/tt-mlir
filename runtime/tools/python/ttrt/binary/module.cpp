// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "tt/runtime/types.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() =
      "ttrt.binary python extension for loading / inspecting tt binary files";

  py::class_<tt::runtime::Flatbuffer>(m, "Flatbuffer")
      .def_property_readonly("version", &tt::runtime::Flatbuffer::getVersion)
      .def_property_readonly("ttmlir_git_hash",
                             &tt::runtime::Flatbuffer::getTTMLIRGitHash)
      .def_property_readonly("file_identifier",
                             &tt::runtime::Flatbuffer::getFileIdentifier)
      .def("as_json", &tt::runtime::Flatbuffer::asJson)
      .def("store", &tt::runtime::Flatbuffer::store);
  py::class_<tt::runtime::Binary>(m, "Binary")
      .def_property_readonly("version", &tt::runtime::Binary::getVersion)
      .def_property_readonly("ttmlir_git_hash",
                             &tt::runtime::Binary::getTTMLIRGitHash)
      .def_property_readonly("file_identifier",
                             &tt::runtime::Binary::getFileIdentifier)
      .def("as_json", &tt::runtime::Binary::asJson)
      .def("store", &tt::runtime::Binary::store)
      .def("get_debug_info_golden", [](tt::runtime::Binary &binary,
                                       std::string &loc) {
        const ::tt::target::GoldenTensor *goldenTensor =
            binary.getDebugInfoGolden(loc);
        if (goldenTensor == nullptr) {
          return std::vector<float>();
        }

        int totalDataSize = std::accumulate((*goldenTensor->shape()).begin(),
                                            (*goldenTensor->shape()).end(), 1,
                                            std::multiplies<int64_t>());
        std::vector<float> dataVec(totalDataSize);
        std::memcpy(dataVec.data(), goldenTensor->data(), totalDataSize);
        return dataVec;
      });
  py::class_<tt::runtime::SystemDesc>(m, "SystemDesc")
      .def_property_readonly("version", &tt::runtime::SystemDesc::getVersion)
      .def_property_readonly("ttmlir_git_hash",
                             &tt::runtime::SystemDesc::getTTMLIRGitHash)
      .def_property_readonly("file_identifier",
                             &tt::runtime::SystemDesc::getFileIdentifier)
      .def("as_json", &tt::runtime::SystemDesc::asJson)
      .def("store", &tt::runtime::SystemDesc::store);
  m.def("load_from_path", &tt::runtime::Flatbuffer::loadFromPath);
  m.def("load_binary_from_path", &tt::runtime::Binary::loadFromPath);
  m.def("load_binary_from_capsule", [](py::capsule capsule) {
    std::shared_ptr<void> *binary =
        static_cast<std::shared_ptr<void> *>(capsule.get_pointer());
    return tt::runtime::Binary(
        tt::runtime::Flatbuffer(*binary)
            .handle); // Dereference capsule, and then dereference shared_ptr*
  });
  m.def("load_system_desc_from_path", &tt::runtime::SystemDesc::loadFromPath);
}
