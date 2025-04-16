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
      .def("get_debug_info_golden", &::tt::runtime::Binary::getDebugInfoGolden,
           py::return_value_policy::reference);
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

  /**
   * Binding for the `GoldenTensor` type
   */
  py::class_<tt::target::GoldenTensor>(m, "GoldenTensor", py::buffer_protocol())
      .def_property_readonly(
          "name",
          [](::tt::target::GoldenTensor const *t) -> std::string {
            assert(t != nullptr && t->name() != nullptr);
            return t->name()->str();
          })
      .def_property_readonly(
          "shape",
          [](::tt::target::GoldenTensor const *t) -> std::vector<int> {
            assert(t != nullptr && t->shape() != nullptr);
            return std::vector<int>(t->shape()->begin(), t->shape()->end());
          })
      .def_property_readonly(
          "stride",
          [](::tt::target::GoldenTensor const *t) -> std::vector<int> {
            assert(t != nullptr && t->stride() != nullptr);
            return std::vector<int>(t->stride()->begin(), t->stride()->end());
          })
      .def_property_readonly("dtype", &::tt::target::GoldenTensor::dtype)
      .def_buffer([](tt::target::GoldenTensor const *t) -> py::buffer_info {
        assert(t != nullptr && t->data() != nullptr && t->shape() != nullptr &&
               t->stride() != nullptr);

        // Format string to be passed to `py::buffer_info`
        std::string format;

        // Element size to be passed to `py::buffer_info`
        size_t size;

        switch (t->dtype()) {

        case tt::target::DataType::UInt8:
          format = py::format_descriptor<uint8_t>::format();
          size = sizeof(uint8_t);
          break;

        case tt::target::DataType::UInt16:
          format = py::format_descriptor<uint16_t>::format();
          size = sizeof(uint16_t);
          break;

        case tt::target::DataType::UInt32:
          format = py::format_descriptor<uint32_t>::format();
          size = sizeof(uint32_t);
          break;

        case tt::target::DataType::Int32:
          format = py::format_descriptor<int32_t>::format();
          size = sizeof(int32_t);
          break;

        case tt::target::DataType::Float32:
          format = py::format_descriptor<float>::format();
          size = sizeof(float);
          break;

        default:
          throw std::runtime_error(
              "Only 32-bit floats and unsigned ints are currently supported "
              "for GoldenTensor bindings");
        }

        return py::buffer_info(
            (void *)t->data()->data(), /* ptr to underlying data */
            size,                      /* size of element */
            format,                    /* format */
            t->shape()->size(),        /* rank */
            *(t->shape()),             /* shape */
            *(t->stride()),            /* stride of buffer */
            false                      /* read only */
        );
      });
}
