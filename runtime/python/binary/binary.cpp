// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/types.h"

#include "tt/runtime/detail/python/nanobind_headers.h"
#include <nanobind/stl/pair.h>

namespace nb = nanobind;

namespace tt::runtime::python {

void registerBinaryBindings(nb::module_ &m) {
  nb::class_<tt::runtime::Flatbuffer>(m, "Flatbuffer")
      .def_prop_ro("version", &tt::runtime::Flatbuffer::getVersion)
      .def_prop_ro("schema_hash", &tt::runtime::Flatbuffer::getSchemaHash)
      .def("check_schema_hash", &tt::runtime::Flatbuffer::checkSchemaHash)
      .def_prop_ro("ttmlir_git_hash",
                   &tt::runtime::Flatbuffer::getTTMLIRGitHash)
      .def_prop_ro("file_identifier",
                   &tt::runtime::Flatbuffer::getFileIdentifier)
      .def("as_json", &tt::runtime::Flatbuffer::asJson)
      .def("store", &tt::runtime::Flatbuffer::store);

  nb::class_<tt::runtime::Binary>(m, "Binary")
      .def_prop_ro("version", &tt::runtime::Binary::getVersion)
      .def_prop_ro("schema_hash", &tt::runtime::Flatbuffer::getSchemaHash)
      .def("check_schema_hash", &tt::runtime::Flatbuffer::checkSchemaHash)
      .def_prop_ro("ttmlir_git_hash", &tt::runtime::Binary::getTTMLIRGitHash)
      .def_prop_ro("file_identifier", &tt::runtime::Binary::getFileIdentifier)
      .def("as_json", &tt::runtime::Binary::asJson)
      .def("store", &tt::runtime::Binary::store)
      .def("get_debug_info_golden", &::tt::runtime::Binary::getDebugInfoGolden,
           nb::rv_policy::reference)
      .def("get_system_desc_as_json", &tt::runtime::Binary::getSystemDescAsJson)
      .def("get_num_programs", &tt::runtime::Binary::getNumPrograms)
      .def("get_program_name", &tt::runtime::Binary::getProgramName)
      .def("is_program_private", &tt::runtime::Binary::isProgramPrivate)
      .def("get_program_ops_as_json", &tt::runtime::Binary::getProgramOpsAsJson)
      .def("get_program_inputs_as_json",
           &tt::runtime::Binary::getProgramInputsAsJson)
      .def("get_program_outputs_as_json",
           &tt::runtime::Binary::getProgramOutputsAsJson)
      .def("get_mlir_as_json", &tt::runtime::Binary::getMlirAsJson)
      .def("get_tensor_cache",
           [](tt::runtime::Binary &bin) {
             return bin.getConstEvalTensorCache();
           })
      .def("get_program_mesh_shape", &tt::runtime::Binary::getProgramMeshShape);

  nb::class_<tt::runtime::SystemDesc>(m, "SystemDesc")
      .def_prop_ro("version", &tt::runtime::SystemDesc::getVersion)
      .def_prop_ro("schema_hash", &tt::runtime::Flatbuffer::getSchemaHash)
      .def("check_schema_hash", &tt::runtime::Flatbuffer::checkSchemaHash)
      .def_prop_ro("ttmlir_git_hash",
                   &tt::runtime::SystemDesc::getTTMLIRGitHash)
      .def_prop_ro("file_identifier",
                   &tt::runtime::SystemDesc::getFileIdentifier)
      .def("as_json", &tt::runtime::SystemDesc::asJson)
      .def("store", &tt::runtime::SystemDesc::store);

  m.def("load_from_path", &tt::runtime::Flatbuffer::loadFromPath);
  m.def("load_binary_from_path", &tt::runtime::Binary::loadFromPath);
  m.def("load_binary_from_capsule", [](nb::capsule capsule) {
    std::shared_ptr<void> *binary =
        static_cast<std::shared_ptr<void> *>(capsule.data());
    return tt::runtime::Binary(tt::runtime::Flatbuffer(*binary).handle);
  });
  m.def("load_system_desc_from_path", &tt::runtime::SystemDesc::loadFromPath);

  /**
   * Binding for the `GoldenTensor` type
   */
  nb::class_<tt::target::GoldenTensor>(m, "GoldenTensor")
      .def_prop_ro("name",
                   [](const ::tt::target::GoldenTensor *t) -> std::string {
                     assert(t != nullptr && t->name() != nullptr);
                     return t->name()->str();
                   })
      .def_prop_ro("shape",
                   [](const ::tt::target::GoldenTensor *t) -> std::vector<int> {
                     assert(t != nullptr && t->shape() != nullptr);
                     return std::vector<int>(t->shape()->begin(),
                                             t->shape()->end());
                   })
      .def_prop_ro("stride",
                   [](const ::tt::target::GoldenTensor *t) -> std::vector<int> {
                     assert(t != nullptr && t->stride() != nullptr);
                     return std::vector<int>(t->stride()->begin(),
                                             t->stride()->end());
                   })
      .def_prop_ro("dtype", &::tt::target::GoldenTensor::dtype)
      .def("get_data_buffer",
           [](const tt::target::GoldenTensor *t) -> nb::bytearray {
             assert(t != nullptr && t->data() != nullptr &&
                    t->shape() != nullptr && t->stride() != nullptr);

             size_t itemSize = 0;
             switch (t->dtype()) {
             case tt::target::DataType::UInt8:
               itemSize = sizeof(uint8_t);
               break;

             case tt::target::DataType::UInt16:
               itemSize = sizeof(uint16_t);
               break;

             case tt::target::DataType::UInt32:
               itemSize = sizeof(uint32_t);
               break;

             case tt::target::DataType::Int32:
               itemSize = sizeof(int32_t);
               break;

             case tt::target::DataType::Float32:
               itemSize = sizeof(float);
               break;

             case tt::target::DataType::BFloat16:
               itemSize = 2;
               break;

             default:
               throw std::runtime_error(
                   "Only 32-bit floats, unsigned ints, and bfloat16 "
                   "are currently supported "
                   "for GoldenTensor bindings");
             }

             const uint8_t *data = t->data()->data();
             size_t size =
                 std::accumulate(t->shape()->begin(), t->shape()->end(),
                                 static_cast<size_t>(1),
                                 std::multiplies<size_t>()) *
                 itemSize;
             return nb::bytearray(data, size);
           });

  nb::class_<tt::runtime::TensorCache>(m, "TensorCache")
      .def(nb::init<>())
      .def("clear", &tt::runtime::TensorCache::clear)
      .def("size", &tt::runtime::TensorCache::size)
      .def(
          "remove_program",
          [](tt::runtime::TensorCache &cache, const int meshId,
             size_t programIndex) {
            std::string outerKey =
                tt::runtime::generateCacheOuterKey(meshId, programIndex);
            cache.remove(outerKey);
          },
          "Remove cache entries for a specific device id and program index");
}
} // namespace tt::runtime::python
