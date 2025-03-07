// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include <optional>

namespace mlir::ttmlir::python {

void populateOptimizerOverridesModule(nb::module_ &m) {

  nb::class_<tt::ttnn::OptimizerOverridesHandler>(m,
                                                  "OptimizerOverridesHandler")
      .def(nb::init<>())

      .def("set_enable_optimizer",
           &tt::ttnn::OptimizerOverridesHandler::setEnableOptimizer)
      .def("get_enable_optimizer",
           &tt::ttnn::OptimizerOverridesHandler::getEnableOptimizer)

      .def("set_memory_reconfig",
           &tt::ttnn::OptimizerOverridesHandler::setMemoryReconfig)
      .def("get_memory_reconfig",
           &tt::ttnn::OptimizerOverridesHandler::getMemoryReconfig)

      .def("set_enable_memory_layout_analysis",
           &tt::ttnn::OptimizerOverridesHandler::setEnableMemoryLayoutAnalysis)
      .def("get_enable_memory_layout_analysis",
           &tt::ttnn::OptimizerOverridesHandler::getEnableMemoryLayoutAnalysis)

      .def("set_enable_memory_layout_analysis_policy",
           &tt::ttnn::OptimizerOverridesHandler::
               setEnableMemoryLayoutAnalysisPolicy)
      .def("get_enable_memory_layout_analysis_policy",
           &tt::ttnn::OptimizerOverridesHandler::
               getEnableMemoryLayoutAnalysisPolicy)

      .def("set_memory_layout_analysis_policy",
           &tt::ttnn::OptimizerOverridesHandler::setMemoryLayoutAnalysisPolicy)
      .def("get_memory_layout_analysis_policy",
           &tt::ttnn::OptimizerOverridesHandler::getMemoryLayoutAnalysisPolicy)

      .def("set_system_desc_path",
           &tt::ttnn::OptimizerOverridesHandler::setSystemDescPath)
      .def("get_system_desc_path",
           &tt::ttnn::OptimizerOverridesHandler::getSystemDescPath)

      .def("set_max_legal_layouts",
           &tt::ttnn::OptimizerOverridesHandler::setMaxLegalLayouts)
      .def("get_max_legal_layouts",
           &tt::ttnn::OptimizerOverridesHandler::getMaxLegalLayouts)

      .def("set_mesh_shape", &tt::ttnn::OptimizerOverridesHandler::setMeshShape)
      .def("get_mesh_shape", &tt::ttnn::OptimizerOverridesHandler::getMeshShape)

      .def("get_input_layout_overrides",
           &tt::ttnn::OptimizerOverridesHandler::
               getInputLayoutOverridesNanobindWrapper)
      .def("get_output_layout_overrides",
           &tt::ttnn::OptimizerOverridesHandler::
               getOutputLayoutOverridesNanobindWrapper)

      .def("add_input_layout_override",
           &tt::ttnn::OptimizerOverridesHandler::
               addInputLayoutOverrideNanobindWrapper)
      .def("add_output_layout_override",
           &tt::ttnn::OptimizerOverridesHandler::
               addOutputLayoutOverrideNanobindWrapper)

      .def("to_string", &tt::ttnn::OptimizerOverridesHandler::toString);

  nb::enum_<mlir::tt::MemoryLayoutAnalysisPolicyType>(
      m, "MemoryLayoutAnalysisPolicyType")
      .value("DFSharding", mlir::tt::MemoryLayoutAnalysisPolicyType::DFSharding)
      .value("GreedyL1Interleaved",
             mlir::tt::MemoryLayoutAnalysisPolicyType::GreedyL1Interleaved)
      .value("BFInterleaved",
             mlir::tt::MemoryLayoutAnalysisPolicyType::BFInterleaved);

  nb::enum_<mlir::tt::ttnn::BufferType>(m, "BufferType")
      .value("DRAM", mlir::tt::ttnn::BufferType::DRAM)
      .value("L1", mlir::tt::ttnn::BufferType::L1)
      .value("SystemMemory", mlir::tt::ttnn::BufferType::SystemMemory)
      .value("L1Small", mlir::tt::ttnn::BufferType::L1Small)
      .value("Trace", mlir::tt::ttnn::BufferType::Trace);

  nb::enum_<mlir::tt::ttnn::Layout>(m, "Layout")
      .value("RowMajor", mlir::tt::ttnn::Layout::RowMajor)
      .value("Tile", mlir::tt::ttnn::Layout::Tile)
      .value("Invalid", mlir::tt::ttnn::Layout::Invalid);

  nb::enum_<mlir::tt::ttnn::TensorMemoryLayout>(m, "TensorMemoryLayout")
      .value("Interleaved", mlir::tt::ttnn::TensorMemoryLayout::Interleaved)
      .value("SingleBank", mlir::tt::ttnn::TensorMemoryLayout::SingleBank)
      .value("HeightSharded", mlir::tt::ttnn::TensorMemoryLayout::HeightSharded)
      .value("WidthSharded", mlir::tt::ttnn::TensorMemoryLayout::WidthSharded)
      .value("BlockSharded", mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);

  nb::enum_<mlir::tt::DataType>(m, "DataType")
      .value("Float32", mlir::tt::DataType::Float32)
      .value("Float16", mlir::tt::DataType::Float16)
      .value("BFloat16", mlir::tt::DataType::BFloat16)
      .value("BFP_Float8", mlir::tt::DataType::BFP_Float8)
      .value("BFP_BFloat8", mlir::tt::DataType::BFP_BFloat8)
      .value("BFP_Float4", mlir::tt::DataType::BFP_Float4)
      .value("BFP_BFloat4", mlir::tt::DataType::BFP_BFloat4)
      .value("BFP_Float2", mlir::tt::DataType::BFP_Float2)
      .value("BFP_BFloat2", mlir::tt::DataType::BFP_BFloat2)
      .value("UInt32", mlir::tt::DataType::UInt32)
      .value("UInt16", mlir::tt::DataType::UInt16)
      .value("UInt8", mlir::tt::DataType::UInt8)
      .value("Int32", mlir::tt::DataType::Int32);

  nb::class_<mlir::tt::ttnn::InputLayoutOverrideParams>(
      m, "InputLayoutOverrideParams")
      .def(nb::init<>())
      .def_prop_rw(
          "operand_idxes",
          [](const mlir::tt::ttnn::InputLayoutOverrideParams &obj) {
            // Getter: Convert SmallVector to std::vector
            return std::vector<int64_t>(obj.operandIdxes.begin(),
                                        obj.operandIdxes.end());
          },
          [](mlir::tt::ttnn::InputLayoutOverrideParams &obj,
             const std::vector<int64_t> &input) {
            // Setter: Convert std::vector to SmallVector
            obj.operandIdxes.clear();
            obj.operandIdxes.append(input.begin(), input.end());
          });

  nb::class_<mlir::tt::ttnn::OutputLayoutOverrideParams>(
      m, "OutputLayoutOverrideParams")
      .def(nb::init<>())
      .def_prop_rw(
          "grid",
          [](const mlir::tt::ttnn::OutputLayoutOverrideParams &obj) {
            // Getter: Convert SmallVector to std::vector
            if (obj.grid.has_value()) {
              return std::make_optional<std::vector<int64_t>>(obj.grid->begin(),
                                                              obj.grid->end());
            }
            return std::make_optional<std::vector<int64_t>>();
          },
          [](mlir::tt::ttnn::OutputLayoutOverrideParams &obj,
             const std::vector<int64_t> &input) {
            // Setter: Convert std::vector to SmallVector
            if (!obj.grid.has_value()) {
              obj.grid = SmallVector<int64_t, 2>();
            } else {
              obj.grid->clear();
            }
            obj.grid->append(input.begin(), input.end());
          })
      .def_rw("buffer_type",
              &mlir::tt::ttnn::OutputLayoutOverrideParams::bufferType)
      .def_rw("tensor_memory_layout",
              &mlir::tt::ttnn::OutputLayoutOverrideParams::tensorMemoryLayout)
      .def_rw("memory_layout",
              &mlir::tt::ttnn::OutputLayoutOverrideParams::memoryLayout)
      .def_rw("data_type",
              &mlir::tt::ttnn::OutputLayoutOverrideParams::dataType)
      .def("set_buffer_type_from_str",
           [](mlir::tt::ttnn::OutputLayoutOverrideParams &obj,
              const std::string &value) {
             if (auto bufferType_ =
                     mlir::tt::ttnn::symbolizeBufferType(value)) {
               obj.bufferType = bufferType_;
             } else {
               throw std::invalid_argument("Invalid buffer type: " + value);
             }
           })
      .def("set_tensor_memory_layout_from_str",
           [](mlir::tt::ttnn::OutputLayoutOverrideParams &obj,
              const std::string &value) {
             if (auto tensorMemoryLayout =
                     mlir::tt::ttnn::symbolizeTensorMemoryLayout(value)) {
               obj.tensorMemoryLayout = tensorMemoryLayout;
             } else {
               throw std::invalid_argument("Invalid tensor memory layout: " +
                                           value);
             }
           })
      .def("set_memory_layout_from_str",
           [](mlir::tt::ttnn::OutputLayoutOverrideParams &obj,
              const std::string &value) {
             if (auto memoryLayout_ = mlir::tt::ttnn::symbolizeLayout(value)) {
               obj.memoryLayout = memoryLayout_;
             } else {
               throw std::invalid_argument("Invalid memory layout: " + value);
             }
           })
      .def("set_data_type_from_str",
           [](mlir::tt::ttnn::OutputLayoutOverrideParams &obj,
              const std::string &value) {
             if (auto dataType_ = mlir::tt::DataTypeStringToEnum(value)) {
               obj.dataType = dataType_;
             } else {
               throw std::invalid_argument("Invalid data type: " + value);
             }
           });
}

} // namespace mlir::ttmlir::python
