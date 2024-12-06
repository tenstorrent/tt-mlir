// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"

namespace mlir::ttmlir::python {

void populateOptimizerOverridesModule(py::module &m) {

  py::class_<tt::ttnn::OptimizerOverridesHandler>(m,
                                                  "OptimizerOverridesHandler")
      .def(py::init<>())

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
               getInputLayoutOverridesPybindWrapper)
      .def("get_output_layout_overrides",
           &tt::ttnn::OptimizerOverridesHandler::
               getOutputLayoutOverridesPybindWrapper)

      .def("add_input_layout_override", &tt::ttnn::OptimizerOverridesHandler::
                                            addInputLayoutOverridePybindWrapper)
      .def("add_output_layout_override",
           &tt::ttnn::OptimizerOverridesHandler::
               addOutputLayoutOverridePybindWrapper)

      .def("to_string", &tt::ttnn::OptimizerOverridesHandler::toString);

  py::enum_<mlir::tt::MemoryLayoutAnalysisPolicyType>(
      m, "MemoryLayoutAnalysisPolicyType")
      .value("DFSharding", mlir::tt::MemoryLayoutAnalysisPolicyType::DFSharding)
      .value("L1Interleaved",
             mlir::tt::MemoryLayoutAnalysisPolicyType::L1Interleaved);

  py::enum_<mlir::tt::ttnn::BufferType>(m, "BufferType")
      .value("DRAM", mlir::tt::ttnn::BufferType::DRAM)
      .value("L1", mlir::tt::ttnn::BufferType::L1)
      .value("SystemMemory", mlir::tt::ttnn::BufferType::SystemMemory)
      .value("L1Small", mlir::tt::ttnn::BufferType::L1Small)
      .value("Trace", mlir::tt::ttnn::BufferType::Trace);

  py::enum_<mlir::tt::ttnn::Layout>(m, "Layout")
      .value("RowMajor", mlir::tt::ttnn::Layout::RowMajor)
      .value("Tile", mlir::tt::ttnn::Layout::Tile)
      .value("Invalid", mlir::tt::ttnn::Layout::Invalid);

  py::enum_<mlir::tt::ttnn::TensorMemoryLayout>(m, "TensorMemoryLayout")
      .value("NONE", mlir::tt::ttnn::TensorMemoryLayout::None)
      .value("Interleaved", mlir::tt::ttnn::TensorMemoryLayout::Interleaved)
      .value("SingleBank", mlir::tt::ttnn::TensorMemoryLayout::SingleBank)
      .value("HeightSharded", mlir::tt::ttnn::TensorMemoryLayout::HeightSharded)
      .value("WidthSharded", mlir::tt::ttnn::TensorMemoryLayout::WidthSharded)
      .value("BlockSharded", mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);

  py::enum_<mlir::tt::DataType>(m, "DataType")
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
      .value("UInt8", mlir::tt::DataType::UInt8);

  py::class_<mlir::tt::ttnn::InputLayoutOverrideParams>(
      m, "InputLayoutOverrideParams")
      .def(py::init<>())
      .def_property(
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

  py::class_<mlir::tt::ttnn::OutputLayoutOverrideParams>(
      m, "OutputLayoutOverrideParams")
      .def(py::init<>())
      .def_property(
          "grid",
          [](const mlir::tt::ttnn::OutputLayoutOverrideParams &obj) {
            // Getter: Convert SmallVector to std::vector
            return std::vector<int64_t>(obj.grid.begin(), obj.grid.end());
          },
          [](mlir::tt::ttnn::OutputLayoutOverrideParams &obj,
             const std::vector<int64_t> &input) {
            // Setter: Convert std::vector to SmallVector
            obj.grid.clear();
            obj.grid.append(input.begin(), input.end());
          })
      .def_readwrite("buffer_type",
                     &mlir::tt::ttnn::OutputLayoutOverrideParams::bufferType)
      .def_readwrite(
          "tensor_memory_layout",
          &mlir::tt::ttnn::OutputLayoutOverrideParams::tensorMemoryLayout)
      .def_readwrite("memory_layout",
                     &mlir::tt::ttnn::OutputLayoutOverrideParams::memoryLayout)
      .def_readwrite("data_type",
                     &mlir::tt::ttnn::OutputLayoutOverrideParams::dataType);
}

} // namespace mlir::ttmlir::python
