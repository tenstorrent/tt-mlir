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

      .def("set_enable_l1_interleaved_fallback_analysis",
           &tt::ttnn::OptimizerOverridesHandler::
               setEnableL1InterleavedFallbackAnalysis)
      .def("get_enable_l1_interleaved_fallback_analysis",
           &tt::ttnn::OptimizerOverridesHandler::
               getEnableL1InterleavedFallbackAnalysis)

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

      .def("set_tensor_l1_usage_cap",
           &tt::ttnn::OptimizerOverridesHandler::setTensorL1UsageCap)
      .def("get_tensor_l1_usage_cap",
           &tt::ttnn::OptimizerOverridesHandler::getTensorL1UsageCap)

      .def("get_insert_memreconfig", &tt::ttnn::OptimizerOverridesHandler::
                                         getInsertMemReconfigNanobindWrapper)
      .def("get_output_layout_overrides",
           &tt::ttnn::OptimizerOverridesHandler::
               getOutputLayoutOverridesNanobindWrapper)
      .def("get_conv2d_config_overrides",
           &tt::ttnn::OptimizerOverridesHandler::
               getConv2dConfigOverridesNanobindWrapper)

      .def("add_insert_memreconfig", &tt::ttnn::OptimizerOverridesHandler::
                                         addInsertMemReconfigNanobindWrapper)
      .def("add_output_layout_override",
           &tt::ttnn::OptimizerOverridesHandler::
               addOutputLayoutOverrideNanobindWrapper)
      .def("add_conv2d_config_override",
           &tt::ttnn::OptimizerOverridesHandler::
               addConv2dConfigOverrideNanobindWrapper)

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
      .value("HeightSharded", mlir::tt::ttnn::TensorMemoryLayout::HeightSharded)
      .value("WidthSharded", mlir::tt::ttnn::TensorMemoryLayout::WidthSharded)
      .value("BlockSharded", mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);

  nb::enum_<mlir::tt::ttcore::DataType>(m, "DataType")
      .value("Float32", mlir::tt::ttcore::DataType::Float32)
      .value("Float16", mlir::tt::ttcore::DataType::Float16)
      .value("BFloat16", mlir::tt::ttcore::DataType::BFloat16)
      .value("BFP_Float8", mlir::tt::ttcore::DataType::BFP_Float8)
      .value("BFP_BFloat8", mlir::tt::ttcore::DataType::BFP_BFloat8)
      .value("BFP_Float4", mlir::tt::ttcore::DataType::BFP_Float4)
      .value("BFP_BFloat4", mlir::tt::ttcore::DataType::BFP_BFloat4)
      .value("BFP_Float2", mlir::tt::ttcore::DataType::BFP_Float2)
      .value("BFP_BFloat2", mlir::tt::ttcore::DataType::BFP_BFloat2)
      .value("UInt32", mlir::tt::ttcore::DataType::UInt32)
      .value("UInt16", mlir::tt::ttcore::DataType::UInt16)
      .value("UInt8", mlir::tt::ttcore::DataType::UInt8)
      .value("Int32", mlir::tt::ttcore::DataType::Int32);

  nb::class_<mlir::tt::ttnn::InsertMemReconfigParams>(m,
                                                      "InsertMemReconfigParams")
      .def(nb::init<>())
      .def_prop_rw(
          "operand_idxes",
          [](const mlir::tt::ttnn::InsertMemReconfigParams &obj) {
            // Getter: Convert SmallVector to std::vector
            return std::vector<int64_t>(obj.operandIdxes.begin(),
                                        obj.operandIdxes.end());
          },
          [](mlir::tt::ttnn::InsertMemReconfigParams &obj,
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
             if (auto dataType_ =
                     mlir::tt::ttcore::DataTypeStringToEnum(value)) {
               obj.dataType = dataType_;
             } else {
               throw std::invalid_argument("Invalid data type: " + value);
             }
           })
      .def("empty", &mlir::tt::ttnn::OutputLayoutOverrideParams::empty);

  nb::class_<mlir::tt::ttnn::Conv2dConfigOverrideParams>(
      m, "Conv2dConfigOverrideParams")
      .def(nb::init<>())
      .def_rw("weights_dtype",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::weightsDtype)
      .def_rw("activation",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::activation)
      .def_rw("deallocate_activation",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::deallocateActivation)
      .def_rw("reallocate_halo_output",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::reallocateHaloOutput)
      .def_rw("act_block_h_override",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::actBlockHOverride)
      .def_rw("act_block_w_div",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::actBlockWDiv)
      .def_rw("reshard_if_not_optimal",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::reshardIfNotOptimal)
      .def_rw(
          "override_sharding_config",
          &mlir::tt::ttnn::Conv2dConfigOverrideParams::overrideShardingConfig)
      .def_rw("shard_layout",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::shardLayout)
      .def_rw("core_grid",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::coreGrid)
      .def_rw("transpose_shards",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::transposeShards)
      .def_rw("output_layout",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::outputLayout)
      .def_rw(
          "enable_act_double_buffer",
          &mlir::tt::ttnn::Conv2dConfigOverrideParams::enableActDoubleBuffer)
      .def_rw("enable_weights_double_buffer",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::
                  enableWeightsDoubleBuffer)
      .def_rw("enable_split_reader",
              &mlir::tt::ttnn::Conv2dConfigOverrideParams::enableSplitReader)
      .def("set_weights_dtype_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             if (auto weightsDtype_ =
                     mlir::tt::ttcore::DataTypeStringToEnum(value)) {
               obj.weightsDtype = weightsDtype_;
             } else {
               throw std::invalid_argument("Invalid weights dtype: " + value);
             }
           })
      .def("set_activation_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             if (value != "none" && value != "relu") {
               throw std::invalid_argument("Invalid activation: " + value);
             }
             obj.activation = value;
           })
      .def("set_deallocate_activation_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.deallocateActivation = (value == "True");
           })
      .def("set_reallocate_halo_output_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.reallocateHaloOutput = (value == "True");
           })
      .def("set_act_block_h_override_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.actBlockHOverride = std::stoul(value);
           })
      .def("set_act_block_w_div_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.actBlockWDiv = std::stoul(value);
           })
      .def("set_reshard_if_not_optimal_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.reshardIfNotOptimal = (value == "True");
           })
      .def("set_override_sharding_config_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.overrideShardingConfig = (value == "True");
           })
      .def("set_shard_layout_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             if (auto shardLayout_ =
                     mlir::tt::ttnn::symbolizeTensorMemoryLayout(value)) {
               obj.shardLayout = shardLayout_;
             } else {
               throw std::invalid_argument("Invalid shard layout: " + value);
             }
           })
      // TODO(vkovacevic): #2781
      //.def("set_core_grid_from_str",
      //[](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj, const std::string
      //&value) {
      //    if (auto coreGrid_ = mlir::tt::ttnn::symbolizeAttribute(value)) {
      //        obj.coreGrid = coreGrid_;
      //    } else {
      //        throw std::invalid_argument("Invalid core grid: " + value);
      //    }
      //})
      .def("set_transpose_shards_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.transposeShards = (value == "True");
           })
      .def("set_output_layout_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             if (auto outputLayout_ = mlir::tt::ttnn::symbolizeLayout(value)) {
               obj.outputLayout = outputLayout_;
             } else {
               throw std::invalid_argument("Invalid output layout: " + value);
             }
           })
      .def("set_enable_act_double_buffer_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.enableActDoubleBuffer = (value == "True");
           })
      .def("set_enable_weights_double_buffer_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.enableWeightsDoubleBuffer = (value == "True");
           })
      .def("set_enable_split_reader_from_str",
           [](mlir::tt::ttnn::Conv2dConfigOverrideParams &obj,
              const std::string &value) {
             obj.enableSplitReader = (value == "True");
           })
      .def("empty", &mlir::tt::ttnn::Conv2dConfigOverrideParams::empty);
}

} // namespace mlir::ttmlir::python
