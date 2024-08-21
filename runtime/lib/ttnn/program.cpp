// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>

#include "common/tt_backend_api_types.hpp"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/runtime.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "types_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/generic/generic_op/device/generic_op_device_operation.hpp"
#include "types_generated.h"
#include "utils.h"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

// It seems like `ttnn::to_layout` cannot be called inside of the
// `tt::runtime::ttnn` namespace.  TTNN uses a lot of metaprogramming and for
// some reason a static_assert fails when this is called from within our
// namespace.
ttnn::Tensor tilize(ttnn::Tensor const &input) {
  return ttnn::to_layout(input, ::ttnn::TILE_LAYOUT, std::nullopt, std::nullopt,
                         (Device *)nullptr);
}

ttnn::Tensor untilize(ttnn::Tensor const &input) {
  return ttnn::to_layout(input, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                         std::nullopt, (Device *)nullptr);
}

namespace tt::runtime::ttnn {

static ::ttnn::Tensor convertDataType(const ::ttnn::Tensor &input,
                                      const ::ttnn::DataType &targetDataType) {
  const ::ttnn::StorageType storageType = input.storage_type();
  if (storageType == ::tt::tt_metal::StorageType::BORROWED) {
    return ::ttnn::to_dtype(input, targetDataType);
  } else if (storageType == ::tt::tt_metal::StorageType::DEVICE) {
    if (input.get_layout() != ::ttnn::TILE_LAYOUT) {
      // typecast op requires tilized tensor
      ::ttnn::Tensor converted =
          ::ttnn::typecast(::tilize(input), targetDataType);
      // untilize and return
      return ::untilize(converted);
    }
    return ::ttnn::typecast(input, targetDataType);
  } else {
    throw std::runtime_error("Unsupported storage type");
  }
}

/* TODO: Blocked by issue #272, ideal flow is to determine tilize/untilize with
 * tile_shape */
static ::ttnn::Tensor
updateLayoutAndDataType(const ::ttnn::Tensor &inputTensor,
                        const ::ttnn::DataType targetDataType,
                        const bool shouldTilize, const bool shouldUntilize) {
  ::ttnn::Tensor outputTensor = inputTensor;
  const bool shouldConvertDataType = inputTensor.get_dtype() != targetDataType;
  // const int targetTileX = targetTileShape->x();
  // const int targetTileY = targetTileShape->y();
  // const bool shouldTilize =
  //     targetTileX == 32 and targetTileY == 32 and
  //     inputTensor.get_layout() == ::ttnn::ROW_MAJOR_LAYOUT;
  // const bool shouldUntilize = (targetTileX != 32 or targetTileY != 32) and
  //                             inputTensor.get_layout() ==
  //                             ::ttnn::TILE_LAYOUT;
  if (shouldTilize) {
    outputTensor = ::tilize(outputTensor);
  } else if (shouldUntilize) {
    outputTensor = ::untilize(outputTensor);
  }
  if (shouldConvertDataType) {
    outputTensor = convertDataType(outputTensor, targetDataType);
  }
  return outputTensor;
}

// TODO: right now hardcoding tilize/untilize, should determine with tile shape
// blocked by issue #272
static void
run(::tt::target::ttnn::ToMemoryConfigOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  const ::ttnn::Tensor &inputTensor = *liveTensors.at(op->in0()->global_id());
  assert(inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED or
         inputTensor.storage_type() == ::tt::tt_metal::StorageType::DEVICE);

  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  TT_FATAL(utils::isValidTileShape(targetTileShape),
           "Invalid tile shape ({}, {})", targetTileShape->x(),
           targetTileShape->y());

  ::tt::target::DataType targetDataType =
      op->out()->desc()->layout()->memory_desc()->data_type();
  ::ttnn::DataType targetDataTypeTTNN = utils::toTTNNDataType(targetDataType);

  const ::tt::target::MemorySpace targetMemorySpace =
      op->out()->desc()->layout()->memory_desc()->memory_space();

  switch (targetMemorySpace) {
  // This case should only be used when gathering outputs at the end of the
  // program
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO: {
    ::ttnn::Tensor result;
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      result =
          updateLayoutAndDataType(inputTensor, targetDataTypeTTNN, false, true);
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      result = updateLayoutAndDataType(inputTensor.cpu(), targetDataTypeTTNN,
                                       false, true);
    }
    ::ttnn::Tensor &outputTensor = *liveTensors.at(op->out()->global_id());
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(result);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    std::uint32_t size = result.volume() * result.element_size();
    std::memcpy(dst, src, size);
    break;
  }
  case ::tt::target::MemorySpace::DeviceDRAM: {
    ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::DRAM_MEMORY_CONFIG;
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      ::ttnn::Tensor result = inputTensor;
      bool shouldTilize = true;
      // device tilize requires BFLOAT16, if not then tilize on host
      if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
        result = ::tilize(result);
        shouldTilize = false;
      }
      result = ::ttnn::to_device(result, &device, memConfig);
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       false);
      tensorPool.push_back(result);
      liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      ::ttnn::Tensor result = updateLayoutAndDataType(
          inputTensor, targetDataTypeTTNN, false, false);
      result = ::ttnn::to_memory_config(result, memConfig, std::nullopt);
      tensorPool.push_back(result);
      liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    }
    break;
  }
  // Currently similar to ::tt::target::MemorySpace::DeviceDRAM
  // But will need it's own code path when we add support for sharding
  case ::tt::target::MemorySpace::DeviceL1: {
    ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::L1_MEMORY_CONFIG;
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      ::ttnn::Tensor result = inputTensor;
      bool shouldTilize = true;
      // device tilize requires BFLOAT16, if not then tilize on host
      if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
        result = ::tilize(result);
        shouldTilize = false;
      }
      result = ::ttnn::to_device(result, &device, memConfig);
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       false);
      tensorPool.push_back(result);
      liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      ::ttnn::Tensor result = updateLayoutAndDataType(
          inputTensor, targetDataTypeTTNN, false, false);
      result = ::ttnn::to_memory_config(result, memConfig, std::nullopt);
      tensorPool.push_back(result);
      liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    }
    break;
  }
  }
}

static void
run(::tt::target::ttnn::EmptyOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  ::ttnn::DataType targetDataTypeTTNN = utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());

  // TODO: ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  auto desiredLayout = ::ttnn::Layout::ROW_MAJOR;
  auto shape = ::ttnn::Shape(::tt::tt_metal::Shape(
      utils::toShapeFromFBShape(*op->out()->desc()->shape())));

  tensorPool.push_back(
      ::ttnn::empty(shape, targetDataTypeTTNN, desiredLayout, device));
  // use try emplace here so the program output tensor doesn't get overwritten
  liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
}

static void
run(::tt::target::ttnn::EltwiseOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseOpType::Add: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    auto &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    auto &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::add(lhs, rhs));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Multiply: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    auto &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    auto &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::multiply(lhs, rhs));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Subtract: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    auto &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    auto &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::subtract(lhs, rhs));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterEqual: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    ::ttnn::Tensor &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::ge(lhs, rhs));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Div: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    ::ttnn::Tensor &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::divide(lhs, rhs));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  /* Eltwise Unary */
  case ::tt::target::ttnn::EltwiseOpType::Relu: {
    assert(op->ins()->size() == 1 && "Unsupported number of inputs");
    ::ttnn::Tensor &in = *liveTensors.at(op->ins()->Get(0)->global_id());
    tensorPool.push_back(::ttnn::relu(in));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sqrt: {
    assert(op->ins()->size() == 1 && "Unsupported number of inputs");
    ::ttnn::Tensor &in = *liveTensors.at(op->ins()->Get(0)->global_id());
    tensorPool.push_back(::ttnn::sqrt(in));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sigmoid: {
    assert(op->ins()->size() == 1 && "Unsupported number of inputs");
    ::ttnn::Tensor &in = *liveTensors.at(op->ins()->Get(0)->global_id());
    tensorPool.push_back(::ttnn::sigmoid(in));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Reciprocal: {
    assert(op->ins()->size() == 1 && "Unsupported number of inputs");
    ::ttnn::Tensor &in = *liveTensors.at(op->ins()->Get(0)->global_id());
    tensorPool.push_back(::ttnn::reciprocal(in));
    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  }
}

static void
run(::tt::target::ttnn::ReductionOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  switch (op->type()) {
  case ::tt::target::ttnn::ReductionOpType::Sum: {
    auto &in = *liveTensors.at(op->in()->global_id());

    const auto *dim_arg_fb_ptr = op->dim_arg();
    std::optional<vector<int>> dim_arg =
        dim_arg_fb_ptr ? std::make_optional(std::vector<int>(
                             dim_arg_fb_ptr->begin(), dim_arg_fb_ptr->end()))
                       : std::nullopt;

    tensorPool.push_back(::ttnn::sum(in, dim_arg, op->keep_dim()));

    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Mean: {
    auto &in = *liveTensors.at(op->in()->global_id());

    const auto *dim_arg_fb_ptr = op->dim_arg();
    std::optional<vector<int>> dim_arg =
        dim_arg_fb_ptr ? std::make_optional(std::vector<int>(
                             dim_arg_fb_ptr->begin(), dim_arg_fb_ptr->end()))
                       : std::nullopt;

    tensorPool.push_back(::ttnn::mean(in, dim_arg, op->keep_dim()));

    liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
    break;
  }
  }
}

static void
run(::tt::target::ttnn::EmbeddingOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  ::ttnn::Tensor &input = *liveTensors.at(op->input()->global_id());
  ::ttnn::Tensor &weight = *liveTensors.at(op->weight()->global_id());

  tensorPool.push_back(::ttnn::embedding(input, weight));
  liveTensors.insert_or_assign(op->output()->global_id(), &tensorPool.back());
}

static void
run(::tt::target::ttnn::SoftmaxOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  ::ttnn::Tensor &in = *liveTensors.at(op->in()->global_id());
  int32_t dimension = op->dimension();

  tensorPool.push_back(::ttnn::softmax(in, dimension));
  liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
}

static void
run(::tt::target::ttnn::TransposeOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  ::ttnn::Tensor &in = *liveTensors.at(op->in()->global_id());
  int32_t dim0 = op->dim0();
  int32_t dim1 = op->dim1();
  auto input_rank = in.get_shape().rank();
  // for the current version of permute, we need to work in 4D, so we add
  // leading dimensions of size 1
  std::vector<std::int64_t> dimensionOrder(4);
  std::iota(dimensionOrder.begin(), dimensionOrder.end(), 0);
  if (dim0 < 0) {
    dim0 += 4;
  } else {
    dim0 = dim0 + 4 - input_rank;
  }
  if (dim1 < 0) {
    dim1 += 4;
  } else {
    dim1 = dim1 + 4 - input_rank;
  }
  std::swap(dimensionOrder[dim0], dimensionOrder[dim1]);
  // Ideally this would use ttnn::transpose, but since ttnn::transpose doesn't
  // work at the moment, we use this temporary solution.
  auto unsqueezed_input = ::ttnn::unsqueeze_to_4D(in);
  tensorPool.push_back(::ttnn::permute(unsqueezed_input, dimensionOrder));
  liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
}

static int computeTensorRuntimeArgument(
    const target::RuntimeArgType runtime_argument_type,
    const int tensor_index, 
    const std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    const std::list<::ttnn::Tensor> &tensorPool,
    const ::flatbuffers::Vector<::flatbuffers::Offset<target::TensorRef>>& ins) {
  
    switch(runtime_argument_type) {
      case target::RuntimeArgType::Invalid:
        assert(false && "Invalid runtime argument type");
        break;
      case target::RuntimeArgType::TensorAddr:
        auto &t = *liveTensors.at(ins[tensor_index]->global_id());
        return t.buffer()->address();
        break;
    }

    assert(false && "Invalid runtime argument type");
}

static void
run(::tt::target::ttnn::GenericOp const *op, ::ttnn::device::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {

    std::cout << "Preparing CB attributes" << std::endl;
    std::unordered_map<uint8_t, ::ttnn::operations::generic::circular_buffer_attributes_t> circular_buffer_attributes;
    for (auto cb_config : *op->cb_configs()) {
        CoreRangeSet crs = utils::get_core_range_set(cb_config->core_spec());
        
        circular_buffer_attributes.insert({
          cb_config->cb_id(),
          {
              .core_spec = crs,
              .total_size = cb_config->total_size(),
              .page_size = cb_config->page_size(),
              .data_format = tt::tt_metal::datatype_to_dataformat_converter(utils::toTTNNDataType(cb_config->data_format())),
          }
        });
    }
    std::cout << "Finished preparing CB attributes" << std::endl;

    std::cout << "Preparing compute attributes" << std::endl;
    std::vector<::ttnn::operations::generic::compute_attributes_t> compute_attributes;
    for (auto compute_kernel: *op->compute_kernels()) {
      // Convert compile args to vector.
      auto compile_args_vec = compute_kernel->compute_kernel_config()->compile_args();
      std::vector<uint32_t> compile_args(compile_args_vec->begin(), compile_args_vec->end());

      // Convert defines to map.
      auto map_defines = compute_kernel->compute_kernel_config()->defines();
      std::map<std::string, std::string> compute_config_defines;
      for (auto one_define : *map_defines) {
        compute_config_defines[one_define->key()->str()] = one_define->val()->str();
      }

      CoreRangeSet crs = utils::get_core_range_set(compute_kernel->core_spec());

      ::ttnn::operations::generic::compute_attributes_t compute_kernel_attribute = {
        .core_spec = crs,
        .kernel_path = compute_kernel->kernel_path()->str(),
        .config = ComputeConfig {
          .math_fidelity = utils::toTTNNMathFidelity(compute_kernel->compute_kernel_config()->math_fidelity()),
          .fp32_dest_acc_en = compute_kernel->compute_kernel_config()->fp32_dest(),
          .preserve_fp32_precision = compute_kernel->compute_kernel_config()->preserve_fp32(),  
          .math_approx_mode = compute_kernel->compute_kernel_config()->math_approx_mode(),
          .compile_args = compile_args,
          .defines = compute_config_defines,
        },
        .runtime_args_per_core = {}
      };

      std::cout << "Preparing runtime arguments for compute attributes" << std::endl;
      for (auto runtime_arg : *compute_kernel->runtime_args()) {
        const bool ttnn_compute = runtime_arg->ttnn_compute();
        unsigned int runtime_arg_val;

        if (ttnn_compute) {
          computeTensorRuntimeArgument(
              runtime_arg->runtime_arg_type(),
              runtime_arg->tensor_glob_id(),
              liveTensors,
              tensorPool,
              *op->ins());
        } else {
          runtime_arg_val = runtime_arg->val();
        }
        
        const uint32_t x_start = runtime_arg->core_range()->loc().x();
        const uint32_t y_start = runtime_arg->core_range()->loc().y();
        const uint32_t x_size = runtime_arg->core_range()->size().x();
        const uint32_t y_size = runtime_arg->core_range()->size().y();

        for (uint32_t x_curr = x_start; x_curr < x_start + x_size; x_curr++) {
          for (uint32_t y_curr = y_start; y_curr < y_start + y_size; y_curr++) {
            CoreCoord coord = {x_curr, y_curr};

            int arg_index = runtime_arg->argument_index();
            
            if (arg_index + 1 > (int)compute_kernel_attribute.runtime_args_per_core[coord].size()) {
              compute_kernel_attribute.runtime_args_per_core[coord].resize(arg_index + 1);
            }
            
            compute_kernel_attribute.runtime_args_per_core[coord][arg_index] = runtime_arg_val;
          }
        }
      }
      std::cout << "Finished preparing runtime args for compute attributes" << std::endl;
      
      compute_attributes.push_back(compute_kernel_attribute);
    }
    std::cout << "Finished preparing compute attributes" << std::endl;
    
    std::cout << "Preparing data movement attributes" << std::endl; 
    std::vector<::ttnn::operations::generic::data_movement_attributes_t> data_movement_attributes;
    for (auto data_movement_kernel : *op->data_movement_kernels()) {

      // Convert compile args to vector.
      auto compile_args_vec = data_movement_kernel->data_movement_config()->compile_args();
      std::vector<uint32_t> compile_args(compile_args_vec->begin(), compile_args_vec->end());

      // Convert defines to map.
      auto dm_defines = data_movement_kernel->data_movement_config()->defines();
      std::map<std::string, std::string> dm_config_defines;
      for (auto one_define : *dm_defines) {
        dm_config_defines[one_define->key()->str()] = one_define->val()->str();
      }
      
      CoreRangeSet crs = utils::get_core_range_set(data_movement_kernel->core_spec());

      DataMovementConfig data_movement_config;

      if (data_movement_kernel->data_movement_config()->reader_writer() == 0) {
        data_movement_config = tt::tt_metal::ReaderDataMovementConfig(compile_args, dm_config_defines);
      } else {
        data_movement_config = tt::tt_metal::WriterDataMovementConfig(compile_args, dm_config_defines);
      }

      ::ttnn::operations::generic::data_movement_attributes_t dm_attr = {
        .core_spec = crs,
        .kernel_path = data_movement_kernel->kernel_path()->str(),
        .config = data_movement_config,
        .runtime_args_per_core = {}
      };
      
      std::cout << "Preparing runtime arguments for data movement attributes" << std::endl;
      for (auto runtime_arg : *data_movement_kernel->runtime_args()) {
        const bool ttnn_compute = runtime_arg->ttnn_compute();
        unsigned int runtime_arg_val;

        if (ttnn_compute) {
          computeTensorRuntimeArgument(
              runtime_arg->runtime_arg_type(),
              runtime_arg->tensor_glob_id(),
              liveTensors,
              tensorPool,
              *op->ins());
        } else {
          runtime_arg_val = runtime_arg->val();
        }
        
        const uint32_t x_start = runtime_arg->core_range()->loc().x();
        const uint32_t y_start = runtime_arg->core_range()->loc().y();
        const uint32_t x_size = runtime_arg->core_range()->size().x();
        const uint32_t y_size = runtime_arg->core_range()->size().y();

        for (uint32_t x_curr = x_start; x_curr < x_start + x_size; x_curr++) {
          for (uint32_t y_curr = y_start; y_curr < y_start + y_size; y_curr++) {
            CoreCoord coord = {x_curr, y_curr};

            int arg_index = runtime_arg->argument_index();
            
            if (arg_index + 1 > (int)dm_attr.runtime_args_per_core[coord].size()) {
              dm_attr.runtime_args_per_core[coord].resize(arg_index + 1);
            }
            
            dm_attr.runtime_args_per_core[coord][arg_index] = runtime_arg_val;
          }
        }
      }

      std::cout << "Finished preparing runtime args for data movement attributes" << std::endl;

      data_movement_attributes.push_back(dm_attr);
    }

    std::cout << "Finished preparing data movement attributes" << std::endl;

    throw("Reached the end of generic op run function...");
}

// ANCHOR: adding_an_op_matmul_runtime
static void
run(::tt::target::ttnn::MatmulOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  auto &lhs = *liveTensors.at(op->in0()->global_id());
  auto &rhs = *liveTensors.at(op->in1()->global_id());
  tensorPool.push_back(::ttnn::operations::matmul::matmul(
      lhs, rhs, std::nullopt, ::tt::operations::primary::Matmul{}));
  liveTensors.insert_or_assign(op->out()->global_id(), &tensorPool.back());
}
// ANCHOR_END: adding_an_op_matmul_runtime

static void
run(::tt::target::ttnn::Operation const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::OpenDeviceOp: {
    // Skip for now, do we want device externally supplied?
    break;
  }
  case ::tt::target::ttnn::OpType::CloseDeviceOp: {
    // Skip for now, do we want device externally supplied?
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    return run(op->type_as_ToMemoryConfigOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    return run(op->type_as_EmptyOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    // Skip for now, we need an empty op
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    return run(op->type_as_EltwiseOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    return run(op->type_as_MatmulOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    return run(op->type_as_ReductionOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    return run(op->type_as_EmbeddingOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    return run(op->type_as_SoftmaxOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    return run(op->type_as_TransposeOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::GenericOp: {
    return run(op->type_as_GenericOp(), device, liveTensors, tensorPool);
  }
  default:
    throw std::runtime_error("Unsupported operation type");
  }
}

// Nop is single input, output tensor where input is returned as output.
bool handleNopProgram(::tt::target::ttnn::Program const *program,
                      std::vector<::ttnn::Tensor *> const &inputs,
                      std::vector<::ttnn::Tensor *> const &outputs) {

  bool is_nop = program->inputs()->size() == 1 &&
                program->outputs()->size() == 1 &&
                program->inputs()->Get(0)->global_id() ==
                    program->outputs()->Get(0)->global_id();

  if (is_nop) {
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(*inputs.at(0));
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(*outputs.at(0));
    std::uint32_t size = outputs[0]->volume() * outputs[0]->element_size();
    std::memcpy(dst, src, size);
  }
  return is_nop;
}

void runProgram(::ttnn::Device &device,
                ::tt::target::ttnn::Program const *program,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs) {
  std::unordered_map<std::uint32_t, ::ttnn::Tensor *> liveTensors;
  std::list<::ttnn::Tensor> tensorPool;

  int inputIndex = 0;
  assert(program->inputs()->size() == inputs.size() &&
         "Mismatch between program inputs and input tensors");
  bool is_nop = handleNopProgram(program, inputs, outputs);
  for (::tt::target::TensorRef const *input : *program->inputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(input->global_id(), inputs[inputIndex++]);
    assert(inserted && "Duplicate input tensor");
  }

  int outputIndex = 0;
  assert(program->outputs()->size() == outputs.size() &&
         "Mismatch between program outputs and output tensors");
  for (::tt::target::TensorRef const *output : *program->outputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(output->global_id(), outputs[outputIndex++]);
    assert(is_nop || inserted && "Duplicate output tensor");
  }

  for (::tt::target::ttnn::Operation const *op : *program->operations()) {
    run(op, device, liveTensors, tensorPool);
  }
}
} // namespace tt::runtime::ttnn
