#include <assert.h>
#include <iostream>

#include "mlir_interface_api.hpp"

void checker(bool condition, std::string message = "") {
  std::cout << message << " ";
  if (!condition) {
    std::cout << "failed\n";
  } else {
    std::cout << "passed\n";
  }
  assert(condition);
}

int main() {
  std::cout << "Hello world!\n";

  // binary
  {
    std::vector<uint32_t> shape_a = {2, 1, 32, 32};
    std::vector<uint32_t> shape_b = {1, 1, 32, 32};
    ttnn::mlir_interface::memory_config_tuple memory_config = {
        "interleaved", "dram", std::nullopt};
    std::string data_type = "bf16";

    checker(true == ttnn::mlir_interface::
                        does_binary_op_support_input_output_constraints(
                            shape_a, memory_config, data_type, shape_b,
                            memory_config, data_type, memory_config, data_type),
            "binary");
  }
  // binary sharded
  {
    std::vector<uint32_t> shape = {1, 1, 32, 32 * 64 * 5};
    ttnn::mlir_interface::shard_spec_tuple shard_spec = {
        {{0, 0, 7, 7}}, {32, 32 * 5}, "col_major", false};
    ttnn::mlir_interface::memory_config_tuple memory_config = {
        "width_sharded", "l1", shard_spec};
    std::string data_type = "bf16";

    checker(
        ttnn::mlir_interface::does_binary_op_support_input_output_constraints(
            shape, memory_config, data_type, shape, memory_config, data_type,
            memory_config, data_type),
        "binary sharded");
  }
  // unary
  {
    std::vector<uint32_t> shape = {1, 1, 32, 32 * 5 * 64};
    ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {
        "interleaved", "l1", std::nullopt};
    std::string data_type = "bf16";

    checker(
        ttnn::mlir_interface::does_unary_op_support_input_output_constraints(
            "RELU", shape, l1_interleaved_memory_config, data_type, shape,
            l1_interleaved_memory_config, data_type),
        "unary");
  }

  // softmax
  {
    std::vector<uint32_t> shape = {1, 1, 32, 32 * 5 * 64};
    ttnn::mlir_interface::memory_config_tuple l1_interleaved_memory_config = {
        "interleaved", "l1", std::nullopt};
    std::string data_type = "bf16";

    checker(
        ttnn::mlir_interface::does_softmax_op_support_input_output_constraints(
            shape, l1_interleaved_memory_config, data_type, shape,
            l1_interleaved_memory_config, data_type),
        "softmax");
  }
  return 0;
}