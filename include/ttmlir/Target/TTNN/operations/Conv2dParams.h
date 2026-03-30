// // SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
// //
// // SPDX-License-Identifier: Apache-2.0

// #ifndef TTMLIR_TARGET_TTNN_OPERATIONS_CONV2DPARAMS_H
// #define TTMLIR_TARGET_TTNN_OPERATIONS_CONV2DPARAMS_H

// #include "ttnn/graph/graph_query_op_constraints.hpp"
// #include "ttnn/graph/graph_query_op_runtime.hpp"
// #include "ttnn/operations/conv/conv2d/conv2d.hpp"
// #include "ttnn/tensor/tensor.hpp"
// #include "ttnn/tensor/tensor_spec.hpp"
// #include "ttnn/types.hpp"

// #include <array>
// #include <optional>
// #include <variant>

// namespace tt::target::ttnn::conv2d {

// using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;
// using Conv2dOpResult = std::variant<::ttnn::graph::ConstraintQueryResponse,
//                                     ::ttnn::graph::RuntimeQueryResponse,
//                                     ::ttnn::Conv2dResultWithOptions>;

// struct Conv2dResolvedParams {
//   std::array<uint32_t, 2> kernelSize;
//   std::array<uint32_t, 2> stride;
//   std::array<uint32_t, 2> dilation;
//   std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
//   std::optional<::ttnn::DataType> outputDtype;
//   std::optional<::ttnn::Conv2dConfig> conv2dConfig;
//   std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
//   std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
//   std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
// };

// } // namespace tt::target::ttnn::conv2d

// #endif // TTMLIR_TARGET_TTNN_OPERATIONS_CONV2DPARAMS_H
