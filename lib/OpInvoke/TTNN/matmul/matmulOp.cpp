// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/matmul/matmulOp.h"
#include "operations/functions.hpp"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>
#include <variant>

namespace ttnn_op_invoke {

// Returns true if the matmul program config already carries a fused
// activation.
static bool programCarriesFusedActivation(
    const std::optional<::ttnn::operations::matmul::MatmulProgramConfig> &pc) {
  if (!pc) {
    return false;
  }
  return std::visit(
      [](const auto &cfg) -> bool {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (
            std::is_same_v<T, ::ttnn::operations::matmul::
                                  MatmulMultiCoreReuseMultiCastProgramConfig> ||
            std::is_same_v<T,
                           ::ttnn::operations::matmul::
                               MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
            std::is_same_v<
                T, ::ttnn::operations::matmul::
                       MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> ||
            std::is_same_v<
                T,
                ::ttnn::operations::matmul::
                    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
          return cfg.fused_activation.has_value();
        }
        return false;
      },
      *pc);
}

MatmulResolvedParams
resolveMatmulParams(const ::tt::target::ttnn::MatmulOpT &matmulOpT,
                    CallType callType) {

  MatmulResolvedParams params;

  if (matmulOpT.out) {
    params.outputDType = operations::utils::getDataType(*matmulOpT.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(matmulOpT);
  if (!matmulOpT.activation.empty() &&
      !programCarriesFusedActivation(params.matmulProgramConfig)) {
    params.activation = std::make_optional(matmulOpT.activation);
  }
  if (matmulOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *matmulOpT.compute_config);
  }

  if (matmulOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*matmulOpT.out), callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*matmulOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createMatmulTuple(Tag tag, const ::tt::target::ttnn::MatmulOpT &matmulOpT,
                       TensorArg lhs, TensorArg rhs, ::ttnn::MeshDevice *device,
                       const MatmulResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(lhs, tag), resolveTensorArg(rhs, tag),
      matmulOpT.transpose_a, matmulOpT.transpose_b, params.outputMemoryConfig,
      params.outputDType, params.matmulProgramConfig, params.activation,
      params.computeConfig, /*core_grid=*/std::nullopt,
      /*output_tile=*/std::nullopt, /*optional_output_tensor=*/std::nullopt,
      /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
}

MatmulOpResult callMatmul(CallType callType,
                          const ::tt::target::ttnn::MatmulOpT &matmulOpT,
                          TensorArg lhs, TensorArg rhs,
                          ::ttnn::MeshDevice *device) {

  MatmulResolvedParams params = resolveMatmulParams(matmulOpT, callType);

  auto makeTuple = [&](auto tag) {
    return createMatmulTuple(tag, matmulOpT, lhs, rhs, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(::ttnn::matmul, device,
                                      std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::matmul, device,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::matmul(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

LinearResolvedParams
resolveLinearParams(const ::tt::target::ttnn::LinearOpT &linearOpT,
                    CallType callType) {

  LinearResolvedParams params;

  if (linearOpT.out) {
    params.outputDType = operations::utils::getDataType(*linearOpT.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(linearOpT);
  if (!linearOpT.activation.empty() &&
      !programCarriesFusedActivation(params.matmulProgramConfig)) {
    params.activation = std::make_optional(linearOpT.activation);
  }
  if (linearOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *linearOpT.compute_config);
  }

  if (linearOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*linearOpT.out), callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*linearOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createLinearTuple(Tag tag, const ::tt::target::ttnn::LinearOpT &linearOpT,
                       TensorArg a, TensorArg b,
                       const std::optional<TensorArg> bias,
                       ::ttnn::MeshDevice *device,
                       const LinearResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(a, tag), resolveTensorArg(b, tag),
      bias.has_value()
          ? std::make_optional(*std::get<const ::ttnn::Tensor *>(bias.value()))
          : std::nullopt,
      linearOpT.transpose_a, linearOpT.transpose_b, params.outputMemoryConfig,
      params.outputDType, params.matmulProgramConfig, params.activation,
      params.computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /*optional_output_tensor=*/std::nullopt,
      /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
}

LinearOpResult callLinear(CallType callType,
                          const ::tt::target::ttnn::LinearOpT &linearOpT,
                          TensorArg a, TensorArg b,
                          const std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device) {
  LinearResolvedParams params = resolveLinearParams(linearOpT, callType);

  auto makeTuple = [&](auto tag) {
    return createLinearTuple(tag, linearOpT, a, b, bias, device, params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_CONSTRAINTS(::ttnn::linear, device,
                                      std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::QUERY_OP_RUNTIME:
    return std::apply(
        [&](auto &&...args) {
          return QUERY_OP_RUNTIME(::ttnn::linear, device,
                                  std::forward<decltype(args)>(args)...);
        },
        makeTuple(QueryTag{}));
  case CallType::EXECUTE:
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::linear(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  llvm_unreachable("unhandled CallType");
}

SparseMatmulResolvedParams resolveSparseMatmulParams(
    const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT,
    CallType callType) {

  SparseMatmulResolvedParams params;

  if (sparseMatmulOpT.nnz) {
    params.nnz =
        std::make_optional(static_cast<uint32_t>(*sparseMatmulOpT.nnz));
  }

  auto matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(sparseMatmulOpT);
  LOG_ASSERT(matmulProgramConfig.has_value(),
             "SparseMatmulOp requires program_config to be set at compile "
             "time");
  params.matmulProgramConfig = matmulProgramConfig.value();

  if (sparseMatmulOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *sparseMatmulOpT.compute_config);
  }

  if (sparseMatmulOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*sparseMatmulOpT.out),
        callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*sparseMatmulOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createSparseMatmulTuple(
    Tag tag, const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT,
    TensorArg a, TensorArg b, TensorArg sparsity, ::ttnn::MeshDevice *device,
    const SparseMatmulResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(a, tag), resolveTensorArg(b, tag),
      resolveTensorArg(sparsity, tag), params.matmulProgramConfig, params.nnz,
      sparseMatmulOpT.is_input_a_sparse, sparseMatmulOpT.is_input_b_sparse,
      params.outputMemoryConfig, /*dtype=*/std::nullopt, params.computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt);
}

SparseMatmulOpResult
callSparseMatmul(CallType callType,
                 const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT,
                 TensorArg a, TensorArg b, TensorArg sparsity,
                 ::ttnn::MeshDevice *device) {
  SparseMatmulResolvedParams params =
      resolveSparseMatmulParams(sparseMatmulOpT, callType);

  auto makeTuple = [&](auto tag) {
    return createSparseMatmulTuple(tag, sparseMatmulOpT, a, b, sparsity, device,
                                   params);
  };

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS: {
    ::ttnn::graph::ConstraintQueryResponse response;
    response.error_message =
        "Constraint query not implemented for SparseMatmulOp yet";
    return response;
  }
  case CallType::QUERY_OP_RUNTIME: {
    ::ttnn::graph::RuntimeQueryResponse response;
    response.error_message =
        "Runtime query not implemented for SparseMatmulOp yet";
    return response;
  }
  case CallType::EXECUTE: {
    return std::apply(
        [&](auto &&...args) {
          return ::ttnn::sparse_matmul(std::forward<decltype(args)>(args)...);
        },
        makeTuple(ExecuteTag{}));
  }
  }
  llvm_unreachable("unhandled CallType");
}

} // namespace ttnn_op_invoke
