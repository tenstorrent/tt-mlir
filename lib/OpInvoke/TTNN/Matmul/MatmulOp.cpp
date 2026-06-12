// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Matmul/MatmulOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"

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
resolveMatmulParams(const ::tt::target::ttnn::MatmulOpT &matmulOp) {

  MatmulResolvedParams params;

  if (matmulOp.out) {
    params.outputDType = operations::utils::getDataType(*matmulOp.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(matmulOp);
  if (!matmulOp.activation.empty() &&
      !programCarriesFusedActivation(params.matmulProgramConfig)) {
    params.activation = std::make_optional(matmulOp.activation);
  }
  if (matmulOp.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *matmulOp.compute_config);
  }

  if (matmulOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*matmulOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*matmulOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createMatmulTuple(Tag tag, const ::tt::target::ttnn::MatmulOpT &matmulOp,
                       TensorArg lhs, TensorArg rhs,
                       const MatmulResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(lhs, tag), resolveTensorArg(rhs, tag),
      matmulOp.transpose_a, matmulOp.transpose_b, params.outputMemoryConfig,
      params.outputDType, params.matmulProgramConfig, params.activation,
      params.computeConfig, /*core_grid=*/std::nullopt,
      /*output_tile=*/std::nullopt, /*optional_output_tensor=*/std::nullopt,
      /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
}

MatmulOpResult callMatmul(CallType callType,
                          const ::tt::target::ttnn::MatmulOpT &matmulOp,
                          TensorArg lhs, TensorArg rhs,
                          ::ttnn::MeshDevice *device) {

  MatmulResolvedParams params = resolveMatmulParams(matmulOp);

  auto makeTuple = [&](auto tag) {
    return createMatmulTuple(tag, matmulOp, lhs, rhs, params);
  };

  return callOp<MatmulOpResult>(WRAP_OP(::ttnn::matmul), callType, makeTuple,
                                device);
}

LinearResolvedParams
resolveLinearParams(const ::tt::target::ttnn::LinearOpT &linearOp) {

  LinearResolvedParams params;

  if (linearOp.out) {
    params.outputDType = operations::utils::getDataType(*linearOp.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(linearOp);
  if (!linearOp.activation.empty() &&
      !programCarriesFusedActivation(params.matmulProgramConfig)) {
    params.activation = std::make_optional(linearOp.activation);
  }
  if (linearOp.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *linearOp.compute_config);
  }

  if (linearOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*linearOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*linearOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createLinearTuple(Tag tag, const ::tt::target::ttnn::LinearOpT &linearOp,
                       TensorArg a, TensorArg b,
                       const std::optional<TensorArg> bias,
                       const LinearResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(a, tag), resolveTensorArg(b, tag),
      bias.has_value()
          ? std::make_optional(*std::get<const ::ttnn::Tensor *>(bias.value()))
          : std::nullopt,
      linearOp.transpose_a, linearOp.transpose_b, params.outputMemoryConfig,
      params.outputDType, params.matmulProgramConfig, params.activation,
      params.computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
      /*optional_output_tensor=*/std::nullopt,
      /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
}

LinearOpResult callLinear(CallType callType,
                          const ::tt::target::ttnn::LinearOpT &linearOp,
                          TensorArg a, TensorArg b,
                          const std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device) {
  LinearResolvedParams params = resolveLinearParams(linearOp);

  auto makeTuple = [&](auto tag) {
    return createLinearTuple(tag, linearOp, a, b, bias, params);
  };

  return callOp<LinearOpResult>(WRAP_OP(::ttnn::linear), callType, makeTuple,
                                device);
}

SparseMatmulResolvedParams resolveSparseMatmulParams(
    const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOp) {

  SparseMatmulResolvedParams params;

  if (sparseMatmulOp.nnz) {
    params.nnz = std::make_optional(static_cast<uint32_t>(*sparseMatmulOp.nnz));
  }

  auto matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(sparseMatmulOp);
  TT_INVOKE_ASSERT(
      matmulProgramConfig.has_value(),
      "SparseMatmulOp requires program_config to be set at compile "
      "time");
  params.matmulProgramConfig = matmulProgramConfig.value();

  if (sparseMatmulOp.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *sparseMatmulOp.compute_config);
  }

  if (sparseMatmulOp.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*sparseMatmulOp.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*sparseMatmulOp.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createSparseMatmulTuple(
    Tag tag, const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOp,
    TensorArg a, TensorArg b, TensorArg sparsity,
    const SparseMatmulResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(a, tag), resolveTensorArg(b, tag),
      resolveTensorArg(sparsity, tag), params.matmulProgramConfig, params.nnz,
      sparseMatmulOp.is_input_a_sparse, sparseMatmulOp.is_input_b_sparse,
      params.outputMemoryConfig, /*dtype=*/std::nullopt, params.computeConfig,
      /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt);
}

SparseMatmulOpResult
callSparseMatmul(CallType callType,
                 const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOp,
                 TensorArg a, TensorArg b, TensorArg sparsity,
                 ::ttnn::MeshDevice *device) {
  SparseMatmulResolvedParams params = resolveSparseMatmulParams(sparseMatmulOp);

  auto makeTuple = [&](auto tag) {
    return createSparseMatmulTuple(tag, sparseMatmulOp, a, b, sparsity, params);
  };

  return callOp<SparseMatmulOpResult, false, false>(
      WRAP_OP(::ttnn::sparse_matmul), callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
