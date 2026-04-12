// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_ARGUMENTS_H
#define RUNTIME_LIB_TTMETAL_ARGUMENTS_H

#include "executor_utils.h"

#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/TTMetal/types_generated.h"

#include "tt-metalium/tensor_accessor_args.hpp"

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

template <bool isCompileTime>
std::vector<std::uint32_t> processKernelArgs(
    const flatbuffers::Vector<flatbuffers::Offset<target::metal::KernelArg>>
        *args,
    const flatbuffers::Vector<flatbuffers::Offset<void>> *argRefs,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::CBRef>>
        *cbs,
    const std::unordered_map<std::uint32_t, Tensor> &hostBuffers,
    const std::unordered_map<
        std::uint32_t, std::shared_ptr<distributed::MeshBuffer>> &meshBuffers,
    const std::unordered_map<std::uint32_t, tt_metal::GlobalSemaphore>
        &globalSemaphoresCache,
    const std::unordered_map<std::uint32_t, std::uint32_t>
        &localSemaphoresCache,
    const DeviceAddressValidator &deviceAddressValidator,
    std::function<std::uint32_t(std::uint32_t)> createSemaphoreFn) {

  // Goal of this function is to convert the kernel args into a vector of
  // uint32_t. This vector will be passed to the kernel at runtime.
  std::vector<std::uint32_t> argsVec;
  if (args == nullptr || args->size() == 0) {
    return argsVec;
  }

  // Iterate through all args and process them based on their type.
  argsVec.reserve(args->size());
  for (uint32_t argIdx = 0; argIdx < args->size(); ++argIdx) {
    const auto *kernelArg = args->Get(argIdx);

    switch (kernelArg->arg_type()) {

    // For CB arg, we want the port number.
    case target::metal::KernelArgType::KernelArgCBPort: {
      const auto *arg = kernelArg->arg_as_KernelArgCBPort();
      auto operand_idx = arg->operand_idx();

      // CB operand index can be:
      // -1: signify scratchpad CB
      // >=0: index into the enqueue program command cb list
      LOG_ASSERT(operand_idx >= -1,
                 "Invalid operand index for CB arg: ", operand_idx);

      if (operand_idx == -1) {
        // We need to update this to support scratch pad CBs.
        LOG_ASSERT(false, "Runtime currently does not support scratchpad CBs.");
      } else {
        LOG_ASSERT(operand_idx < static_cast<int32_t>(cbs->size()),
                   "Invalid operand index: ", operand_idx);
        argsVec.push_back(cbs->Get(operand_idx)->port());
      }

      break;
    }

    // For buffer address arg, we want the device address. We also need to
    // validate that the buffer is still alive.
    case target::metal::KernelArgType::KernelArgBufferAddress: {
      const auto *arg = kernelArg->arg_as_KernelArgBufferAddress();
      const target::metal::BufferRef *buffer =
          reinterpret_cast<const target::metal::BufferRef *>(
              argRefs->Get(arg->operand_idx()));
      LOG_ASSERT(meshBuffers.find(buffer->global_id()) != meshBuffers.end(),
                 "Buffer id referenced by rt args is no longer alive or was "
                 "never created ",
                 logger::Buffer(buffer->global_id()));

      const target::metal::BufferDesc *bufferDesc = buffer->desc();
      LOG_ASSERT(bufferDesc->buffer_detail_type() ==
                 target::metal::BufferDetail::MetalBuffer);
      const target::metal::MetalBuffer *metalBuffer =
          bufferDesc->buffer_detail_as_MetalBuffer();
      argsVec.push_back(deviceAddressValidator(buffer->address(),
                                               metalBuffer->buffer_type()));
      break;
    }

    // For local semaphore arg, look up the LocalSemaphoreRef via operand_idx to
    // get the initial_value, then create a new semaphore with that value.
    case target::metal::KernelArgType::KernelArgLocalSemaphore: {
      LOG_ASSERT(createSemaphoreFn, "createSemaphoreFn is not set");
      const auto *arg = kernelArg->arg_as_KernelArgLocalSemaphore();
      const tt::target::metal::LocalSemaphoreRef *local_sem_ref =
          reinterpret_cast<const target::metal::LocalSemaphoreRef *>(
              argRefs->Get(arg->operand_idx()));
      LOG_ASSERT(localSemaphoresCache.find(local_sem_ref->global_id()) !=
                     localSemaphoresCache.end(),
                 "Local semaphore id referenced by rt args is no longer alive "
                 "or was never created ",
                 logger::Buffer(local_sem_ref->global_id()));
      argsVec.push_back(createSemaphoreFn(
          localSemaphoresCache.at(local_sem_ref->global_id())));

      break;
    }

    // For named argument, we just need to pass the value specified in the arg.
    case target::metal::KernelArgType::KernelArgNamedArgument: {
      const auto *arg = kernelArg->arg_as_KernelArgNamedArgument();
      argsVec.push_back(arg->value());
      break;
    }

    // For global semaphore arg, we need to look up the global semaphore cache
    // with the global id specified in the arg, validate that it's still alive,
    // and pass its address to the kernel.
    case target::metal::KernelArgType::KernelArgGlobalSemaphore: {
      const auto *arg = kernelArg->arg_as_KernelArgGlobalSemaphore();
      const tt::target::metal::GlobalSemaphoreRef *global_semaphore_operand =
          reinterpret_cast<const target::metal::GlobalSemaphoreRef *>(
              argRefs->Get(arg->operand_idx()));
      LOG_ASSERT(
          globalSemaphoresCache.find(global_semaphore_operand->global_id()) !=
              globalSemaphoresCache.end(),
          "Global semaphore id referenced by rt args is no longer alive or was "
          "never created ",
          logger::Buffer(global_semaphore_operand->global_id()));

      argsVec.push_back(deviceAddressValidator(
          globalSemaphoresCache.at(global_semaphore_operand->global_id())
              .address(),
          target::BufferType::L1));
      break;
    }

    // For scalar arg, we look up the scalar value from the global index and
    // pass it to the kernel.
    case target::metal::KernelArgType::KernelArgScalar: {
      const auto *arg = kernelArg->arg_as_KernelArgScalar();
      const tt::target::metal::BufferRef *buffer =
          reinterpret_cast<const target::metal::BufferRef *>(
              argRefs->Get(arg->operand_idx()));
      LOG_ASSERT(hostBuffers.find(buffer->global_id()) != hostBuffers.end(),
                 "Scalar id is no longer alive or was never created ",
                 logger::Buffer(buffer->global_id()));
      const MetalTensor &metalTensor =
          hostBuffers.at(buffer->global_id())
              .as<MetalTensor>(DeviceRuntime::TTMetal);
      std::uint32_t scalarValue = std::get<std::uint32_t>(metalTensor);
      argsVec.push_back(scalarValue);
      break;
    }

    // For TensorAccessor arg, we construct TensorAccessorArgs from the
    // referenced buffer and append its compile-time or runtime args.
    // Unlike other arg types, this pushes multiple uint32_t values.
    case target::metal::KernelArgType::KernelArgTensorAccessor: {
      const auto *arg = kernelArg->arg_as_KernelArgTensorAccessor();
      const target::metal::BufferRef *buffer =
          reinterpret_cast<const target::metal::BufferRef *>(
              argRefs->Get(arg->operand_idx()));
      LOG_ASSERT(meshBuffers.find(buffer->global_id()) != meshBuffers.end(),
                 "Buffer id referenced by TensorAccessor arg is no longer "
                 "alive or was never created ",
                 logger::Buffer(buffer->global_id()));

      auto meshBuffer = meshBuffers.at(buffer->global_id());
      tt_metal::TensorAccessorArgs tensorAccessorArgs(*meshBuffer);

      if constexpr (isCompileTime) {
        auto ctArgs = tensorAccessorArgs.get_compile_time_args();
        for (auto a : ctArgs) {
          LOG_TRACE(logger::LogRuntimeTTMetalKernelArg, "asdf a: ", a);
        }
        argsVec.insert(argsVec.end(), ctArgs.begin(), ctArgs.end());

        LOG_ASSERT(argsVec.size() <= args->size(), "Not enough TensorAccessor argument slots reverved in kernel argument spec");
        for (size_t i = 1; i < ctArgs.size(); ++i) {
          LOG_ASSERT(args->Get(argIdx + i)->arg_type() == target::metal::KernelArgType::KernelArgReserved, "Not enough TensorAccessor argument slots reverved in kernel argument spec");
        }
      } else {
        auto rtArgs = tensorAccessorArgs.get_common_runtime_args();
        argsVec.insert(argsVec.end(), rtArgs.begin(), rtArgs.end());

        LOG_ASSERT(argsVec.size() <= args->size(), "Not enough TensorAccessor argument slots reverved in kernel argument spec");
        for (size_t i = 1; i < rtArgs.size(); ++i) {
          LOG_ASSERT(args->Get(argIdx + i)->arg_type() == target::metal::KernelArgType::KernelArgReserved, "Not enough TensorAccessor argument slots reverved in kernel argument spec");
        }
      }
      break;
    }

    case target::metal::KernelArgType::KernelArgTensorStride: {
      const auto *arg = kernelArg->arg_as_KernelArgTensorStride();
      const target::metal::BufferRef *buffer =
          reinterpret_cast<const target::metal::BufferRef *>(
              argRefs->Get(arg->operand_idx()));
      LOG_ASSERT(meshBuffers.find(buffer->global_id()) != meshBuffers.end(),
                 "Buffer id referenced by TensorStride arg is no longer "
                 "alive or was never created ",
                 logger::Buffer(buffer->global_id()));

      auto meshBuffer = meshBuffers.at(buffer->global_id());
      auto physicalShardShape = meshBuffer->physical_shard_shape();
      std::array<std::uint32_t, 2> stride = {static_cast<std::uint32_t>(physicalShardShape.width()), static_cast<std::uint32_t>(1)};
      argsVec.insert(argsVec.end(), stride.begin(), stride.end());

      LOG_ASSERT(argsVec.size() <= args->size(), "Not enough TensorStride argument slots reverved in kernel argument spec");
      for (size_t i = 1; i < stride.size(); ++i) {
        LOG_ASSERT(args->Get(argIdx + i)->arg_type() == target::metal::KernelArgType::KernelArgReserved, "Not enough TensorStride argument slots reverved in kernel argument spec");
      }
      break;
    }

    case target::metal::KernelArgType::KernelArgReserved: {
      // Some Reserved users, like TensorAccessor, conservatively reserve space for runtime dependent slots. Just zero fill slots that ended up not being used.
      if (argIdx == argsVec.size()) {
        argsVec.push_back(0);
      }
      break;
    }

    case target::metal::KernelArgType::NONE:
      LOG_FATAL("Unsupported runtime arg type");
    }
    LOG_TRACE(logger::LogRuntimeTTMetalKernelArg,
              isCompileTime ? "Compile" : "Runtime", " arg[",
              argsVec.size() - 1, "] = ", argsVec.back(), " [",
              target::metal::EnumNameKernelArgType(kernelArg->arg_type()), "]");
  }

  return argsVec;
}

template <typename... Args>
std::vector<std::uint32_t> processRuntimeArgs(Args... args) {
  return processKernelArgs<false>(args...);
}

template <typename... Args>
std::vector<std::uint32_t> processCompileArgs(Args... args) {
  return processKernelArgs<true>(args...);
}

} // namespace tt::runtime::ttmetal

#endif // RUNTIME_LIB_TTMETAL_ARGUMENTS_H
