// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_DISTRIBUTED_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_DISTRIBUTED_H

#include "tt/runtime/types.h"

namespace tt::runtime::distributed {

void launchDistributedRuntime(const DistributedOptions &options = {});
void shutdownDistributedRuntime();

SystemDesc getCurrentSystemDesc(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType =
        std::nullopt,
    std::optional<::tt::runtime::Device> deviceHandle = std::nullopt);

::tt::runtime::Device openMeshDevice(const MeshDeviceOptions &options = {});

void closeMeshDevice(::tt::runtime::Device parentMesh);

::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType);

::tt::runtime::Layout getLayout(::tt::runtime::Binary executableHandle,
                                std::uint32_t programIndex,
                                std::uint32_t inputIndex);

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor,
                               ::tt::runtime::Device device,
                               ::tt::runtime::Layout layout,
                               std::optional<bool> retain = std::nullopt);

std::vector<::tt::runtime::Tensor>
submit(::tt::runtime::Device deviceHandle,
       ::tt::runtime::Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs);

std::vector<::tt::runtime::Tensor>
toHost(const ::tt::runtime::Tensor &tensorHandle, bool untilize = false,
       bool blocking = true);

void memcpy(void *dst, const ::tt::runtime::Tensor &srcHandle,
            std::optional<tt::target::DataType> targetDataType = std::nullopt);

} // namespace tt::runtime::distributed

#endif
