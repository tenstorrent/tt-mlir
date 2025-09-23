// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_CUDA_TTCUDA_H
#define TT_RUNTIME_DETAIL_CUDA_TTCUDA_H

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/CUDA/program_generated.h"
#pragma clang diagnostic pop
#include <cuda.h>
#include <cuda_runtime.h>

namespace tt::runtime::cuda {

struct CudaTensorHandle {
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> stride;
  ::tt::target::DataType dataType;
  std::uint32_t itemsize;
};

struct CudaDeviceHandle {
  CUdevice device;
  CUcontext context;
};

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs);
::tt::runtime::Tensor
createBorrowedHostTensor(void *data, const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType);
std::vector<::tt::runtime::Tensor> toHost(::tt::runtime::Tensor tensor,
                                          bool untilize = false,
                                          bool blocking = true);
::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType);
std::vector<std::uint32_t> getTensorShape(Tensor tensor);
std::vector<std::uint32_t> getTensorStride(Tensor tensor);

size_t getNumAvailableDevices();

Device openMeshDevice(const MeshDeviceOptions &options = {});

void closeMeshDevice(Device parentMesh);

void memcpy(void *dst, Tensor src,
            std::optional<tt::target::DataType> targetDataType = std::nullopt);

void memcpy(Tensor dst, Tensor src);

void deallocateTensor(Tensor &tensor, bool force = false);

} // namespace tt::runtime::cuda

#endif
