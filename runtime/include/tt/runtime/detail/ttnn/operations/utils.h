// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UTILS_H
#define TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UTILS_H

#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "types_generated.h"
#include <concepts>
#include <cstdint>

namespace tt::runtime::ttnn::operations::utils {

// True when the current device is Quasar ("metal 2.0").
//
// Quasar reimplements the op stack under
// ttnn::operations::experimental::quasar. The mainline ops do not merely
// perform badly there, they are refused: their program factories construct
// DataMovementKernel / ComputeKernel, whose constructors TT_FATAL on Quasar
// ("... is not supported on Quasar. Use Quasar*Kernel instead.",
// tt_metal/impl/kernels/kernel.hpp). Op runners therefore have to select the
// Quasar entry point explicitly. Where the Quasar op takes the same arguments
// this is a straight substitution; where it does not, the op has no Quasar
// path yet and should fail loudly rather than silently mis-dispatch.
inline bool isQuasar() {
  return ::tt::runtime::ttnn::getArch() == ::tt::target::Arch::Quasar;
}

// Quasar: ROW_MAJOR -> TILE for a tensor whose trailing dims are not tile
// aligned, routed through host logical data.
//
// The device conversion corrupts such tensors. Measured on the ZeBu emulator,
// tilizing [1, 32, 16, 8] returns data a full 1.0 of scale away from its input,
// while every neighbouring step -- reshape, transpose, from_device, the host
// untilize -- is exact. craq-sim converts the same tensor correctly, which is
// why none of it shows up on the simulator.
//
// Tensor::to_vector returns logical row-major order whatever the physical layout
// is and from_vector tilizes on the host, so this route cannot be affected by
// the device conversion. Aligned tensors keep the device path: a host round trip
// shifts the allocation pattern, and that alone has been enough to hang a later
// op.
//
// Callers pass the tensor the device conversion already produced, and its spec
// is reused rather than rebuilt -- constructing a fresh TensorSpec is what broke
// small-spatial convolutions on craq-sim (trailing dims 2x2 and 4x4 fell from
// 0.99999 to ~0.5), because from_vector's own spec need not match what the rest
// of the graph was compiled against.
// True when TTMLIR_OP_TRACE is set. Bring-up on the emulator costs ~30 s per
// program launch, so a hang has to be attributed on the first run.
bool opTraceEnabled();

bool quasarTilizeNeedsHostRoute(const ::ttnn::Tensor &input,
                                ::ttnn::Layout targetLayout,
                                std::optional<::ttnn::DataType> dtype);

::ttnn::Tensor rebuildFromHostData(const ::ttnn::Tensor &input,
                                   const ::ttnn::Tensor &deviceResult);

void eventSync(::ttnn::MeshDevice *meshDevice, const ::ttnn::QueueId &recordCq,
               const ::ttnn::QueueId &waitCq);

bool isTilized(const ::tt::target::ttnn::TensorRef *tensorRef);

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRef *tensorRef);

::ttnn::operations::unary::UnaryOpType
toTTNNUnaryOpType(::tt::target::ttnn::EltwiseUnaryOpType unaryOpType);

::ttnn::operations::unary::UnaryWithParam
toTTNNUnaryWithParam(const ::tt::target::ttnn::UnaryWithParam &unaryWithParam);

std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
createMatmulProgramConfigIfNeeded(const ::tt::target::ttnn::MatmulOp *op);

std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
createMatmulProgramConfigIfNeeded(const ::tt::target::ttnn::LinearOp *op);

::ttnn::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *memcfg);

::ttnn::Conv2dSliceConfig
createConv2dSliceConfig(const ::tt::target::ttnn::Conv2dSliceConfig *config);

::ttnn::operations::transformer::SDPAProgramConfig
createSDPAProgramConfig(const ::tt::target::ttnn::SDPAConfig *config);

::ttnn::DeviceComputeKernelConfig createDeviceComputeKernelConfig(
    const ::tt::target::ttnn::DeviceComputeKernelConfig *config);

::ttnn::prim::LayerNormProgramConfig
createLayerNormShardedMultiCoreProgramConfig(
    const ::tt::target::ttnn::LayerNormShardedMultiCoreProgramConfig *config);

::ttnn::Tensor toTTNNTensor(const ::flatbuffers::Vector<uint8_t> *input,
                            const ::ttnn::DataType &inputDataType,
                            const ::ttnn::Shape &shape,
                            const ::ttnn::DataType &outputDataType,
                            ::ttnn::MeshDevice *meshDevice,
                            const ::ttnn::Layout &layout,
                            const ::ttnn::MemoryConfig &memoryConfig);

::ttnn::Tensor
allocateTensorOnDevice(const ::tt::target::ttnn::TensorRef *tensorRef,
                       ::ttnn::MeshDevice &meshDevice);

std::vector<::tt::runtime::GlobalSemaphore> collectSemaphoreInputs(
    const ::flatbuffers::Vector<
        ::flatbuffers::Offset<::tt::target::ttnn::GlobalSemaphoreRef>>
        *semaphoreInputs,
    ProgramContext &context);

template <std::integral T>
inline ::ttnn::Shape toTTNNShape(const flatbuffers::Vector<T> &vec) {
  std::vector<uint32_t> rawShape;
  rawShape.reserve(vec.size());
  std::transform(
      vec.begin(), vec.end(), std::back_inserter(rawShape),
      [](const T &x) -> uint32_t { return static_cast<uint32_t>(x); });
  return ::ttnn::Shape(rawShape);
}

} // namespace tt::runtime::ttnn::operations::utils
#endif
