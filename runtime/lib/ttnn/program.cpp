// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"
#include "operations/ccl/all_gather.h"
#include "operations/ccl/collective_permute.h"
#include "operations/ccl/mesh_shard.h"
#include "operations/ccl/reduce_scatter.h"
#include "operations/context/get_device.h"
#include "operations/conv/conv2d.h"
#include "operations/conv/conv_transpose2d.h"
#include "operations/conv/prepare_conv2d_weights.h"
#include "operations/cpu/cpu.h"
#include "operations/creation/arange.h"
#include "operations/creation/constant.h"
#include "operations/creation/construct_tensor.h"
#include "operations/creation/empty.h"
#include "operations/creation/full.h"
#include "operations/creation/full_with.h"
#include "operations/data_movement/concat.h"
#include "operations/data_movement/pad.h"
#include "operations/data_movement/permute.h"
#include "operations/data_movement/repeat.h"
#include "operations/data_movement/repeat_interleave.h"
#include "operations/data_movement/reshape.h"
#include "operations/data_movement/slice.h"
#include "operations/data_movement/transpose.h"
#include "operations/deletion/deallocate.h"
#include "operations/eltwise/binary/binary.h"
#include "operations/eltwise/binary/binary_composite.h"
#include "operations/eltwise/quantization/quantization.h"
#include "operations/eltwise/ternary/where.h"
#include "operations/eltwise/unary/unary.h"
#include "operations/eltwise/unary/unary_composite.h"
#include "operations/embedding/embedding.h"
#include "operations/embedding/embedding_backward.h"
#include "operations/kv_cache/fill_cache.h"
#include "operations/kv_cache/update_cache.h"
#include "operations/layout/from_device.h"
#include "operations/layout/to_device.h"
#include "operations/layout/to_dtype.h"
#include "operations/layout/to_layout.h"
#include "operations/layout/to_memory_config.h"
#include "operations/layout/typecast.h"
#include "operations/matmul/matmul.h"
#include "operations/moreh/moreh_cumsum.h"
#include "operations/normalization/softmax.h"
#include "operations/pool/maxpool2d.h"
#include "operations/pool/upsample.h"
#include "operations/reduction/argmax.h"
#include "operations/reduction/prod.h"
#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/dylib.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/program_executor.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

#ifdef TT_RUNTIME_ENABLE_PERF_TRACE
#include "tracy/Tracy.hpp"
#endif

namespace tt::runtime::ttnn {

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

std::vector<Tensor> runProgram(::ttnn::MeshDevice &meshDevice,
                               Binary executableHandle,
                               std::uint32_t programIndex,
                               std::vector<::tt::runtime::Tensor> &inputs,
                               std::shared_ptr<TensorCache> externalCache) {
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  ::tt::target::ttnn::Program const *program =
      fbb.programs()->Get(programIndex);
  ProgramExecutor executor(program, executableHandle, inputs, &meshDevice,
                           externalCache);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputTensors =
      executor.gatherOutputTensors();
  return outputTensors;
}

} // namespace tt::runtime::ttnn
