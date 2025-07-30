// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/rand/rand.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttmlir/Target/TTNN/operations/rand_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include <optional>
#include <vector>

namespace tt::runtime::ttnn::operations::rand {
void run(const ::tt::target::ttnn::RandOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Shape size =
      ::tt::runtime::ttnn::operations::utils::toTTNNShape(*op->size());
  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();
  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());
  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());

  ::ttnn::MemoryConfig memoryConfig =
      ::ttnn::DRAM_MEMORY_CONFIG; // Default in rand implementation
  if (op->memcfg()) {
    memoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg())
            .value();
  }
  float low = op->low();
  float high = op->high();
  uint32_t seed = op->seed();

  ::ttnn::Tensor out = ::ttnn::rand(size, targetDevice, dtype, layout,
                                    memoryConfig, low, high, seed);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::rand
