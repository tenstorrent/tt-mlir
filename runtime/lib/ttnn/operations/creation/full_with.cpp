// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/full_with.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::creation {
static void runFullWithOp(const ::tt::target::ttnn::NamedFullOp *op,
                          ProgramContext &context, auto &&ttnnOp) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Shape shape(utils::toTTNNShape(*op->shape()));

  std::optional<::ttnn::DataType> dtype;
  std::optional<::ttnn::Layout> layout;
  std::optional<std::reference_wrapper<::ttnn::IDevice>> device;
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memcfg()
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg())
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(
                    op->out()));

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  if (op->layout()) {
    layout = ::tt::runtime::ttnn::utils::toTTNNLayout(*(op->layout()));
  }

  if (op->device()) {
    DeviceVariant targetDevice =
        context.getTargetDevice(op->device()->global_id());
    LOG_ASSERT(std::holds_alternative<std::reference_wrapper<::ttnn::IDevice>>(
                   targetDevice),
               "ttnn::", ttnnOp.base_name(), " does not support MeshDevice.");
    device = std::get<std::reference_wrapper<::ttnn::IDevice>>(targetDevice);
  }

  ::ttnn::Tensor out = ttnnOp(shape, dtype, layout, device, memoryConfig);

  tensorPool.insertAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::NamedFullOp *op, ProgramContext &context) {
  switch (op->type()) {
  case ::tt::target::ttnn::NamedFullOpType::Zeros: {
    return runFullWithOp(op, context, ::ttnn::zeros);
  }
  case ::tt::target::ttnn::NamedFullOpType::Ones: {
    return runFullWithOp(op, context, ::ttnn::ones);
  }
  }
}
} // namespace tt::runtime::ttnn::operations::creation
