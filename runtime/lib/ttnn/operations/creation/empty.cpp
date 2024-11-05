// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "empty.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::EmptyOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::DataType dtype = utils::getDataType(op->out());
  ::ttnn::Layout layout [[maybe_unused]] =
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());

  // TODO(bug #582): ttnn::empty doesn't work properly with tile layout,
  // using ROW_MAJOR until we fix it
  if (workaround::Env::get().emptyOpForceRowMajor) {
    layout = ::ttnn::Layout::ROW_MAJOR;
  }

  ::ttnn::Shape shape(::tt::tt_metal::LegacyShape(
      ::tt::runtime::ttnn::utils::toShapeFromFBShape(
          *op->out()->desc()->shape())));

  ::ttnn::Tensor out;
  if (op->device()) {
    // TODO (jnie): Update this once we support multi device tensors
    ::ttnn::Device &device =
        context.getDeviceFromSubMesh(op->device()->global_id(), 0);
    ::ttnn::MemoryConfig memoryConfig =
        utils::createMemoryConfig(op->memcfg(), op->out());
    out = ::ttnn::empty(shape, dtype, layout, &device, memoryConfig);
  } else {
    out = ::ttnn::zeros(shape, dtype, layout);
  }
  if (tensorPool.isUserOutput(op->out()->global_id())) {
    tensorPool.copyTensorToUserOutput(out, op->out()->global_id());
  } else {
    tensorPool.insert_or_assign(op->out()->global_id(), out);
  }
}
} // namespace tt::runtime::ttnn::operations::creation
