// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/global_semaphore.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::creation {
void run(const ::tt::target::ttnn::CreateGlobalSemaphoreOp *op,
         ProgramContext &context) {
  ProgramGlobalSemaphorePool &globalSemaphorePool =
      context.getGlobalSemaphorePool();
  ::tt::tt_metal::CoreRangeSet coreRangeSet =
      ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*op->core_range_set());
  ::ttnn::MeshDevice *meshDevice = context.getMeshDevicePtr().get();
  ::ttnn::GlobalSemaphore globalSemaphore =
      ::ttnn::global_semaphore::create_global_semaphore(
          meshDevice, coreRangeSet, op->initial_value());
  globalSemaphorePool.insertTTNNGlobalSemaphoreAndValidate(op->out(),
                                                           globalSemaphore);
}

void run(const ::tt::target::ttnn::ResetGlobalSemaphoreOp *op,
         ProgramContext &context) {
  ProgramGlobalSemaphorePool &globalSemaphorePool =
      context.getGlobalSemaphorePool();
  ::ttnn::GlobalSemaphore &semaphore =
      globalSemaphorePool.getTTNNGlobalSemaphoreAndValidate(op->semaphore());
  ::ttnn::global_semaphore::reset_global_semaphore_value(semaphore,
                                                         op->value());
}
} // namespace tt::runtime::ttnn::operations::creation
