// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/from_device.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <cstdlib>
#include <cstring>
namespace tt::runtime::ttnn::operations::layout {

// On clusters with asymmetric fabric connectivity (e.g. BH galaxy with skipped
// ETH cross-MMIO cores on boundary chips), a from_device read whose enqueued
// shard reads queue up behind boundary-chip workers that haven't completed can
// hang the dispatch core's `expected_num_workers_completed_` gating. Draining
// the command queue before issuing the reads lets the prior commands settle
// via the record_event path (which doesn't share the read-side gating bug)
// and unblocks the read.
//
// Default-off so workloads on uniform-fabric clusters (e.g. proper 1D ring
// configurations) don't pay the per-from_device sync cost.
//
// Opt in via TT_RUNTIME_FROM_DEVICE_PRESYNC=1 .
static bool fromDevicePresyncEnabled() {
  const char *env = std::getenv("TT_RUNTIME_FROM_DEVICE_PRESYNC");
  return env != nullptr && std::strcmp(env, "0") != 0;
}

void run(const ::tt::target::ttnn::FromDeviceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::from_device on a host tensor");

  if (fromDevicePresyncEnabled()) {
    inputTensor.device()->mesh_command_queue().finish();
  }

  ::ttnn::Tensor out = ::ttnn::from_device(inputTensor);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::layout
