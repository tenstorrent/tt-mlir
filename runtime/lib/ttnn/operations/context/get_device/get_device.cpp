// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "get_device.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"

namespace tt::runtime::ttnn::operations::context {
void run(const ::tt::target::ttnn::GetDeviceOp *op, ProgramContext &context) {
  DeviceMap &devicePool = context.devicePool;
  DeviceMap &allDevices = context.allDevices;
  const flatbuffers::Vector<uint32_t> *chipIds = op->chip_ids();
  assert(chipIds->size() == 1 && "Expected 1 chip id");
  for (const uint32_t chipId : *chipIds) {
    assert(allDevices.contains(chipId) && "Device not found");
    auto [iter, inserted] =
        devicePool.try_emplace(chipId, allDevices.at(chipId));
    assert(inserted && "Duplicate device");
  }
}
} // namespace tt::runtime::ttnn::operations::context
