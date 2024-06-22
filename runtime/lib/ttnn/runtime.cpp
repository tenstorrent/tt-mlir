// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/utils.h"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime {
static ::tt::target::Arch toFlatbuffer(::tt::ARCH arch) {
  switch (arch) {
  case ::tt::ARCH::GRAYSKULL:
    return ::tt::target::Arch::Grayskull;
  case ::tt::ARCH::WORMHOLE_B0:
    return ::tt::target::Arch::Wormhole_b0;
  case ::tt::ARCH::BLACKHOLE:
    return ::tt::target::Arch::Blackhole;
  default:
    break;
  }

  throw std::runtime_error("Unsupported arch");
}

static ::tt::target::Dim2d toFlatbuffer(CoreCoord coreCoord) {
  return ::tt::target::Dim2d(coreCoord.y, coreCoord.x);
}

Device wrap(ttnn::Device &device) {
  return Device{utils::unsafe_borrow_shared(&device)};
}

ttnn::Device& unwrap(Device device) {
  return *static_cast<ttnn::Device *>(device.handle.get());
}

SystemDesc getCurrentSystemDesc() {
  auto &device = ttnn::open_device(0);
  ::flatbuffers::FlatBufferBuilder fbb;
  ::ttmlir::Version ttmlirVersion = ::ttmlir::getVersion();
  ::tt::target::Version version(ttmlirVersion.major, ttmlirVersion.minor,
                                ttmlirVersion.release);
  ::tt::target::Dim2d deviceGrid = toFlatbuffer(device.logical_grid_size());
  std::vector<::flatbuffers::Offset<tt::target::ChipDesc>> chipDescs = {
      ::tt::target::CreateChipDesc(fbb, toFlatbuffer(device.arch()),
                                   &deviceGrid),
  };
  std::vector<uint32_t> chipDescIndices = {
      0,
  };
  std::vector<int> chipIds = {
      device.id(),
  };
  ::tt::target::ChipCapability chipCapability =
      ::tt::target::ChipCapability::PCIE;
  if (device.is_mmio_capable()) {
    chipCapability = chipCapability | ::tt::target::ChipCapability::HostMMIO;
  }
  std::vector<::tt::target::ChipCapability> chipCapabilities = {
      chipCapability,
  };
  std::vector<::tt::target::ChipCoord> chipCoord = {
      ::tt::target::ChipCoord(0, 0, 0, 0),
  };
  std::vector<::tt::target::ChipChannel> chipChannel;
  auto systemDesc = ::tt::target::CreateSystemDescDirect(
      fbb, &chipDescs, &chipDescIndices, &chipIds, &chipCapabilities,
      &chipCoord, &chipChannel);
  auto root = ::tt::target::CreateSystemDescRootDirect(
      fbb, &version, ::ttmlir::getGitHash(), "unknown", systemDesc);
  ::tt::target::FinishSizePrefixedSystemDescRootBuffer(fbb, root);
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  if (not ::tt::target::VerifySizePrefixedSystemDescRootBuffer(verifier)) {
    throw std::runtime_error("Failed to verify system desc root buffer");
  }
  uint8_t *buf = fbb.GetBufferPointer();
  auto size = fbb.GetSize();
  auto handle = utils::malloc_shared(size);
  std::memcpy(handle.get(), buf, size);
  ttnn::close_device(device);
  return SystemDesc{handle};
}

Device openDevice(std::vector<int> deviceIds) {
  assert(deviceIds.size() == 1 && "Only one device is supported for now");
  auto &device = ttnn::open_device(deviceIds.front());
  return wrap(device);
}

void closeDevice(Device device) {
  auto &ttnn_device = unwrap(device);
  ttnn::close_device(ttnn_device);
}

Event submit(Device, Binary, std::vector<Tensor> const &,
             std::vector<Tensor> const &, std::function<void()>) {
  throw std::runtime_error("Not implemented");
}

void wait(Event) { throw std::runtime_error("Not implemented"); }

} // namespace tt::runtime
