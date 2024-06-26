// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc() {
  auto &device = ::ttnn::open_device(0);
  std::vector<int> chipIds = {
      device.id(),
  };
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
      fbb, &chipDescs, &chipDescIndices, &chipCapabilities, &chipCoord,
      &chipChannel);
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
  ::ttnn::close_device(device);
  return std::make_pair(SystemDesc{handle}, chipIds);
}

template <typename T>
static BorrowedStorage createStorage(void *ptr, std::uint32_t numElements) {
  return BorrowedStorage(
      borrowed_buffer::Buffer<T>(static_cast<T *>(ptr), numElements), [] {},
      [] {});
}

static BorrowedStorage createStorage(void *ptr, std::uint32_t numElements,
                                     ::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return createStorage<float>(ptr, numElements);
  // case ::tt::target::DataType::Float16:
  //   return createStorage<float16>(ptr, numElements);
  case ::tt::target::DataType::BFloat16:
    return createStorage<bfloat16>(ptr, numElements);
  case ::tt::target::DataType::UInt32:
    return createStorage<std::uint32_t>(ptr, numElements);
  case ::tt::target::DataType::UInt16:
    return createStorage<std::uint16_t>(ptr, numElements);
  // case ::tt::target::DataType::UInt8:
  //   return createStorage<std::uint8_t>(ptr, numElements);
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

static ::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return ::ttnn::DataType::FLOAT32;
  // case ::tt::target::DataType::Float16:
  //   return ::ttnn::DataType::FLOAT16;
  case ::tt::target::DataType::BFloat16:
    return ::ttnn::DataType::BFLOAT16;
  case ::tt::target::DataType::UInt32:
    return ::ttnn::DataType::UINT32;
  case ::tt::target::DataType::UInt16:
    return ::ttnn::DataType::UINT16;
  // case ::tt::target::DataType::UInt8:
  //   return ::ttnn::DataType::UINT8;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];
  auto tensor = std::make_shared<::ttnn::Tensor>(
      createStorage(data.get(), numElements, dataType), shape,
      toTTNNDataType(dataType), ::ttnn::Layout::ROW_MAJOR);
  return Tensor(tensor, data);
}

Device openDevice(std::vector<int> deviceIds) {
  assert(deviceIds.size() == 1 && "Only one device is supported for now");
  auto &device = ::ttnn::open_device(deviceIds.front());
  return Device(device);
}

void closeDevice(Device device) {
  auto &ttnn_device = device.as<::ttnn::Device>();
  ::ttnn::close_device(ttnn_device);
}

::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  if (not isTTNN) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
  ::ttnn::Device &device = deviceHandle.as<::ttnn::Device>();
  ::tt::target::ttnn::TTNNBinary const &fbb =
      *getBinary(executableHandle);
  std::vector<::ttnn::Tensor *> inputs;
  inputs.reserve(inputHandles.size());
  for (auto &input : inputHandles) {
    inputs.push_back(static_cast<::ttnn::Tensor *>(input.handle.get()));
  }
  std::vector<::ttnn::Tensor *> outputs;
  outputs.reserve(outputHandles.size());
  for (auto &output : outputHandles) {
    outputs.push_back(static_cast<::ttnn::Tensor *>(output.handle.get()));
  }
  tt::runtime::ttnn::runProgram(device, fbb.programs()->Get(programIndex),
                                inputs, outputs);
  return Event(nullptr);
}

void wait(Event) { throw std::runtime_error("Not implemented"); }

} // namespace tt::runtime
