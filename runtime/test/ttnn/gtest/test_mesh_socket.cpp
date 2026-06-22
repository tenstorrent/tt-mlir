// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Pipeline-parallel device-to-device socket round-trip between two 1x1
// submeshes of a [2,2] mesh over FABRIC_2D.

#include "tt/runtime/detail/test/ttnn/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

class MeshSocketTest : public ::testing::Test {
protected:
  void SetUp() override {
    tt::runtime::setCurrentDeviceRuntime(tt::runtime::DeviceRuntime::TTNN);
  }
};

TEST_F(MeshSocketTest, RoundTripBetweenTwoSubmeshes) {
  // Fabric MUST be configured before opening the mesh: openMeshDevice does not
  // auto-configure it, and cross-submesh sockets TT_FATAL without it.
  // NOTE: confirm the exact enum qualifier on-box (generated from
  // runtime/include/tt/runtime/flatbuffer/types.fbs).
  tt::runtime::setFabricConfig(tt::runtime::FabricConfig::FABRIC_2D);

  tt::runtime::MeshDeviceOptions opts;
  opts.meshShape = std::vector<uint32_t>{2, 2};
  tt::runtime::Device parent = tt::runtime::openMeshDevice(opts);

  tt::runtime::Device mA = tt::runtime::createSubMeshDevice(
      parent, /*meshShape=*/{1, 1}, /*meshOffset=*/std::vector<uint32_t>{0, 0});
  tt::runtime::Device mB = tt::runtime::createSubMeshDevice(
      parent, /*meshShape=*/{1, 1}, /*meshOffset=*/std::vector<uint32_t>{0, 1});

  // Host source data.
  std::vector<uint32_t> shape = {32, 32};
  std::vector<int64_t> stride = tt::runtime::utils::calculateStride(shape);
  tt::target::DataType dataType = tt::target::DataType::Float32;
  uint32_t itemSize = sizeof(float);
  std::vector<float> data(shape[0] * shape[1]);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(i) * 0.5f - 3.0f;
  }
  tt::runtime::Tensor host = tt::runtime::createOwnedHostTensor(
      data.data(), shape, stride, itemSize, dataType);

  tt::runtime::Layout deviceLayout =
      tt::runtime::test::ttnn::getDramInterleavedTileLayout(dataType);

  // Source on submesh A; pre-allocated destination on submesh B.
  tt::runtime::Tensor srcA =
      tt::runtime::toLayout(host, mA, deviceLayout, /*retain=*/true);
  tt::runtime::Tensor dstB =
      tt::runtime::toLayout(host, mB, deviceLayout, /*retain=*/true);

  // Move srcA -> dstB device-to-device over a fabric socket.
  auto [sendSocket, recvSocket] =
      tt::runtime::createSocketPair(mA, mB, /*senderCore=*/{0, 0},
                                    /*receiverCore=*/{0, 1}, /*fifoSize=*/16 * 1024);
  tt::runtime::socketSend(sendSocket, srcA); // enqueue on mA
  tt::runtime::socketRecv(recvSocket, dstB); // enqueue on mB
  // Fence BOTH queues only after issuing both (fencing between can deadlock on a
  // full socket FIFO). wait(Tensor) is insufficient here (no MeshEvent attached).
  tt::runtime::synchronizeDevice(mA);
  tt::runtime::synchronizeDevice(mB);

  std::vector<tt::runtime::Tensor> srcHost =
      tt::runtime::toHost(srcA, /*untilize=*/true, /*blocking=*/true);
  std::vector<tt::runtime::Tensor> dstHost =
      tt::runtime::toHost(dstB, /*untilize=*/true, /*blocking=*/true);

  EXPECT_EQ(tt::runtime::getTensorDataBuffer(srcHost[0]),
            tt::runtime::getTensorDataBuffer(dstHost[0]));

  tt::runtime::closeSocket(sendSocket);
  tt::runtime::closeSocket(recvSocket);
  tt::runtime::releaseSubMeshDevice(mA);
  tt::runtime::releaseSubMeshDevice(mB);
  tt::runtime::closeMeshDevice(parent);
}
