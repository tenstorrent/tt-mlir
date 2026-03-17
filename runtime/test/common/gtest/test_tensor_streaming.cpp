// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the distributed large-tensor streaming protocol.
#include "tt/runtime/detail/distributed/controller/command_factory.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/detail/distributed/worker/response_factory.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace fb = ::tt::runtime::distributed::flatbuffer;
using ::tt::runtime::distributed::controller::CommandFactory;
using ::tt::runtime::distributed::controller::kMaxTensorFrameBytes;
using ::tt::runtime::distributed::worker::ResponseFactory;

static std::vector<uint8_t> makeSequentialData(size_t numBytes) {
  std::vector<uint8_t> data(numBytes);
  for (size_t i = 0; i < numBytes; ++i) {
    data[i] = static_cast<uint8_t>(i & 0xFF);
  }
  return data;
}

TEST(TensorStreamingCommandFactory, BuildFrameCommandProducesValidFlatbuffer) {
  std::vector<uint8_t> payload = makeSequentialData(1024);

  ::flatbuffers::FlatBufferBuilder fbb;
  uint64_t commandId = CommandFactory::buildTensorDataFrameCommand(
      fbb, 42u, 0u, payload.data(), payload.size());

  EXPECT_GT(fbb.GetSize(), 0u);
  EXPECT_TRUE(fb::CommandBufferHasIdentifier(fbb.GetBufferPointer()));

  const fb::Command *cmd = fb::GetCommand(fbb.GetBufferPointer());
  ASSERT_NE(cmd, nullptr);
  EXPECT_EQ(cmd->command_id(), commandId);
  EXPECT_EQ(cmd->type_type(), fb::CommandType::TensorDataFrameCommand);

  const fb::TensorDataFrameCommand *frame =
      cmd->type_as_TensorDataFrameCommand();
  ASSERT_NE(frame, nullptr);
  EXPECT_EQ(frame->output_global_id(), 42u);
  EXPECT_EQ(frame->frame_index(), 0u);
  ASSERT_NE(frame->data(), nullptr);
  EXPECT_EQ(frame->data()->size(), payload.size());
  EXPECT_EQ(std::memcmp(frame->data()->data(), payload.data(), payload.size()),
            0);
}

TEST(TensorStreamingCommandFactory, BuildFrameCommandPreservesFrameIndex) {
  std::vector<uint8_t> payload(256, 0xAB);
  for (uint32_t idx : {0u, 1u, 5u, 255u}) {
    ::flatbuffers::FlatBufferBuilder fbb;
    CommandFactory::buildTensorDataFrameCommand(fbb, 7u, idx, payload.data(),
                                                payload.size());
    const fb::TensorDataFrameCommand *frame =
        fb::GetCommand(fbb.GetBufferPointer())
            ->type_as_TensorDataFrameCommand();
    ASSERT_NE(frame, nullptr);
    EXPECT_EQ(frame->frame_index(), idx);
  }
}

TEST(TensorStreamingCommandFactory, FinalFrameEncodeNumFrames) {
  ::tt::runtime::Tensor outputHandle;
  std::vector<uint32_t> shape = {4};
  std::vector<uint32_t> stride = {1};
  std::vector<float> data = {1.f, 2.f, 3.f, 4.f};

  ::flatbuffers::FlatBufferBuilder fbb;
  CommandFactory::buildCreateHostTensorCommand(
      fbb, outputHandle, data.data(), shape, stride, sizeof(float),
      ::tt::target::DataType::Float32, 3u);

  const fb::CreateHostTensorCommand *createCmd =
      fb::GetCommand(fbb.GetBufferPointer())->type_as_CreateHostTensorCommand();
  ASSERT_NE(createCmd, nullptr);
  EXPECT_EQ(createCmd->num_frames(), 3u);
}

TEST(TensorStreamingCommandFactory, SingleFrameDefaultsToNumFramesOne) {
  ::tt::runtime::Tensor outputHandle;
  std::vector<uint32_t> shape = {2};
  std::vector<uint32_t> stride = {1};
  std::vector<float> data = {1.f, 2.f};

  ::flatbuffers::FlatBufferBuilder fbb;
  CommandFactory::buildCreateHostTensorCommand(fbb, outputHandle, data.data(),
                                               shape, stride, sizeof(float),
                                               ::tt::target::DataType::Float32);

  const fb::CreateHostTensorCommand *createCmd =
      fb::GetCommand(fbb.GetBufferPointer())->type_as_CreateHostTensorCommand();
  ASSERT_NE(createCmd, nullptr);
  EXPECT_EQ(createCmd->num_frames(), 1u);
}

TEST(TensorStreamingResponseFactory,
     BuildFrameResponseProducesValidFlatbuffer) {
  ::flatbuffers::FlatBufferBuilder fbb;
  ResponseFactory::buildTensorDataFrameResponse(fbb, 99u);

  EXPECT_GT(fbb.GetSize(), 0u);
  EXPECT_TRUE(fb::ResponseBufferHasIdentifier(fbb.GetBufferPointer()));

  const fb::Response *resp = fb::GetResponse(fbb.GetBufferPointer());
  ASSERT_NE(resp, nullptr);
  EXPECT_EQ(resp->command_id(), 99u);
  EXPECT_EQ(resp->type_type(), fb::ResponseType::TensorDataFrameResponse);
}

TEST(TensorStreamingProtocol, FramePayloadSizeRoundtrip) {
  constexpr uint64_t kFrameSize = kMaxTensorFrameBytes;
  constexpr uint64_t kTotalBytes = kFrameSize + 128;

  std::vector<uint8_t> payload = makeSequentialData(kTotalBytes);

  ::flatbuffers::FlatBufferBuilder frameFbb;
  CommandFactory::buildTensorDataFrameCommand(frameFbb, 1u, 0u, payload.data(),
                                              kFrameSize);

  const fb::TensorDataFrameCommand *frame =
      fb::GetCommand(frameFbb.GetBufferPointer())
          ->type_as_TensorDataFrameCommand();
  ASSERT_NE(frame, nullptr);
  EXPECT_EQ(frame->data()->size(), kFrameSize);
  EXPECT_EQ(std::memcmp(frame->data()->data(), payload.data(), kFrameSize), 0);

  ::tt::runtime::Tensor outputHandle;
  std::vector<uint32_t> shape = {static_cast<uint32_t>(kTotalBytes)};
  std::vector<uint32_t> stride = {1};

  ::flatbuffers::FlatBufferBuilder finalFbb;
  CommandFactory::buildCreateHostTensorCommand(
      finalFbb, outputHandle, payload.data() + kFrameSize, shape, stride,
      sizeof(uint8_t), ::tt::target::DataType::UInt8, 2u, /*dataBytes=*/128u);

  const fb::CreateHostTensorCommand *finalCmd =
      fb::GetCommand(finalFbb.GetBufferPointer())
          ->type_as_CreateHostTensorCommand();
  ASSERT_NE(finalCmd, nullptr);
  EXPECT_EQ(finalCmd->num_frames(), 2u);
  EXPECT_EQ(finalCmd->data()->size(), 128u);
  EXPECT_EQ(
      std::memcmp(finalCmd->data()->data(), payload.data() + kFrameSize, 128),
      0);
}

TEST(TensorStreamingProtocol, AssembledBufferMatchesOriginalPayload) {
  constexpr uint32_t kNumFrames = 3;
  constexpr uint64_t kFrameSize = 64;
  constexpr uint64_t kTotalBytes = kFrameSize * (kNumFrames - 1) + 32;

  std::vector<uint8_t> original = makeSequentialData(kTotalBytes);
  std::vector<uint8_t> accumulated;
  uint32_t framesReceived = 0;

  for (uint32_t i = 0; i < kNumFrames - 1; ++i) {
    ::flatbuffers::FlatBufferBuilder fbb;
    CommandFactory::buildTensorDataFrameCommand(
        fbb, 5u, i, original.data() + i * kFrameSize, kFrameSize);

    const fb::TensorDataFrameCommand *frame =
        fb::GetCommand(fbb.GetBufferPointer())
            ->type_as_TensorDataFrameCommand();
    ASSERT_NE(frame, nullptr);

    size_t offset = static_cast<size_t>(frame->frame_index()) * kFrameSize;
    if (accumulated.size() < offset + frame->data()->size()) {
      accumulated.resize(offset + frame->data()->size());
    }
    std::memcpy(accumulated.data() + offset, frame->data()->data(),
                frame->data()->size());
    ++framesReceived;
  }

  ::tt::runtime::Tensor outputHandle;
  std::vector<uint32_t> shape = {static_cast<uint32_t>(kTotalBytes)};
  std::vector<uint32_t> stride = {1};
  uint64_t lastFrameOffset = static_cast<uint64_t>(kNumFrames - 1) * kFrameSize;
  uint64_t lastFrameBytes = kTotalBytes - lastFrameOffset;

  ::flatbuffers::FlatBufferBuilder finalFbb;
  CommandFactory::buildCreateHostTensorCommand(
      finalFbb, outputHandle, original.data() + lastFrameOffset, shape, stride,
      sizeof(uint8_t), ::tt::target::DataType::UInt8, kNumFrames,
      lastFrameBytes);

  const fb::CreateHostTensorCommand *finalCmd =
      fb::GetCommand(finalFbb.GetBufferPointer())
          ->type_as_CreateHostTensorCommand();
  ASSERT_NE(finalCmd, nullptr);
  ASSERT_EQ(framesReceived, kNumFrames - 1);

  accumulated.resize(lastFrameOffset + lastFrameBytes);
  std::memcpy(accumulated.data() + lastFrameOffset, finalCmd->data()->data(),
              lastFrameBytes);

  ASSERT_EQ(accumulated.size(), original.size());
  EXPECT_EQ(std::memcmp(accumulated.data(), original.data(), original.size()),
            0);
}

TEST(TensorStreamingProtocol, SmallTensorSingleFrameUnchanged) {
  std::vector<float> data(16, 3.14f);
  std::vector<uint32_t> shape = {16};
  std::vector<uint32_t> stride = {1};

  ::tt::runtime::Tensor outputHandle;
  ::flatbuffers::FlatBufferBuilder fbb;
  CommandFactory::buildCreateHostTensorCommand(fbb, outputHandle, data.data(),
                                               shape, stride, sizeof(float),
                                               ::tt::target::DataType::Float32);

  const fb::CreateHostTensorCommand *cmd =
      fb::GetCommand(fbb.GetBufferPointer())->type_as_CreateHostTensorCommand();
  ASSERT_NE(cmd, nullptr);
  EXPECT_EQ(cmd->num_frames(), 1u);
  EXPECT_EQ(cmd->data()->size(), data.size() * sizeof(float));
}

TEST(TensorStreamingProtocol, MultipleIndependentTensorsDoNotInterleave) {
  constexpr uint64_t kGlobalIdA = 100;
  constexpr uint64_t kGlobalIdB = 200;
  constexpr uint64_t kFrameBytes = 32;

  std::vector<uint8_t> payloadA = makeSequentialData(kFrameBytes);
  std::vector<uint8_t> payloadB(kFrameBytes, 0xFF);

  ::flatbuffers::FlatBufferBuilder fbbA;
  CommandFactory::buildTensorDataFrameCommand(fbbA, kGlobalIdA, 0u,
                                              payloadA.data(), kFrameBytes);

  ::flatbuffers::FlatBufferBuilder fbbB;
  CommandFactory::buildTensorDataFrameCommand(fbbB, kGlobalIdB, 0u,
                                              payloadB.data(), kFrameBytes);

  const fb::TensorDataFrameCommand *frameA =
      fb::GetCommand(fbbA.GetBufferPointer())->type_as_TensorDataFrameCommand();
  const fb::TensorDataFrameCommand *frameB =
      fb::GetCommand(fbbB.GetBufferPointer())->type_as_TensorDataFrameCommand();

  ASSERT_NE(frameA, nullptr);
  ASSERT_NE(frameB, nullptr);
  EXPECT_EQ(frameA->output_global_id(), kGlobalIdA);
  EXPECT_EQ(frameB->output_global_id(), kGlobalIdB);
  EXPECT_NE(
      std::memcmp(frameA->data()->data(), frameB->data()->data(), kFrameBytes),
      0);
}
