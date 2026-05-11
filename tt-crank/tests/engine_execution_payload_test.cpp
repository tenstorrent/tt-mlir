// Every test is DISABLED_-prefixed while the local device is unavailable.
// Drop the prefix to re-enable; the inner getNumAvailableDevices() guard
// keeps them skipping cleanly on card-less hosts.

#include "engine/compile.hpp"
#include "engine/execution_payload.hpp"

#include <gtest/gtest.h>

#include <tt/runtime/runtime.h>
#include <tt/runtime/types.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view k_trivial_add_ttir = R"mlir(
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
)mlir";

bool device_available() {
    return tt::runtime::getNumAvailableDevices() > 0;
}

std::shared_ptr<tt::kurbla::CompiledProgram> compile_for_current_device() {
    tt::kurbla::CompileOptions opts;
    opts.system_desc = tt::runtime::getCurrentSystemDesc();
    return std::make_shared<tt::kurbla::CompiledProgram>(
        tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir, opts));
}

tt::runtime::Tensor make_filled_host_tensor(const tt::runtime::TensorDesc &desc, float fill) {
    std::vector<float> buf(desc.volume(), fill);
    return tt::runtime::createOwnedHostTensor(buf.data(), desc);
}

std::vector<float> readback_floats(const tt::runtime::Tensor &device_tensor) {
    auto host_shards = tt::runtime::toHost(device_tensor, /*untilize=*/true);
    std::vector<float> out(tt::runtime::getTensorLogicalVolume(host_shards[0]));
    tt::runtime::memcpy(out.data(), host_shards[0]);
    return out;
}

} // namespace

TEST(EngineExecutionPayloadTest, DISABLED_RunsTrivialAdd) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto input_descs = program->input_descs(0);
    ASSERT_EQ(input_descs.size(), 2U);

    payload.bind_tensor(make_filled_host_tensor(input_descs[0], 0.0F), 0);
    payload.bind_tensor(make_filled_host_tensor(input_descs[1], 0.0F), 1);

    std::vector<tt::runtime::Tensor> outputs = payload.run();
    ASSERT_EQ(outputs.size(), 1U);

    std::vector<float> values = readback_floats(outputs[0]);
    for (float v : values) {
        EXPECT_EQ(v, 0.0F);
    }
}

TEST(EngineExecutionPayloadTest, DISABLED_ReuseAcrossRuns) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto input_descs = program->input_descs(0);
    payload.bind_tensor(make_filled_host_tensor(input_descs[0], 1.0F), 0);
    payload.bind_tensor(make_filled_host_tensor(input_descs[1], 2.0F), 1);

    auto outputs_a = payload.run();
    auto outputs_b = payload.run();
    ASSERT_EQ(outputs_a.size(), 1U);
    ASSERT_EQ(outputs_b.size(), 1U);

    EXPECT_EQ(readback_floats(outputs_a[0]), readback_floats(outputs_b[0]));
}

TEST(EngineExecutionPayloadTest, DISABLED_RebindSlotReplacesTensor) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto input_descs = program->input_descs(0);
    payload.bind_tensor(make_filled_host_tensor(input_descs[0], 0.0F), 0);
    payload.bind_tensor(make_filled_host_tensor(input_descs[1], 0.0F), 1);

    auto zeros = readback_floats(payload.run()[0]);

    payload.bind_tensor(make_filled_host_tensor(input_descs[0], 3.0F), 0);
    auto threes = readback_floats(payload.run()[0]);

    EXPECT_NE(zeros, threes);
}

TEST(EngineExecutionPayloadTest, DISABLED_SiblingPayloadsShareProgram) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto program = compile_for_current_device();
    tt::kurbla::ExecutionPayload a(program);
    tt::kurbla::ExecutionPayload b(program);

    EXPECT_EQ(a.compiled_program().get(), b.compiled_program().get());

    const auto input_descs = program->input_descs(0);
    a.bind_tensor(make_filled_host_tensor(input_descs[0], 1.0F), 0);
    a.bind_tensor(make_filled_host_tensor(input_descs[1], 1.0F), 1);
    b.bind_tensor(make_filled_host_tensor(input_descs[0], 4.0F), 0);
    b.bind_tensor(make_filled_host_tensor(input_descs[1], 4.0F), 1);

    EXPECT_NE(readback_floats(a.run()[0]), readback_floats(b.run()[0]));
}

TEST(EngineExecutionPayloadTest, DISABLED_RejectsWrongShape) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    // Build a desc with deliberately wrong shape (64x64 instead of 64x128).
    tt::runtime::TensorDesc wrong = program->input_descs(0)[0];
    wrong.shape = {64, 64};
    std::vector<float> buf(wrong.volume(), 0.0F);
    auto wrong_tensor = tt::runtime::createOwnedHostTensor(buf.data(), wrong);

    EXPECT_THROW(payload.bind_tensor(wrong_tensor, 0), tt::kurbla::InputBindingError);
}

TEST(EngineExecutionPayloadTest, DISABLED_RejectsMissingInput) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto input_descs = program->input_descs(0);
    payload.bind_tensor(make_filled_host_tensor(input_descs[0], 0.0F), 0);
    // Slot 1 deliberately left unbound.

    EXPECT_THROW(payload.run(), tt::kurbla::InputBindingError);
}
