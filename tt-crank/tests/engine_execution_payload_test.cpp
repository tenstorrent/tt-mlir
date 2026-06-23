// The inner getNumAvailableDevices() guard skips cleanly on card-less hosts.
// Running with TT_KURBLA_USE_SIMULATOR=1 (e.g. `ctest --preset sim`) routes the
// runtime through ttsim via src/engine/sim_env.cpp, so these execute without a
// physical device.

#include "cast.hpp"
#include "engine/compile.hpp"
#include "engine/device.hpp"
#include "engine/execution_payload.hpp"

#include <gtest/gtest.h>

#include <tt/runtime/runtime.h>
#include <tt/runtime/types.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

namespace {

// bf16, not f32: ttsim hits "tensix_execute_unpacr: in_data_format=0" UB on
// TTNN-emitted f32 kernels (the unpacker config registers come up zeroed and
// ttsim flags that as undefined). Compiling the same op to bf16 sidesteps that
// path; mirrors how tt-mlir's own ttsim CI only exercises bf16 paths.
constexpr std::string_view k_trivial_add_ttir = R"mlir(
func.func @add(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  return %0 : tensor<64x128xbf16>
}
)mlir";

bool device_available() {
    return tt::runtime::getNumAvailableDevices() > 0;
}

// Returns the cached CompiledProgram by reference; sibling payloads compiled
// from the same TTIR share one program instance (the compile cache dedupes).
tt::kurbla::CompiledProgram &compile_for_current_device() {
    tt::kurbla::CompileOptions opts;
    opts.system_desc = tt::kurbla::runtime_system_desc();
    return tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir, opts);
}

// bf16 has the same sign+exponent encoding as f32; truncating the low 16 bits
// rounds toward zero, which is fine for the integral fill values we use here.
std::uint16_t float_to_bf16(float f) {
    return as<std::uint16_t>(std::bit_cast<std::uint32_t>(f) >> 16);
}

float bf16_to_float(std::uint16_t b) {
    return std::bit_cast<float>(as<std::uint32_t>(b) << 16);
}

tt::runtime::Tensor make_filled_host_tensor(const tt::runtime::TensorDesc &desc, float fill) {
    std::vector<std::uint16_t> buf(desc.volume(), float_to_bf16(fill));
    return tt::runtime::createOwnedHostTensor(buf.data(), desc);
}

// bind_tensor takes a non-const Tensor& (it may rewrite the tensor's layout in
// place), so the make_filled_host_tensor temporary must be materialized as an
// lvalue before binding.
void bind_filled(tt::kurbla::ExecutionPayload &payload, const tt::runtime::TensorDesc &desc, float fill,
                 std::uint32_t index) {
    tt::runtime::Tensor tensor = make_filled_host_tensor(desc, fill);
    payload.bind_tensor(tensor, index);
}

std::vector<float> readback_floats(const tt::runtime::Tensor &device_tensor) {
    auto host_shards = tt::runtime::toHost(device_tensor, /*untilize=*/true);
    const size_t n = tt::runtime::getTensorLogicalVolume(host_shards[0]);
    std::vector<std::uint16_t> bf16_buf(n);
    tt::runtime::memcpy(bf16_buf.data(), host_shards[0]);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = bf16_to_float(bf16_buf[i]);
    }
    return out;
}

} // namespace

TEST(EngineExecutionPayloadTest, RunsTrivialAdd) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto &program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto &input_descs = program.input_descs;
    ASSERT_EQ(input_descs.size(), 2U);

    bind_filled(payload, input_descs[0], 0.0F, 0);
    bind_filled(payload, input_descs[1], 0.0F, 1);

    std::vector<tt::runtime::Tensor> outputs = payload.run();
    ASSERT_EQ(outputs.size(), 1U);

    std::vector<float> values = readback_floats(outputs[0]);
    for (float v : values) {
        EXPECT_EQ(v, 0.0F);
    }
}

TEST(EngineExecutionPayloadTest, RunsMultipleAdds) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    for (size_t i = 0; i < 16; ++i) {
        auto &program = compile_for_current_device();
        tt::kurbla::ExecutionPayload payload(program);

        const auto &input_descs = program.input_descs;
        ASSERT_EQ(input_descs.size(), 2U);

        bind_filled(payload, input_descs[0], 2.0F, 0);
        bind_filled(payload, input_descs[1], 2.0F, 1);

        std::vector<tt::runtime::Tensor> outputs = payload.run();
        ASSERT_EQ(outputs.size(), 1U);

        std::vector<float> values = readback_floats(outputs[0]);
        for (float v : values) {
            EXPECT_EQ(v, 4.0F);
        }
    }
}

TEST(EngineExecutionPayloadTest, ReuseAcrossRuns) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto &program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto &input_descs = program.input_descs;
    bind_filled(payload, input_descs[0], 1.0F, 0);
    bind_filled(payload, input_descs[1], 2.0F, 1);

    auto outputs_a = payload.run();
    auto outputs_b = payload.run();
    ASSERT_EQ(outputs_a.size(), 1U);
    ASSERT_EQ(outputs_b.size(), 1U);

    EXPECT_EQ(readback_floats(outputs_a[0]), readback_floats(outputs_b[0]));
}

TEST(EngineExecutionPayloadTest, RebindSlotReplacesTensor) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto &program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto &input_descs = program.input_descs;
    bind_filled(payload, input_descs[0], 0.0F, 0);
    bind_filled(payload, input_descs[1], 0.0F, 1);

    auto zeros = readback_floats(payload.run()[0]);

    bind_filled(payload, input_descs[0], 3.0F, 0);
    auto threes = readback_floats(payload.run()[0]);

    EXPECT_NE(zeros, threes);
}

TEST(EngineExecutionPayloadTest, SiblingPayloadsShareProgram) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto &program = compile_for_current_device();
    tt::kurbla::ExecutionPayload a(program);
    tt::kurbla::ExecutionPayload b(program);

    EXPECT_EQ(&a.compiled_program(), &b.compiled_program());

    const auto &input_descs = program.input_descs;
    bind_filled(a, input_descs[0], 1.0F, 0);
    bind_filled(a, input_descs[1], 1.0F, 1);
    bind_filled(b, input_descs[0], 4.0F, 0);
    bind_filled(b, input_descs[1], 4.0F, 1);

    EXPECT_NE(readback_floats(a.run()[0]), readback_floats(b.run()[0]));
}

TEST(EngineExecutionPayloadTest, RejectsWrongShape) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto &program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    // Build a desc with deliberately wrong shape (64x64 instead of 64x128).
    tt::runtime::TensorDesc wrong = program.input_descs[0];
    wrong.shape = {64, 64};
    auto wrong_tensor = make_filled_host_tensor(wrong, 0.0F);

    EXPECT_THROW(payload.bind_tensor(wrong_tensor, 0), std::runtime_error);
}

TEST(EngineExecutionPayloadTest, RejectsMissingInput) {
    if (!device_available()) {
        GTEST_SKIP() << "no Tenstorrent device available.";
    }

    auto &program = compile_for_current_device();
    tt::kurbla::ExecutionPayload payload(program);

    const auto &input_descs = program.input_descs;
    bind_filled(payload, input_descs[0], 0.0F, 0);
    // Slot 1 deliberately left unbound.

    EXPECT_THROW(payload.run(), std::runtime_error);
}
