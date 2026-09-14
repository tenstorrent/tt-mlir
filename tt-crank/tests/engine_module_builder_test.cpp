// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "engine/ttir_module_builder.hpp"

#include <gtest/gtest.h>

#include "mlir/IR/BuiltinTypes.h"
#include <stdexcept>
#include <vector>

#include "engine/compile.hpp"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

// Besides covering the builder itself, this test is the RTTI tripwire for the
// exported class: the test target builds with default RTTI against the
// -fno-rtti libtt_crank.so, so a public-header change that drags RTTI-needing
// tt-mlir machinery into consumers fails to link here first.

namespace {

using tt::crank::ModuleBuilder;
using tt::crank::TensorTypeSpec;

TensorTypeSpec f32_spec(std::vector<std::int64_t> shape) {
    return TensorTypeSpec{std::move(shape), ::tt::target::DataType::Float32};
}

} // namespace

TEST(EngineModuleBuilderTest, BuildsAndCompilesAddModule) {
    auto mb = ModuleBuilder::init({f32_spec({64, 128}), f32_spec({64, 128})});
    ASSERT_EQ(mb.args().size(), 2U);

    auto result_type = mlir::cast<mlir::RankedTensorType>(mb.args()[0].getType());
    mlir::Value sum = mb.create<mlir::tt::ttir::AddOp>(result_type, mb.args()[0], mb.args()[1]).getResult();
    auto module_op = std::move(mb).finalize({sum});

    tt::crank::CompiledProgram &program = *tt::crank::compile_ttir_to_ttnn_flatbuffer(*module_op).program;
    ASSERT_EQ(program.num_inputs, 2U);
    ASSERT_EQ(program.output_descs.size(), 1U);
    EXPECT_EQ(program.output_descs[0].shape, (std::vector<std::uint32_t>{64, 128}));
}

TEST(EngineModuleBuilderTest, InsertTypecastIsNoOpOnMatchingType) {
    auto mb = ModuleBuilder::init({f32_spec({32, 32})});
    auto f32 = mb.attrs().getF32Type();

    // Same element type: must return the value unchanged, no op emitted.
    EXPECT_EQ(mb.insert_typecast(mb.args()[0], f32), mb.args()[0]);
    // Different element type: must produce a new value.
    EXPECT_NE(mb.insert_typecast(mb.args()[0], mb.attrs().getBF16Type()), mb.args()[0]);
}

TEST(EngineModuleBuilderTest, ThrowsOnUnsupportedDtype) {
    EXPECT_THROW(ModuleBuilder::init({TensorTypeSpec{{2, 2}, ::tt::target::DataType::Float16}}), std::runtime_error);
}

TEST(EngineModuleBuilderTest, ThrowsOnArgTypesSizeMismatch) {
    EXPECT_THROW(ModuleBuilder::init({f32_spec({2, 2})},
                                     {mlir::tt::ttcore::ArgumentType::Input, mlir::tt::ttcore::ArgumentType::Input}),
                 std::runtime_error);
}
