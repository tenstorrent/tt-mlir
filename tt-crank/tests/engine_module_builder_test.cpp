// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "engine/ttir_module_builder.hpp"

#include <gtest/gtest.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "engine/compile.hpp"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
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

// create_composite emits a `ttcore.composite` plus a private decomposition function; tt-mlir
// inlines the function for a name its registry does not know, so the program still compiles.
TEST(EngineModuleBuilderTest, CreateCompositeEmitsOpAndDecomposition) {
    auto mb = ModuleBuilder::init({f32_spec({32, 32}), f32_spec({32, 32})});
    auto result_type = mb.args()[0].getType();
    auto attrs = mb.attrs().getNamedAttr("flag", mb.attrs().getBoolAttr(true));
    auto results = mb.create_composite(
        "test_add", mb.args(), {result_type}, {attrs}, [](ModuleBuilder &body, mlir::ValueRange args) {
            return llvm::SmallVector<mlir::Value, 4>{tt::crank::build_add(body, args[0], args[1])};
        });
    ASSERT_EQ(results.size(), 1U);
    auto module_op = std::move(mb).finalize({results[0]});

    auto func = mlir::SymbolTable(*module_op).lookup<mlir::func::FuncOp>("test_add_decomposition");
    ASSERT_TRUE(func);
    EXPECT_TRUE(func.isPrivate());
    EXPECT_EQ(func.getFunctionType().getNumInputs(), 2U);
    int adds_in_body = 0;
    func.walk([&](mlir::tt::ttir::AddOp) { ++adds_in_body; });
    EXPECT_EQ(adds_in_body, 1);

    int composites = 0;
    module_op->walk([&](mlir::tt::ttcore::CompositeOp op) {
        ++composites;
        EXPECT_EQ(op.getCompositeName(), "test_add");
        EXPECT_EQ(op.getDecomposition(), "test_add_decomposition");
        ASSERT_TRUE(op.getCompositeAttributes().has_value());
        EXPECT_TRUE(op.getCompositeAttributes()->contains("flag"));
    });
    EXPECT_EQ(composites, 1);

    // Unknown to the promotion registry: the pipeline inlines the decomposition.
    tt::crank::CompiledProgram &program = *tt::crank::compile_ttir_to_ttnn_flatbuffer(*module_op).program;
    EXPECT_NE(program.ttnn_ir().find("ttnn.add"), std::string_view::npos);
    EXPECT_EQ(program.ttnn_ir().find("composite"), std::string_view::npos);
}

// Two composites with the same name in one module: SymbolTable::insert renames the second decomposition
// function and the composite references the renamed symbol, so both stay resolvable.
TEST(EngineModuleBuilderTest, CreateCompositeUniquifiesDecompositionSymbols) {
    auto mb = ModuleBuilder::init({f32_spec({32, 32}), f32_spec({32, 32})});
    auto result_type = mb.args()[0].getType();
    auto add = [](ModuleBuilder &body, mlir::ValueRange args) {
        return llvm::SmallVector<mlir::Value, 4>{tt::crank::build_add(body, args[0], args[1])};
    };
    auto first = mb.create_composite("test_add", mb.args(), {result_type}, {}, add);
    auto second = mb.create_composite("test_add", {first[0], mb.args()[1]}, {result_type}, {}, add);
    auto module_op = std::move(mb).finalize({second[0]});

    llvm::SmallVector<std::string, 2> referenced;
    module_op->walk([&](mlir::tt::ttcore::CompositeOp op) {
        EXPECT_EQ(op.getCompositeName(), "test_add");
        referenced.push_back(op.getDecomposition().str());
        EXPECT_TRUE(mlir::SymbolTable(*module_op).lookup<mlir::func::FuncOp>(op.getDecomposition()));
    });
    ASSERT_EQ(referenced.size(), 2U);
    EXPECT_NE(referenced[0], referenced[1]);

    tt::crank::CompiledProgram &program = *tt::crank::compile_ttir_to_ttnn_flatbuffer(*module_op).program;
    EXPECT_EQ(program.num_inputs, 2U);
}

// Composites interleaved with plain ops: matmul -> composite(add) -> matmul -> composite(add). Both
// composites survive into the TTIR handed to the pipeline (distinct decomposition symbols), and the
// compiled TTNN keeps the op order with the decompositions inlined in place.
TEST(EngineModuleBuilderTest, CompositesInterleavedWithMatmuls) {
    auto mb = ModuleBuilder::init({f32_spec({32, 64}), f32_spec({64, 32}), f32_spec({32, 32})});
    auto a = mb.args();
    auto add = [](ModuleBuilder &body, mlir::ValueRange args) {
        return llvm::SmallVector<mlir::Value, 4>{tt::crank::build_add(body, args[0], args[1])};
    };
    mlir::Value m1 = tt::crank::build_matmul(mb, a[0], a[1]); // [32, 32]
    mlir::Value c1 = mb.create_composite("test_add", {m1, a[2]}, {m1.getType()}, {}, add)[0];
    mlir::Value m2 = tt::crank::build_matmul(mb, c1, a[2]); // [32, 32]
    mlir::Value c2 = mb.create_composite("test_add", {m2, c1}, {m2.getType()}, {}, add)[0];
    auto module_op = std::move(mb).finalize({c2});

    tt::crank::CompileResult result = tt::crank::compile_ttir_to_ttnn_flatbuffer(*module_op, {}, /*capture_ttir=*/true);
    const std::string &ttir = result.ttir;
    EXPECT_EQ(std::count(ttir.begin(), ttir.end(), '\n') > 0, true);
    auto count = [](std::string_view text, std::string_view needle) {
        std::size_t n = 0;
        for (std::size_t pos = text.find(needle); pos != std::string_view::npos; pos = text.find(needle, pos + 1)) {
            ++n;
        }
        return n;
    };
    EXPECT_EQ(count(ttir, "ttcore.composite"), 2U);
    EXPECT_EQ(count(ttir, "func.func private @test_add_decomposition"), 2U); // base name + renamed sibling
    EXPECT_EQ(count(ttir, "\"ttir.matmul\""), 2U);
    EXPECT_EQ(count(ttir, "\"ttir.add\""), 2U); // one per decomposition body

    std::string_view ttnn = result.program->ttnn_ir();
    EXPECT_EQ(count(ttnn, "composite"), 0U);
    EXPECT_EQ(count(ttnn, "\"ttnn.matmul\""), 2U);
    EXPECT_EQ(count(ttnn, "\"ttnn.add\""), 2U);
    // Order is preserved: matmul, add, matmul, add.
    const std::size_t m1_pos = ttnn.find("\"ttnn.matmul\""), add1 = ttnn.find("\"ttnn.add\"");
    const std::size_t m2_pos = ttnn.find("\"ttnn.matmul\"", m1_pos + 1), add2 = ttnn.find("\"ttnn.add\"", add1 + 1);
    EXPECT_LT(m1_pos, add1);
    EXPECT_LT(add1, m2_pos);
    EXPECT_LT(m2_pos, add2);
    EXPECT_EQ(result.program->num_inputs, 3U);
}

namespace {

tt::crank::TensorTypeSpec bf16_spec(std::vector<std::int64_t> shape) {
    return tt::crank::TensorTypeSpec{std::move(shape), ::tt::target::DataType::BFloat16};
}

int64_t last_dim(mlir::Value v) {
    return mlir::cast<mlir::RankedTensorType>(v.getType()).getShape().back();
}

template <typename Op> int count_ops(mlir::ModuleOp module) {
    int n = 0;
    module.walk([&](Op) { ++n; });
    return n;
}

} // namespace

// ttml's sdpa kernels refuse Q/K whose head_dim is not tile-aligned. build_sdpa_fw/bw zero-pad Q and K to the
// next multiple of 32 around the composites and slice dQ/dK back, so the aten-facing shapes stay D = 40.
TEST(EngineModuleBuilderTest, SdpaPadsHeadDimForTtml) {
    const std::vector<std::int64_t> qkv{1, 8, 32, 40};
    auto mb = ModuleBuilder::init({bf16_spec(qkv), bf16_spec(qkv), bf16_spec(qkv), bf16_spec(qkv)});
    auto a = mb.args();
    auto [out, lse] = tt::crank::build_sdpa_fw(mb, a[0], a[1], a[2], /*is_causal=*/true, std::nullopt, {});
    EXPECT_EQ(last_dim(out), 40);
    EXPECT_EQ(mlir::cast<mlir::RankedTensorType>(lse.getType()).getShape().size(), 3U);
    auto [dq, dk, dv] =
        tt::crank::build_sdpa_bw(mb, a[3], out, a[0], a[1], a[2], lse, /*is_causal=*/true, std::nullopt, {});
    EXPECT_EQ(last_dim(dq), 40);
    EXPECT_EQ(last_dim(dk), 40);
    EXPECT_EQ(last_dim(dv), 40);
    auto module_op = std::move(mb).finalize({dq, dk, dv});

    // The composites see padded Q/K (64) and unpadded V (40); the backward's dQ/dK results are padded too.
    module_op->walk([&](mlir::tt::ttcore::CompositeOp op) {
        const bool forward = op.getCompositeName() == "sdpa_fw";
        const unsigned q_idx = forward ? 0 : 2;
        EXPECT_EQ(last_dim(op.getInputs()[q_idx]), 64) << op.getCompositeName().str();
        EXPECT_EQ(last_dim(op.getInputs()[q_idx + 1]), 64) << op.getCompositeName().str();
        EXPECT_EQ(last_dim(op.getInputs()[q_idx + 2]), 40) << op.getCompositeName().str();
        if (forward) {
            EXPECT_EQ(last_dim(op.getResults()[0]), 40);
        } else {
            EXPECT_EQ(last_dim(op.getResults()[0]), 64);
            EXPECT_EQ(last_dim(op.getResults()[1]), 64);
            EXPECT_EQ(last_dim(op.getResults()[2]), 40);
        }
    });
    EXPECT_EQ(count_ops<mlir::tt::ttcore::CompositeOp>(*module_op), 2);
    EXPECT_EQ(count_ops<mlir::tt::ttir::PadOp>(*module_op), 4); // Q and K, forward and backward

    // Inlined decomposition path compiles with the padded shapes.
    tt::crank::CompiledProgram &program = *tt::crank::compile_ttir_to_ttnn_flatbuffer(*module_op).program;
    EXPECT_EQ(program.num_inputs, 4U);
}

TEST(EngineModuleBuilderTest, SdpaAlignedHeadDimIsNotPadded) {
    const std::vector<std::int64_t> qkv{1, 8, 32, 64};
    auto mb = ModuleBuilder::init({bf16_spec(qkv), bf16_spec(qkv), bf16_spec(qkv)});
    auto a = mb.args();
    auto [out, lse] = tt::crank::build_sdpa_fw(mb, a[0], a[1], a[2], /*is_causal=*/true, std::nullopt, {});
    auto module_op = std::move(mb).finalize({out});
    EXPECT_EQ(count_ops<mlir::tt::ttir::PadOp>(*module_op), 0);
    EXPECT_EQ(count_ops<mlir::tt::ttir::SliceStaticOp>(*module_op), 1); // only the lse column slice
}
