// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "engine/ttir_module_builder.hpp"

#include <gtest/gtest.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
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
