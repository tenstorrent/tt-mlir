#include "engine/compile.hpp"

#include <gtest/gtest.h>

#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::string_view k_trivial_add_ttir = R"mlir(
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
)mlir";

} // namespace

TEST(EngineCompileTest, CompilesTrivialModule) {
    tt::kurbla::CompiledProgram &program = *tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir).program;

    EXPECT_GE(program.num_programs(), 1U);
}

TEST(EngineCompileTest, ThrowsOnParseError) {
    try {
        tt::kurbla::compile_ttir_to_ttnn_flatbuffer("not valid mlir");
        FAIL() << "expected ParseError";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string_view(e.what()).find_first_not_of(' '), std::string_view::npos);
    } catch (...) {
        FAIL() << "expected ParseError, got a different exception";
    }
}

TEST(EngineCompileTest, ReusesEngineAcrossCalls) {
    auto &program_a = *tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir).program;
    auto &program_b = *tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir).program;

    EXPECT_GE(program_a.num_programs(), 1U);
    EXPECT_GE(program_b.num_programs(), 1U);
}

// Exercises the ModuleOp overload — the entry point in-memory TTIR producers
// (e.g. PyTorch FX -> TTIR lowering) will use. Builds the module by parsing
// text into the shared MLIRContext; a real producer would construct the IR
// via mlir::OpBuilder instead, but the compile call site is identical.
TEST(EngineCompileTest, CompilesPreBuiltModule) {
    mlir::MLIRContext &ctx = tt::kurbla::mlir_context();

    mlir::OwningOpRef<mlir::ModuleOp> module_op = mlir::parseSourceString<mlir::ModuleOp>(k_trivial_add_ttir, &ctx);
    ASSERT_TRUE(module_op);

    tt::kurbla::CompiledProgram &program = *tt::kurbla::compile_ttir_to_ttnn_flatbuffer(module_op.get()).program;

    EXPECT_GE(program.num_programs(), 1U);
}

// ttnn_ir() recovers the post-pipeline module from the compiled flatbuffer.
// Confirm it recovers the real lowered module, not an empty or partial buffer.
TEST(EngineCompileTest, RecoversTTNNIRFromFlatbuffer) {
    auto &program = *tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir).program;

    std::string_view ir = program.ttnn_ir();
    EXPECT_FALSE(ir.empty());
    EXPECT_NE(ir.find("func.func"), std::string_view::npos);
    EXPECT_NE(ir.find("ttnn.add"), std::string_view::npos);
}

// Capturing the TTIR costs a full module print, so it is opt-in.
TEST(EngineCompileTest, DoesNotCaptureTTIRByDefault) {
    tt::kurbla::CompileResult result = tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir);

    EXPECT_NE(result.program, nullptr);
    EXPECT_TRUE(result.ttir.empty());
}

// Verify that the TTIR is properly captured when the user demands it.
TEST(EngineCompileTest, CapturesTTIRFromModuleBeforePipeline) {
    mlir::MLIRContext &ctx = tt::kurbla::mlir_context();

    mlir::OwningOpRef<mlir::ModuleOp> module_op = mlir::parseSourceString<mlir::ModuleOp>(k_trivial_add_ttir, &ctx);
    ASSERT_TRUE(module_op);

    tt::kurbla::CompileResult result =
        tt::kurbla::compile_ttir_to_ttnn_flatbuffer(module_op.get(), {}, /*capture_ttir=*/true);

    EXPECT_NE(result.program, nullptr);
    ASSERT_FALSE(result.ttir.empty());
    EXPECT_NE(result.ttir.find("ttir.add"), std::string::npos);
    EXPECT_EQ(result.ttir.find("ttnn."), std::string::npos);
}

// The duration must be populated on both the compile and the cache-hit path.
// No cold-vs-warm comparison: the on-disk cache persists across runs, so the
// first call here isn't guaranteed to be a real compile.
TEST(EngineCompileTest, ReportsCompileDuration) {
    tt::kurbla::CompileResult first = tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir);
    EXPECT_GT(first.compile_duration.count(), 0.0);

    tt::kurbla::CompileResult second = tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir);
    EXPECT_TRUE(second.cache_hit);
    EXPECT_GT(second.compile_duration.count(), 0.0);
}

// Verify that the TTIR is properly captured when the user demands it.
TEST(EngineCompileTest, CapturesTTIRFromStringInput) {
    tt::kurbla::CompileResult result =
        tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir, {}, /* capture_ttir= */ true);

    EXPECT_NE(result.program, nullptr);
    EXPECT_EQ(std::string_view(result.ttir), k_trivial_add_ttir);
}
