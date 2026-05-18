// v1 tests rely on CompileOptions::mock_arch (no system descriptor),
// so they do not require a real Tenstorrent device to run. Device-backed
// tests will arrive alongside the runtime/execute API and may opt into a
// real system_desc_path.

#include "engine/compile.hpp"

#include <gtest/gtest.h>

#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

namespace {

constexpr std::string_view k_trivial_add_ttir = R"mlir(
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
)mlir";

} // namespace

TEST(EngineCompileTest, CompilesTrivialModule) {
    tt::kurbla::CompileOptions opts;
    opts.mock_arch = tt::kurbla::CompileOptions::MockArch::WormholeB0;

    tt::kurbla::CompiledProgram program = tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir, opts);

    EXPECT_GE(program.num_programs(), 1U);
}

TEST(EngineCompileTest, ThrowsOnParseError) {
    try {
        tt::kurbla::compile_ttir_to_ttnn_flatbuffer("not valid mlir");
        FAIL() << "expected ParseError";
    } catch (const tt::kurbla::ParseError &e) {
        EXPECT_NE(std::string_view(e.what()).find_first_not_of(' '), std::string_view::npos);
    } catch (...) {
        FAIL() << "expected ParseError, got a different exception";
    }
}

TEST(EngineCompileTest, ReusesEngineAcrossCalls) {
    tt::kurbla::CompileOptions opts;
    opts.mock_arch = tt::kurbla::CompileOptions::MockArch::WormholeB0;

    auto program_a = tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir, opts);
    auto program_b = tt::kurbla::compile_ttir_to_ttnn_flatbuffer(k_trivial_add_ttir, opts);

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

    tt::kurbla::CompileOptions opts;
    opts.mock_arch = tt::kurbla::CompileOptions::MockArch::WormholeB0;

    tt::kurbla::CompiledProgram program = tt::kurbla::compile_ttir_to_ttnn_flatbuffer(module_op.get(), opts);

    EXPECT_GE(program.num_programs(), 1U);
}
