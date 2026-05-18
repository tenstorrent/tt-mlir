#include "torch/ttir_module_builder.hpp"

#include <utility>

#include <c10/util/Exception.h>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include "engine/compile.hpp"

namespace tt::kurbla::torch_backend {

namespace {

mlir::Type to_mlir_element_type(mlir::MLIRContext &ctx, ::tt::target::DataType dtype) {
    mlir::Builder b(&ctx);
    switch (dtype) {
        case ::tt::target::DataType::BFloat16:
            return b.getBF16Type();
        case ::tt::target::DataType::Float32:
            return b.getF32Type();
        case ::tt::target::DataType::Int32:
            return b.getI32Type();
        default:
            break;
    }
    TORCH_CHECK(false,
                "tt-kurbla ModuleBuilder: unsupported runtime dtype for MLIR element type: ", static_cast<int>(dtype));
}

mlir::RankedTensorType to_tensor_type(mlir::MLIRContext &ctx, const TensorTypeSpec &spec) {
    return mlir::RankedTensorType::get(spec.shape, to_mlir_element_type(ctx, spec.dtype));
}

} // namespace

ModuleBuilder::ModuleBuilder(mlir::OwningOpRef<mlir::ModuleOp> module_op, mlir::func::FuncOp func,
                             mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<mlir::Value> args)
    : module_op_(std::move(module_op)), func_(func), builder_(std::move(builder)), loc_(loc), args_(std::move(args)) {}

ModuleBuilder ModuleBuilder::init(llvm::ArrayRef<TensorTypeSpec> inputs) {
    auto &ctx = ::tt::kurbla::mlir_context();
    auto loc = mlir::UnknownLoc::get(&ctx);

    llvm::SmallVector<mlir::Type> input_types;
    input_types.reserve(inputs.size());
    for (const auto &spec : inputs) {
        input_types.push_back(to_tensor_type(ctx, spec));
    }

    auto module_op = mlir::ModuleOp::create(loc);
    mlir::OpBuilder module_builder(module_op.getBodyRegion());
    auto fn_type = mlir::FunctionType::get(&ctx, input_types, /*results=*/{});
    auto func = module_builder.create<mlir::func::FuncOp>(loc, "main", fn_type);

    mlir::Block *entry = func.addEntryBlock();
    mlir::OpBuilder body_builder(&ctx);
    body_builder.setInsertionPointToStart(entry);

    llvm::SmallVector<mlir::Value> args(entry->args_begin(), entry->args_end());

    return ModuleBuilder(mlir::OwningOpRef<mlir::ModuleOp>(module_op), func, std::move(body_builder), loc,
                         std::move(args));
}

mlir::OwningOpRef<mlir::ModuleOp> ModuleBuilder::finalize(llvm::ArrayRef<mlir::Value> outputs) && {
    builder_.create<mlir::func::ReturnOp>(loc_, mlir::ValueRange(outputs));

    llvm::SmallVector<mlir::Type> result_types;
    result_types.reserve(outputs.size());
    for (auto v : outputs) {
        result_types.push_back(v.getType());
    }
    auto input_types = func_.getFunctionType().getInputs();
    func_.setFunctionType(mlir::FunctionType::get(builder_.getContext(), input_types, result_types));

    return std::move(module_op_);
}

} // namespace tt::kurbla::torch_backend
