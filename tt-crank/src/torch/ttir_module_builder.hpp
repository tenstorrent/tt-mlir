#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TypeProperties.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <tt/runtime/types.h>
#include <ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h>

namespace tt::kurbla::torch_backend {

struct TensorTypeSpec {
    std::vector<std::int64_t> shape;
    ::tt::target::DataType dtype;
};

// Scoped helper for assembling a single-function TTIR module from an ATen
// kernel: `ModuleBuilder::init(...)` creates the module + `func @main` + entry
// block; ops/attrs are emitted via `create<>` / `attrs()`; `std::move(mb).finalize(...)`
// patches the func signature and yields the ready-to-compile ModuleOp.
//
// `finalize` is `&&`-qualified — you have to `std::move(mb)` to call it, which
// makes accidental post-finalize use a compile error rather than a silent bug.
class ModuleBuilder {
public:
    static ModuleBuilder init(llvm::ArrayRef<TensorTypeSpec> inputs,
                              llvm::ArrayRef<mlir::tt::ttcore::ArgumentType> arg_types = {});

    ModuleBuilder(const ModuleBuilder &) = delete;
    ModuleBuilder &operator=(const ModuleBuilder &) = delete;
    ModuleBuilder(ModuleBuilder &&) = default;
    ModuleBuilder &operator=(ModuleBuilder &&) = default;

    // Emit an op at the current insertion point; loc is auto-supplied.
    template <typename Op, typename... Args> auto create(Args &&...args) {
        return builder_.create<Op>(loc_, std::forward<Args>(args)...);
    }

    // Surface for attribute construction (getDenseI32ArrayAttr, getBoolAttr,
    // etc.). Returns the base `mlir::Builder` rather than `OpBuilder` so
    // callers can't bypass `create<>` and lose loc threading.
    mlir::Builder &attrs() { return builder_; }

    // Emit a `ttir.typecast` from `value` to `target` element type, or return
    // `value` unchanged if already matching. Safe to call unconditionally.
    mlir::Value insert_typecast(mlir::Value value, mlir::Type target);

    llvm::ArrayRef<mlir::Value> args() const { return args_; }
    mlir::Location loc() const { return loc_; }

    mlir::OwningOpRef<mlir::ModuleOp> finalize(llvm::ArrayRef<mlir::Value> outputs) &&;

private:
    ModuleBuilder(mlir::OwningOpRef<mlir::ModuleOp> module_op, mlir::func::FuncOp func, mlir::OpBuilder builder,
                  mlir::Location loc, llvm::SmallVector<mlir::Value> args);

    mlir::OwningOpRef<mlir::ModuleOp> module_op_;
    mlir::func::FuncOp func_;
    mlir::OpBuilder builder_;
    mlir::Location loc_;
    llvm::SmallVector<mlir::Value> args_;
};

// Spec for a TTIR function input, carrying the tensor's logical dtype.
// The TTIR→TTNN rewriter demotes unsupported wide types (f64, i64, ...)
// to their hardware alias at lowering time.
TensorTypeSpec spec_for(const at::Tensor &t);

// Torch scalar type → MLIR element type via the logical runtime dtype. May
// return a type the hardware doesn't support directly; the rewriter handles it.
mlir::Type mlir_element_type_for(c10::ScalarType torch_dtype);

// Kernel-side convenience for native binary/ternary ops. Computes the
// PyTorch-promoted dtype across `tensors` (using `at::result_type` semantics —
// wrapped-scalar handling, etc.) and emits a `ttir.typecast` for each builder
// arg that doesn't already match. Returns the promoted dtype followed by the
// cast values, for structured binding:
//
//     auto [promoted, lhs, rhs] = promote_inputs(mb, a, b);
//
// `tensors` must be in the same order they were passed to `ModuleBuilder::init`.
template <typename... Tensors> auto promote_inputs(ModuleBuilder &mb, const Tensors &...tensors) {
    static_assert(sizeof...(Tensors) > 0, "promote_inputs: at least one input required");
    static_assert((std::is_same_v<std::remove_cvref_t<Tensors>, at::Tensor> && ...),
                  "promote_inputs: all arguments must be at::Tensor");

    at::native::ResultTypeState state{};
    ((state = at::native::update_result_type_state(tensors, state)), ...);
    const c10::ScalarType promoted = at::native::result_type(state);
    const auto promoted_mlir = mlir_element_type_for(promoted);

    auto args = mb.args();
    TORCH_CHECK(args.size() == sizeof...(Tensors), "promote_inputs: ModuleBuilder has ", args.size(),
                " arg(s), expected ", sizeof...(Tensors));

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::tuple{promoted, mb.insert_typecast(args[Is], promoted_mlir)...};
    }(std::make_index_sequence<sizeof...(Tensors)>{});
}

} // namespace tt::kurbla::torch_backend
