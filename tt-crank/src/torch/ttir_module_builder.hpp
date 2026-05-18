#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <tt/runtime/types.h>

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
    static ModuleBuilder init(llvm::ArrayRef<TensorTypeSpec> inputs);

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

} // namespace tt::kurbla::torch_backend
