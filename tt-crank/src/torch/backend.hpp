// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "ttmlir/Target/Common/types_generated.h"
#include "llvm/ADT/ArrayRef.h"
#include <ATen/core/ScalarType.h>
#include <ATen/core/Tensor.h>
#include <tt/runtime/types.h>

#include "engine/compile.hpp"

namespace tt::kurbla::torch_backend {

// Compile a finalized TTIR module through the engine pipeline, execute it with
// `inputs`, and return the raw runtime output tensors. Callers wrap the result
// into at::Tensor via `wrap_tt_tensor` — they know the user-facing shape/dtype
// (e.g. the pre-demotion logical type that the runtime descriptor doesn't
// preserve), no parameter plumbing needed here.
//
// On a cache miss (currently: every call) this is O(compile). Each ATen kernel
// finalizes its own module and calls this once.
std::vector<::tt::runtime::Tensor> compile_and_run(mlir::OwningOpRef<mlir::ModuleOp> module_op,
                                                   llvm::ArrayRef<at::Tensor> inputs);

// Compile a finalized TTIR module into a `CompiledProgram` that can back
// many `run_compiled_program` calls — compile once, bind-and-run many.
::tt::kurbla::CompileResult compile_module(mlir::OwningOpRef<mlir::ModuleOp> module_op,
                                           const ::tt::kurbla::CompileOptions &options = {}, bool capture_ttir = false);

// Bind `inputs` to an already-compiled program, execute it, and wrap the
// device-resident outputs as tt-backend `at::Tensor`s. Output shapes come
// from the program's output descriptors; output dtypes come from the
// `logical_output_dtypes` the caller provides — the caller knows the
// user-facing dtype (e.g. i64) which may differ from the program's physical
// dtype after the rewriter demotes wide types (e.g. i64 → i32).
std::vector<at::Tensor> run_compiled_program(::tt::kurbla::CompiledProgram &program, llvm::ArrayRef<at::Tensor> inputs,
                                             llvm::ArrayRef<::tt::target::DataType> logical_output_dtypes);

// Installs the PrivateUse1 allocator with c10. Idempotent; safe to call from
// the nanobind module init. Most allocations come through aten::empty
// (overridden in ops/tensor.cpp), which constructs TensorStorages directly;
// the allocator below is a defensive fallback for code paths that bypass that
// override (resize_, manual Storage construction, etc.).
void register_allocator();

// Translate dtypes between torch's `c10::ScalarType` and tt-runtime's
// `tt::target::DataType`.
::tt::target::DataType to_runtime_dtype(c10::ScalarType torch_dtype);
c10::ScalarType to_torch_dtype(::tt::target::DataType runtime_dtype);

// Byte width of a runtime dtype's element. Mirrors c10::elementSize for the
// torch side; we keep both APIs symmetrical so allocation/copy code doesn't
// have to translate to torch and back.
std::size_t element_size(::tt::target::DataType runtime_dtype);

} // namespace tt::kurbla::torch_backend
