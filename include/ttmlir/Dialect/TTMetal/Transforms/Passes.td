// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTMETAL_TTMETALPASSES_TD

include "mlir/Pass/PassBase.td"

def TTMetalGenerateDimAlignedHostIO: Pass<"ttmetal-generate-dim-aligned-host-io", "::mlir::ModuleOp"> {
  let summary = "Generate dim-aligned host tensor I/O for TTMetal enqueue ops.";
  let description = [{
    Ensures host buffers used by ttmetal.enqueue_{write,read}_buffer are
    dimension-aligned to the device's expected padded shapes. The pass looks
    for enqueue read/write ops that carry a 'host_layout' MetalLayout attribute
    (attached during ttir-to-ttmetal lowering), derives per-dimension
    alignments from it, and then:

    - For writes: allocates a new host memref with HostLayout (using the
      logical host shape and dim_alignments), copies from the user buffer into
      the aligned allocation, and uses that as the enqueue input.
    - For reads: allocates a new aligned host memref, uses it as the enqueue
      output, and copies from it into the user buffer after the enqueue.

    The 'host_layout' attribute is removed after materialization. This pass is
    required to support unaligned logical shapes end-to-end while keeping the
    user-facing host types unchanged.
  }];
  let dependentDialects = ["mlir::tt::ttmetal::TTMetalDialect", "mlir::memref::MemRefDialect"];
}

#endif
