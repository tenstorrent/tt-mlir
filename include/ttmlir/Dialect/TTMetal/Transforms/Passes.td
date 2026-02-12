include "mlir/Pass/Pass.td"

define_pass_pipeline("ttmetal-bufferize", "TTMetal Bufferization Pass") {
  pass("ttmetal-bufferize-pass", "TTMetal Bufferize Pass")
}

def TTMetalBufferizePass : Pass<TTMetalBufferizePass> {
  let summary = "Bufferizes TTMetal global semaphore creation to memref allocation.";
  let description = "Converts ttmetal.create_global_semaphore operations into memref.alloc and memref.store operations.";
  let dependentDialects = ["mlir::memref::MemRefDialect", "mlir::arith::ArithDialect"];
}
