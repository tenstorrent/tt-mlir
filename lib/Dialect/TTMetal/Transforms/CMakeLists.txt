add_mlir_dialect_pass_registration(
  TTMetalBufferize
  "TTMetal Bufferize Passes"
  TTMetalBufferizePass.cpp
  DEPENDS
  MLIRTTMetalDialect
)
