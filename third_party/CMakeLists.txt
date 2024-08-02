include(ExternalProject)

include(tt-metal.cmake)
if (TTMLIR_ENABLE_ONNX_MLIR)
  include(onnx-mlir.cmake)
endif ()
