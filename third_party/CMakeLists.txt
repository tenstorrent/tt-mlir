include(ExternalProject)

include(tt-metal.cmake)
if (TTMLIR_ENABLE_TORCH_MLIR)
  include(torch-mlir.cmake)
endif ()
