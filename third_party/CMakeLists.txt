include(ExternalProject)

include(tt-metal.cmake)
if (TTMLIR_ENABLE_STABLEHLO)
  include(stablehlo.cmake)
endif()
