option(TTMLIR_ENABLE_STABLEHLO "Enable StableHLO support" OFF)

include(tt-metal.cmake)
if (TTMLIR_ENABLE_STABLEHLO)
include(stablehlo.cmake)
endif()
