add_subdirectory(ttmlir)

install(DIRECTORY ttmlir ttmlir-c ${TTMLIR_BINARY_DIR}/include/ttmlir
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT SharedLib
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
)
