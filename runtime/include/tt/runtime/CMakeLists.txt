add_subdirectory(detail)

# Install runtime headers
install(
  FILES
    types.h
    runtime.h
    utils.h
    workarounds.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime
  COMPONENT TTNNStandalone
)

# Install test headers subdirectory
add_subdirectory(test)
