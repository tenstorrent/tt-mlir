# Install ttnn test headers  
install(
  FILES
    dylib.h
    utils.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime/test/ttnn
  COMPONENT TTNNStandalone
)