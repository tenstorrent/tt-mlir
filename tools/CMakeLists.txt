add_subdirectory(ttmlir-opt)
add_subdirectory(ttmlir-lsp-server)
add_subdirectory(ttmlir-translate)
if (TTMLIR_ENABLE_EXPLORER)
  add_subdirectory(explorer)
endif()
add_subdirectory(ttnn-standalone EXCLUDE_FROM_ALL)

install(
  FILES
    ttnn-standalone/ci_compile_dylib.py
    ttnn-standalone/CMakeLists.txt
    ttnn-standalone/compile_so.cpp
    ttnn-standalone/compile_so.hpp
    ttnn-standalone/README.md
    ttnn-standalone/ttnn-precompiled.hpp
    ttnn-standalone/ttnn-standalone.cpp
    ttnn-standalone/workarounds.hpp
  DESTINATION "tools/ttnn-standalone"
  COMPONENT TTNNStandalone
)

install(
  PROGRAMS
    ttnn-standalone/run
  DESTINATION "tools/ttnn-standalone"
  COMPONENT TTNNStandalone
)
