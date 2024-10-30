# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src/model-explorer")
  file(MAKE_DIRECTORY "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src/model-explorer")
endif()
file(MAKE_DIRECTORY
  "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src/model-explorer-build"
  "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer"
  "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/tmp"
  "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src/model-explorer-stamp"
  "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src"
  "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src/model-explorer-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src/model-explorer-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/localdev/odjuricic/repos/tt-mlir/tools/explorer/model-explorer/src/model-explorer-stamp${cfgdir}") # cfgdir has leading slash
endif()
