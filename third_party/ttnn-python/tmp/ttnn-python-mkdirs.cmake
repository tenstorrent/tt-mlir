# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src/ttnn-python")
  file(MAKE_DIRECTORY "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src/ttnn-python")
endif()
file(MAKE_DIRECTORY
  "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src/ttnn-python/build"
  "/localdev/sgholami/tt-mlir/third_party/ttnn-python"
  "/localdev/sgholami/tt-mlir/third_party/ttnn-python/tmp"
  "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src/ttnn-python-stamp"
  "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src"
  "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src/ttnn-python-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src/ttnn-python-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/localdev/sgholami/tt-mlir/third_party/ttnn-python/src/ttnn-python-stamp${cfgdir}") # cfgdir has leading slash
endif()
