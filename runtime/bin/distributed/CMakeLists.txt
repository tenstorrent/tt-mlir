if (NOT DISTRIBUTED_RUNTIME_ENABLED)
  return()
endif()

#
# Runtime Worker Executable
#

add_executable(worker worker.cpp)

set_target_properties(worker PROPERTIES CXX_STANDARD 20)

target_include_directories(worker PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(worker SYSTEM PRIVATE "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

add_dependencies(worker RUNTIME_FBS_GENERATION)

target_link_libraries(worker PRIVATE
  TTMLIRRuntime
  TTRuntimeDistributedWorker
  ${METAL_LIBS}
)

# All libraries are installed under ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
# Therefore we need to point the install rpath to the library directory
set_target_properties(worker PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

install(
  TARGETS worker
  DESTINATION ${CMAKE_INSTALL_BINDIR}/ttmlir/runtime/distributed
  COMPONENT DistributedRuntime
  EXCLUDE_FROM_ALL
)
