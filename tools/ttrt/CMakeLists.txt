include(AddMLIRPython)

set(TTRT_SOURCES)
file(GLOB_RECURSE TTRT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach (filename ${TTRT_FILES})
    cmake_path(RELATIVE_PATH filename BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE relative_filename)
    configure_file("${filename}" "${CMAKE_CURRENT_BINARY_DIR}/${relative_filename}" COPYONLY)
    list(APPEND TTRT_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${relative_filename}")
endforeach (filename)

set(CUDA_PATH_VAR "")
set(CUDART_PATH_VAR "")
if(TTMLIR_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    get_target_property(CUDA_CUDART_LIB CUDA::cudart IMPORTED_LOCATION)
    get_target_property(CUDA_CUDA_DRIVER_LIB CUDA::cuda_driver IMPORTED_LOCATION)
    if(CUDA_CUDART_LIB)
        get_filename_component(CUDART_PATH_VAR "${CUDA_CUDART_LIB}" DIRECTORY)
    endif()
    if(CUDA_CUDA_DRIVER_LIB)
        get_filename_component(CUDA_PATH_VAR "${CUDA_CUDA_DRIVER_LIB}" DIRECTORY)
    else()
        if(CUDART_PATH_VAR)
            set(STUBS_DIR "${CUDART_PATH_VAR}/stubs")
            if(EXISTS "${STUBS_DIR}/libcuda.so")
                set(CUDA_PATH_VAR "${STUBS_DIR}")
            endif()
        endif()
    endif()
endif()

add_custom_command(
  COMMAND rm -rf build
  COMMAND python -m pip install -r "${CMAKE_CURRENT_BINARY_DIR}/requirements.txt"
  COMMAND TTMLIR_ENABLE_RUNTIME=${TTMLIR_ENABLE_RUNTIME}
          TT_RUNTIME_ENABLE_TTNN=${TT_RUNTIME_ENABLE_TTNN}
          TT_RUNTIME_ENABLE_TTMETAL=${TT_RUNTIME_ENABLE_TTMETAL}
          TTMLIR_ENABLE_CUDA=${TTMLIR_ENABLE_CUDA}
          CUDA_PATH="${CUDA_PATH_VAR}"
          CUDART_PATH="${CUDART_PATH_VAR}"
          TTMLIR_ENABLE_RUNTIME_TESTS=${TTMLIR_ENABLE_RUNTIME_TESTS}
          TT_RUNTIME_ENABLE_PERF_TRACE=${TT_RUNTIME_ENABLE_PERF_TRACE}
          TT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          TTMLIR_BINARY_DIR=${TTMLIR_BINARY_DIR}
          TTMLIR_VERSION_MAJOR=${TTMLIR_VERSION_MAJOR}
          TTMLIR_VERSION_MINOR=${TTMLIR_VERSION_MINOR}
          TTMLIR_VERSION_PATCH=${TTMLIR_VERSION_PATCH}
          CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
          SOURCE_ROOT=${TTMLIR_SOURCE_DIR}
          python -m pip wheel . --wheel-dir build
  COMMAND python -m pip install build/*.whl --force-reinstall
  COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS ${TTRT_SOURCES} ${TTRT_FILES} TTMLIRRuntime TTRuntimeTTNNTypes TTRuntimeTTNNUtils _ttmlir_runtime
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/build/.installed
)

add_custom_target(ttrt ALL
  COMMENT "python ttrt package"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/build/.installed)

declare_mlir_python_sources(TTRTSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    __init__.py
    library_tweaks.py

    # binary subpackage files
    binary/__init__.py
    binary/stats.py

    # common subpackage files
    common/__init__.py
    common/api.py
    common/callback.py
    common/check.py
    common/perf.py
    common/query.py
    common/read.py
    common/run.py
    common/util.py
    common/emitpy.py

    # runtime subpackage files
    runtime/__init__.py
)

add_mlir_python_modules(TTRTPythonModules
  ROOT_PREFIX "${TTMLIR_PYTHON_PACKAGES_DIR}/ttrt"
  INSTALL_PREFIX "python_packages/ttrt"
  DECLARED_SOURCES TTRTSources
)
