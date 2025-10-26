find_program(FLATBUFFERS_COMPILER flatc)

function(build_flatbuffers namespace sources target)
# Query deps if they exist
set(deps)
# Query extra include dirs if they exist
set(extra_include_dirs)

if(ARGC GREATER 3)
  foreach(arg IN LISTS ARGV3)
    if(TARGET ${arg})
      get_property(target_sources TARGET ${arg} PROPERTY SOURCES)
      list(APPEND deps ${target_sources})
    elseif(IS_DIRECTORY ${arg})
      list(APPEND extra_include_dirs ${arg})
    else()
      message(FATAL_ERROR "Invalid argument: ${arg}")
    endif()
  endforeach()
endif()

# Build include args command line arguments
set(INCLUDE_ARGS)
list(APPEND INCLUDE_ARGS ARGS -I ${PROJECT_SOURCE_DIR}/include/)
foreach(extra_include_dir IN LISTS extra_include_dirs)
  list(APPEND INCLUDE_ARGS ARGS -I ${extra_include_dir})
endforeach()

set(FBS_GEN_OUTPUTS)

foreach(FILE ${sources})
  get_filename_component(FILE_DIR ${FILE} DIRECTORY)
  get_filename_component(BASE_NAME ${FILE} NAME_WE)
  set(FBS_GEN_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FILE_DIR}/${BASE_NAME}_generated.h")
  set(FBS_GEN_BFBS_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FILE_DIR}/${BASE_NAME}_bfbs_generated.h")
  add_custom_command(OUTPUT
    "${FBS_GEN_FILE}"
    "${FBS_GEN_BFBS_FILE}"
    COMMAND ${FLATBUFFERS_COMPILER}
    ${INCLUDE_ARGS}
    ARGS --bfbs-gen-embed
    ARGS --cpp --cpp-std c++17
    ARGS --scoped-enums --warnings-as-errors
    ARGS --keep-prefix
    ARGS --gen-name-strings
    ARGS -o "${CMAKE_CURRENT_BINARY_DIR}/${FILE_DIR}/" "${FILE}"
    COMMENT "Generating flatbuffer file: ${FBS_GEN_FILE}"
    DEPENDS ${FILE} ${deps} ${sources}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  list(APPEND FBS_GEN_OUTPUTS ${FBS_GEN_FILE})
  list(APPEND FBS_GEN_OUTPUTS ${FBS_GEN_BFBS_FILE})

  # Compute sha256 hash of the schema (via the bfbs file itself)
  set(FBS_GEN_PYTHON_SHA256_SCRIPT "${PROJECT_SOURCE_DIR}/tools/scripts/sha256-include-gen.py")
  set(FBS_GEN_BFBS_HASH_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FILE_DIR}/${BASE_NAME}_bfbs_hash_generated.h")
  set(FBS_GEN_BFBS_NAMESPACE "tt::target::${namespace}")
  set(FBS_GEN_BFBS_VARIABLE_NAME "${BASE_NAME}_bfbs_schema_hash")

  add_custom_command(OUTPUT
    "${FBS_GEN_BFBS_HASH_FILE}"
    COMMAND ${Python3_EXECUTABLE} "${FBS_GEN_PYTHON_SHA256_SCRIPT}" ${FBS_GEN_BFBS_NAMESPACE} ${FBS_GEN_BFBS_VARIABLE_NAME} ${FBS_GEN_BFBS_FILE} ${FBS_GEN_BFBS_HASH_FILE}
    DEPENDS ${FBS_GEN_BFBS_FILE}
    COMMENT "Generating schema hash file: ${FBS_GEN_BFBS_HASH_FILE}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM
  )
  list(APPEND FBS_GEN_OUTPUTS ${FBS_GEN_BFBS_HASH_FILE})

endforeach()
add_library(${target} INTERFACE ${FBS_GEN_OUTPUTS})
endfunction()
