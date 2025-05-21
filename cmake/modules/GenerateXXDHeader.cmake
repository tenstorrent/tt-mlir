# Function to generate C++ header files from input header files using xxd
function(generate_xxd_header INPUT_FILE OUTPUT_FILE VARIABLE_NAME)
  # Create the output directory if it doesn't exist
  get_filename_component(OUTPUT_DIR ${OUTPUT_FILE} DIRECTORY)
  file(MAKE_DIRECTORY ${OUTPUT_DIR})

  # Get relative path to simplify variable name
  file(RELATIVE_PATH REL_INPUT_FILE ${CMAKE_CURRENT_SOURCE_DIR} ${INPUT_FILE})

  add_custom_command(
    OUTPUT ${OUTPUT_FILE}
    COMMAND xxd -i ${REL_INPUT_FILE} > ${OUTPUT_FILE}
    DEPENDS ${INPUT_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating xxd header for ${REL_INPUT_FILE}"
    VERBATIM
  )
endfunction()
