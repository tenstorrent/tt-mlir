# Function to generate C++ header files as strings from input header files
function(generate_raw_string_header INPUT_FILE OUTPUT_FILE VARIABLE_NAME)
  # Ensure output directory exists
  get_filename_component(OUTPUT_DIR ${OUTPUT_FILE} DIRECTORY)
  file(MAKE_DIRECTORY ${OUTPUT_DIR})

  # Read the ASCII content of the input file
  file(READ "${INPUT_FILE}" FILE_CONTENT)

  # Use a unique raw string delimiter that is unlikely to conflict with content
  set(DELIM "TTMLIR_STR_DELIM")

  # Write the header file content
  file(WRITE "${OUTPUT_FILE}" "// Auto-generated from ${INPUT_FILE} - Do not edit directly\n")
  file(APPEND "${OUTPUT_FILE}" "static constexpr char ${VARIABLE_NAME}[] = R\"${DELIM}(\n${FILE_CONTENT})${DELIM}\";\n")
  file(APPEND "${OUTPUT_FILE}" "static constexpr unsigned int ${VARIABLE_NAME}_len = sizeof(${VARIABLE_NAME}) - 1;\n")
endfunction()
