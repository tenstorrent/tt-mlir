file(REMOVE_RECURSE
  "../../../libMLIRTTDialect.a"
  "../../../libMLIRTTDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
