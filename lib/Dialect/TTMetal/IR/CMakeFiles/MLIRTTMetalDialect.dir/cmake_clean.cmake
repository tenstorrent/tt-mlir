file(REMOVE_RECURSE
  "../../../libMLIRTTMetalDialect.a"
  "../../../libMLIRTTMetalDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTMetalDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
