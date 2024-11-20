file(REMOVE_RECURSE
  "libTTMLIRStatic.a"
  "libTTMLIRStatic.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TTMLIRStatic.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
