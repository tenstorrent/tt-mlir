file(REMOVE_RECURSE
  "../libTTMLIRCAPI.a"
  "../libTTMLIRCAPI.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/TTMLIRCAPI.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
