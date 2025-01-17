
if (TTMLIR_ENABLE_STABLEHLO)
  set(STABLEHLO_VERSION "d40285ef3db0687e3f1e2bb0d716d748485a9739")
  set(SHARDY_VERSION "c8da53a9e0826d1dbb6947de306dce3034d0de83")

  ## StableHLO
  CPMAddPackage(
      NAME stablehlo
      GITHUB_REPOSITORY openxla/stablehlo
      GIT_TAG ${STABLEHLO_VERSION}
  )

  if(stablehlo_ADDED)
    set(STABLEHLO_BINARY_DIR ${stablehlo_BINARY_DIR} PARENT_SCOPE)
    set(STABLEHLO_SOURCE_DIR ${stablehlo_SOURCE_DIR} PARENT_SCOPE)
  endif()

  ## Shardy
  CPMAddPackage(
      NAME shardy
      GITHUB_REPOSITORY openxla/shardy
      GIT_TAG ${SHARDY_VERSION}
      PATCHES ${PROJECT_SOURCE_DIR}/dependencies/shardy.patch
  )

  if(shardy_ADDED)
    set(SHARDY_BINARY_DIR ${shardy_BINARY_DIR} PARENT_SCOPE)
    set(SHARDY_SOURCE_DIR ${shardy_SOURCE_DIR} PARENT_SCOPE)
  endif()
endif()
