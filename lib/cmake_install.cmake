#Install script for directory : / home / vwells / sources / tt - mlir / lib

#Set the install prefix
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
set(CMAKE_INSTALL_PREFIX "/usr/local") endif() string(REGEX REPLACE
                                                      "/$"
                                                      "" CMAKE_INSTALL_PREFIX
                                                      "${CMAKE_INSTALL_PREFIX}")

#Set the install configuration name.
    if (NOT DEFINED CMAKE_INSTALL_CONFIG_NAME) if (BUILD_TYPE) string(
        REGEX REPLACE
        "^[^A-Za-z0-9_]+"
        "" CMAKE_INSTALL_CONFIG_NAME
        "${BUILD_TYPE}") else() set(CMAKE_INSTALL_CONFIG_NAME
                                    "") endif() message(STATUS
                                                        "Install "
                                                        "configuration: "
                                                        "\"${CMAKE_INSTALL_"
                                                        "CONFIG_NAME}\"") endif()

#Set the component getting installed.
        if (NOT CMAKE_INSTALL_COMPONENT) if (COMPONENT) message(STATUS "Install"
                                                                       " compon"
                                                                       "ent: "
                                                                       "\"${"
                                                                       "COMPONE"
                                                                       "NT}\"") set(
            CMAKE_INSTALL_COMPONENT
            "${COMPONENT}") else() set(CMAKE_INSTALL_COMPONENT) endif() endif()

#Install shared libraries without execute permission ?
            if (NOT DEFINED CMAKE_INSTALL_SO_NO_EXE) set(CMAKE_INSTALL_SO_NO_EXE
                                                         "1") endif()

#Is this installation the result of a crosscompile ?
                if (NOT DEFINED CMAKE_CROSSCOMPILING) set(CMAKE_CROSSCOMPILING
                                                          "FALSE") endif()

#Set path to fallback - tool for dependency - resolution.
                    if (NOT DEFINED CMAKE_OBJDUMP) set(
                        CMAKE_OBJDUMP "/usr/bin/objdump") endif()

                        if (NOT CMAKE_INSTALL_LOCAL_ONLY)
#Include the install script for the subdirectory.
                            include("/home/vwells/sources/tt-mlir/lib/CAPI/"
                                    "cmake_install.cmake") endif()

                                if (NOT CMAKE_INSTALL_LOCAL_ONLY)
#Include the install script for the subdirectory.
                                    include(
                                        "/home/vwells/sources/tt-mlir/lib/"
                                        "Conversion/cmake_install.cmake") endif()

                                        if (NOT CMAKE_INSTALL_LOCAL_ONLY)
#Include the install script for the subdirectory.
                                            include(
                                                "/home/vwells/sources/tt-mlir/"
                                                "lib/Dialect/"
                                                "cmake_install.cmake") endif()

                                                if (NOT CMAKE_INSTALL_LOCAL_ONLY)
#Include the install script for the subdirectory.
                                                    include("/home/vwells/"
                                                            "sources/tt-mlir/"
                                                            "lib/Target/"
                                                            "cmake_install."
                                                            "cmake") endif()

                                                        if (NOT CMAKE_INSTALL_LOCAL_ONLY)
#Include the install script for the subdirectory.
                                                            include(
                                                                "/home/vwells/"
                                                                "sources/"
                                                                "tt-mlir/lib/"
                                                                "Scheduler/"
                                                                "cmake_install."
                                                                "cmake") endif()

                                                                if (CMAKE_INSTALL_COMPONENT STREQUAL
                                                                    "TTMLIRStat"
                                                                    "ic" OR NOT
                                                                        CMAKE_INSTALL_COMPONENT)
                                                                    file(INSTALL DESTINATION
                                                                         "${"
                                                                         "CMAKE"
                                                                         "_INST"
                                                                         "ALL_"
                                                                         "PREFI"
                                                                         "X}/"
                                                                         "li"
                                                                         "b" TYPE
                                                                             STATIC_LIBRARY FILES
                                                                         "/home"
                                                                         "/vwel"
                                                                         "ls/"
                                                                         "sourc"
                                                                         "es/"
                                                                         "tt-"
                                                                         "mlir/"
                                                                         "lib/"
                                                                         "libTT"
                                                                         "MLIRS"
                                                                         "tatic"
                                                                         ".a") endif()

                                                                        string(
                                                                            REPLACE
                                                                            ";"
                                                                            "\n" CMAKE_INSTALL_MANIFEST_CONTENT
                                                                            "${"
                                                                            "CM"
                                                                            "AK"
                                                                            "E_"
                                                                            "IN"
                                                                            "ST"
                                                                            "AL"
                                                                            "L_"
                                                                            "MA"
                                                                            "NI"
                                                                            "FE"
                                                                            "ST"
                                                                            "_F"
                                                                            "IL"
                                                                            "ES"
                                                                            "}") if (CMAKE_INSTALL_LOCAL_ONLY)
                                                                            file(
                                                                                WRITE
                                                                                "/home/vwells/sources/tt-mlir/lib/install_local_manifest.txt"
                                                                                "${CMAKE_INSTALL_MANIFEST_CONTENT}")
                                                                                endif()
