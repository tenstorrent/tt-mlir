execute_process(
  COMMAND python -c "from distutils import sysconfig; print(sysconfig.get_python_lib(prefix='', plat_specific=True))"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(TTMLIR_PYTHON_SITE_PACKAGES "${TTMLIR_TOOLCHAIN}/venv/${PYTHON_SITE_PACKAGES}" CACHE STRING "Path to the Python site-packages directory")
