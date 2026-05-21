# util/CMakeLists.txt
# Utility executables for Chimaera runtime

cmake_minimum_required(VERSION 3.10)

#==============================================================================
# Shared command library used by unified chimaera binary and legacy wrappers
#==============================================================================
add_library(chimaera_commands STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_runtime_start.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_runtime_stop.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_monitor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_compose.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_migrate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_cmd_refresh_repo.cc
)
target_link_libraries(chimaera_commands
  PUBLIC chimaera_cxx
  PUBLIC chimaera_admin_client
)
target_include_directories(chimaera_commands
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

#==============================================================================
# Unified chimaera binary
#==============================================================================
add_executable(chimaera
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera.cc
)
target_link_libraries(chimaera
  PUBLIC chimaera_commands
)

# Install the binary under its canonical legacy name `chimaera` AND under
# the new brand-aligned name `clio_run`. Both invocations work; docs and
# new scripts should prefer `clio_run`. See rebranding.md.
install(TARGETS chimaera
  RUNTIME DESTINATION bin
)

# Install `clio_run` as a symlink to `chimaera` in the same bin dir.
# Done at install time (not build time) so it lands at ${CMAKE_INSTALL_PREFIX}/bin
# regardless of CPACK/DESTDIR. On Windows the symlink falls back to a copy.
install(CODE "
  set(_src \"\${CMAKE_INSTALL_PREFIX}/bin/chimaera\")
  set(_dst \"\${CMAKE_INSTALL_PREFIX}/bin/clio_run\")
  if(EXISTS \"\${_src}\")
    if(WIN32)
      file(INSTALL \"\${_src}\" DESTINATION \"\${CMAKE_INSTALL_PREFIX}/bin\"
           RENAME clio_run.exe)
    else()
      execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink chimaera \"\${_dst}\")
    endif()
    message(STATUS \"Installed clio_run -> chimaera symlink at \${_dst}\")
  endif()
")
