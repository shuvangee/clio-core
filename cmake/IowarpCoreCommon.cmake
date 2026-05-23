# IowarpCoreCommon.cmake — backward-compat forwarder.
#
# This file was renamed to ClioCoreCommon.cmake in the iowarp -> clio
# rebranding pass. Out-of-tree CMake code that still does
#   include(IowarpCoreCommon)
# or
#   include("${CMAKE_CURRENT_LIST_DIR}/IowarpCoreCommon.cmake")
# keeps working through this one-line forwarder. New code should
# include ClioCoreCommon directly.
include("${CMAKE_CURRENT_LIST_DIR}/ClioCoreCommon.cmake")
