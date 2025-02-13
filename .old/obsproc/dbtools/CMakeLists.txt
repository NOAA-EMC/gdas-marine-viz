################################################################################
# EMC_OBSDB
################################################################################

cmake_minimum_required( VERSION 3.12 )
project( emc_obsdb VERSION 1.0.0 ) #LANGUAGES CXX )

find_package(ecbuild 3.3.2 REQUIRED)
include( ecbuild_system NO_POLICY_SCOPE )
ecbuild_declare_project()

################################################################################
# Sources
################################################################################

add_subdirectory( src )
add_subdirectory( test )
