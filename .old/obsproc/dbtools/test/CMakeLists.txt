execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink
           ${CMAKE_CURRENT_SOURCE_DIR}/obs-2-move
           ${CMAKE_CURRENT_BINARY_DIR}/obs-2-move )

add_test ( emc_obsdb0 ${CMAKE_CURRENT_SOURCE_DIR}/../test/test.sh )
