
file(GLOB CXX_SRC *.cpp)

if (ENABLE_CUDA)
    file(GLOB CU_SRC *.cu)
    set_source_files_properties(Camera.cpp Light.cpp PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

file(GLOB SHADERS "shaders/*/*.glsl")
source_group("shaders" FILES SHADERS)

add_executable(cuRT ${CXX_SRC} ${CXX_CU_SRC} ${CU_SRC})
add_dependencies(cuRT assimp glfw glm cxxopts nativefiledialog)

target_link_libraries(cuRT ${LINK_LIBS})
add_dependencies(cuRT copy_shader_files copy_benchmark_script)

if (ENABLE_CUDA)
    set_target_properties(cuRT PROPERTIES CUDA_STANDARD 14 CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(ENABLE_CUDA)

#set_target_properties(cuRT
#    PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
#)

add_custom_target(
        copy_shader_files
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/shaders ${CMAKE_BINARY_DIR}/src/shaders/
        COMMENT "Copying shader files for target: cuRT" VERBATIM 
        )

add_custom_target(
        copy_benchmark_script
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/benchmark.sh ${CMAKE_BINARY_DIR}/src
        COMMENT "Copying benchmark script for target: cuRT" VERBATIM 
        )

