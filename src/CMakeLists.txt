
file(GLOB CXX_SRC *.cpp)

if (ENABLE_CUDA)
    file(GLOB CU_SRC *.cu)
    set_source_files_properties(Camera.cpp Light.cpp PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

file(GLOB SHADERS "shaders/*/*.glsl")
source_group("shaders" FILES SHADERS)

add_executable(graphx ${CXX_SRC} ${CXX_CU_SRC} ${CU_SRC})
add_dependencies(graphx assimp)
if (ENABLE_CUDA)
    set_property(TARGET graphx PROPERTY CUDA_STANDARD 11)
    set_property(TARGET graphx PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET graphx PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(ENABLE_CUDA)

target_link_libraries(graphx ${LINK_LIBS})
add_dependencies(graphx copy_shader_files copy_benchmark_script)
