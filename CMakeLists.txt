cmake_minimum_required(VERSION 3.10)

# Parameters
set(ENABLE_CUDA ON CACHE BOOL "Enable Cuda features") # Enable by default

# Set project language
if (ENABLE_CUDA)
    project(graphx LANGUAGES CXX CUDA)
    add_definitions(-DENABLE_CUDA)
else()
    project(graphx LANGUAGES CXX)
    message("CUDA features disabled")
endif(ENABLE_CUDA)

# Define some variables
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules/)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
include(ExternalProject)


# Pre-installed dependencies
if (ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
    list(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
    string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}") # Phew...
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -maxrregcount 48")
endif(ENABLE_CUDA)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Freetype REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK2 REQUIRED gtk+-3.0)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "CCache found")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

# Assimp
ExternalProject_Add(assimp
       PREFIX "${CMAKE_BINARY_DIR}/ext/assimp"
       GIT_REPOSITORY https://github.com/assimp/assimp.git
       #GIT_TAG "v4.0.1"
       GIT_PROGRESS 1
       UPDATE_DISCONNECTED 1
       INSTALL_DIR "${CMAKE_BINARY_DIR}/assimp"
       CMAKE_ARGS
           "-DASSIMP_BUILD_TESTS=OFF"
           "-DASSIMP_BUILD_ASSIMP_TOOLS=OFF"
           "-DBUILD_SHARED_LIBS=OFF" # Build static
           "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/assimp"
           "-DCMAKE_BUILD_TYPE=Release"
           "-DCMAKE_CXX_FLAGS=-fPIC" "-DCMAKE_C_FLAGS=-fPIC"
           )

set(ASSIMP_LIBRARIES "${CMAKE_BINARY_DIR}/assimp/lib/${CMAKE_STATIC_LIBRARY_PREFIX}assimp${CMAKE_STATIC_LIBRARY_SUFFIX};${CMAKE_BINARY_DIR}/assimp/lib/${CMAKE_STATIC_LIBRARY_PREFIX}IrrXML${CMAKE_STATIC_LIBRARY_SUFFIX};z;")
set(ASSIMP_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/assimp/include")

# Native file dialog
ExternalProject_Add(nativefiledialog
       PREFIX "${CMAKE_BINARY_DIR}/ext/nativefiledialog"
       GIT_REPOSITORY https://github.com/mlabbe/nativefiledialog.git
       GIT_PROGRESS 1
       UPDATE_DISCONNECTED 1
       CONFIGURE_COMMAND ""
       BUILD_COMMAND cd ${CMAKE_BINARY_DIR}/ext/nativefiledialog/src/nativefiledialog/build/gmake_linux && make
       INSTALL_COMMAND ""
       )

set(NATIVEFILEDIALOG_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/ext/nativefiledialog/src/nativefiledialog/src/include")
set(NATIVEFILEDIALOG_LIBRARIES "${CMAKE_BINARY_DIR}/ext/nativefiledialog/src/nativefiledialog/build/lib/Release/x64/${CMAKE_STATIC_LIBRARY_PREFIX}nfd${CMAKE_STATIC_LIBRARY_SUFFIX}")


# cxxopts
ExternalProject_Add(cxxopts
       PREFIX "${CMAKE_BINARY_DIR}/ext/cxxopts"
       GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
       GIT_TAG 1_0
       GIT_PROGRESS 1
       UPDATE_DISCONNECTED 1
       INSTALL_DIR "${CMAKE_BINARY_DIR}/cxxopts"
       CMAKE_ARGS
           "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/cxxopts"
           "-DCMAKE_BUILD_TYPE=Release"
       )

set(CXXOPTS_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/cxxopts/include")

# glm
ExternalProject_Add(
       glm
       PREFIX ${CMAKE_BINARY_DIR}/ext/glm
       GIT_REPOSITORY https://github.com/g-truc/glm.git
       GIT_TAG 0.9.9-a2
       GIT_PROGRESS 1
       UPDATE_DISCONNECTED 1
       UPDATE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND ""
       LOG_DOWNLOAD ON
       )

ExternalProject_Get_Property(glm SOURCE_DIR)
set(GLM_INCLUDE_DIRS ${SOURCE_DIR})

include_directories(
    ${ASSIMP_INCLUDE_DIRS}
    ${NATIVEFILEDIALOG_INCLUDE_DIRS}
    ${GTK2_INCLUDE_DIRS}
    ${IL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${FREETYPE_INCLUDE_DIRS}
    ${CXXOPTS_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
)


set(LINK_LIBS
    glfw
    GLEW
    GL
    ${ASSIMP_LIBRARIES}
    ${NATIVEFILEDIALOG_LIBRARIES}
    ${GTK2_LIBRARIES}
    ${IL_LIBRARIES}
    ${ILUT_LIBRARIES}
    ${FREETYPE_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    )

add_custom_target(
        copy_shader_files
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/shaders ${CMAKE_BINARY_DIR}/src/shaders/
        COMMENT "Copying shader files for target: graphx" VERBATIM 
        )

add_custom_target(
        copy_benchmark_script
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/benchmark.sh ${CMAKE_BINARY_DIR}/src/
        COMMENT "Copying benchmark script for target: graphx" VERBATIM 
        )


add_subdirectory(src)


#set_source_files_properties(${CXX_SRC} PROPERTIES COMPILE_FLAGS "\
#-Wall \
#-Wextra \light.getSize()
#-Wcast-align \
#-Wcast-qual \
#-Wctor-dtor-privacy \
#-Wdisabled-optimization \
#-Wformat=2 \
#-Winit-self \
#-Wlogical-op \
#-Wmissing-declarations \
#-Wmissing-include-dirs \
#-Wnoexcept \
#-Wold-style-cast \
#-Woverloaded-virtual \
#-Wredundant-decls \
#-Wshadow \
#-Wsign-conversion \
#-Wsign-promo \
#-Wstrict-null-sentinel \
#-Wstrict-overflow=5 \
#-Wswitch-default \
#-Wundef \
#-Werror \
#-Wno-unused")

