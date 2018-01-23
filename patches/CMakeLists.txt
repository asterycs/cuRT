cmake_minimum_required(VERSION 3.10)
project(imgui)

set(GLFW_INCLUDE_DIRS "" CACHE BOOL "GLFW include dir")

if ("${GLFW_INCLUDE_DIRS}" STREQUAL "")
    message(FATAL_ERROR "GLFW_INCLUDE_DIRS not set.")
endif()

file(GLOB IMGUI_SRC imgui.cpp
                    imgui_draw.cpp
                    imgui_demo.cpp
                    examples/opengl3_example/imgui_impl_glfw_gl3.cpp)

add_library(imgui STATIC ${IMGUI_SRC})

target_include_directories(imgui PRIVATE . examples/opengl3_example/ "${GLFW_INCLUDE_DIRS}")

install(FILES ${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}imgui${CMAKE_STATIC_LIBRARY_SUFFIX}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/lib/)

install(FILES ${CMAKE_SOURCE_DIR}/imgui.h ${CMAKE_SOURCE_DIR}/imconfig.h ${CMAKE_SOURCE_DIR}/examples/opengl3_example/imgui_impl_glfw_gl3.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATAROOTDIR}/include/)
