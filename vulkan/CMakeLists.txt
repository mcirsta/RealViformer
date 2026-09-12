cmake_minimum_required(VERSION 3.22)

project(realviformer_vulkan LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

set(AXPY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/axpy.comp")
set(AXPY_BINARY "${CMAKE_CURRENT_BINARY_DIR}/shaders/axpy.comp.spv")

add_custom_command(
    OUTPUT "${AXPY_BINARY}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND "${GLSLC_EXECUTABLE}" -O "${AXPY_SOURCE}" -o "${AXPY_BINARY}"
    DEPENDS "${AXPY_SOURCE}"
    VERBATIM
)

add_custom_target(realviformer_vulkan_shaders DEPENDS "${AXPY_BINARY}")

add_executable(rvf-vulkan src/main.cpp)
add_dependencies(rvf-vulkan realviformer_vulkan_shaders)
target_compile_features(rvf-vulkan PRIVATE cxx_std_20)
target_compile_options(
    rvf-vulkan PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers
)
target_compile_definitions(
    rvf-vulkan PRIVATE RVF_AXPY_SHADER_PATH="${AXPY_BINARY}"
)
target_link_libraries(rvf-vulkan PRIVATE Vulkan::Vulkan)
