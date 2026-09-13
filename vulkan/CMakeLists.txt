cmake_minimum_required(VERSION 3.22)

project(realviformer_vulkan LANGUAGES CXX)

option(RVF_USE_SYSTEM_NCNN "Use an installed ncnn package instead of the Git submodule" OFF)

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

add_executable(rvf-vulkan src/main.cpp src/weights.cpp)
add_dependencies(rvf-vulkan realviformer_vulkan_shaders)
target_compile_features(rvf-vulkan PRIVATE cxx_std_20)
target_compile_options(
    rvf-vulkan PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers
)
target_compile_definitions(
    rvf-vulkan PRIVATE RVF_AXPY_SHADER_PATH="${AXPY_BINARY}"
)
target_link_libraries(rvf-vulkan PRIVATE Vulkan::Vulkan)

if(RVF_USE_SYSTEM_NCNN)
    find_package(ncnn CONFIG REQUIRED)
else()
    set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn"
        "${CMAKE_CURRENT_BINARY_DIR}/ncnn"
    )
endif()

add_executable(
    rvf-frame-core-test
    tests/frame_core_ncnn_test.cpp
    src/attention_mask_descriptor.cpp
    src/gelu.cpp
)
target_include_directories(rvf-frame-core-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(rvf-frame-core-test PRIVATE cxx_std_20)
target_compile_options(rvf-frame-core-test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-frame-core-test PRIVATE ncnn)

add_executable(
    rvf-flow-warp-test
    tests/flow_warp_test.cpp
    src/flow_warp.cpp
)
target_include_directories(rvf-flow-warp-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(rvf-flow-warp-test PRIVATE cxx_std_20)
target_compile_options(rvf-flow-warp-test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-flow-warp-test PRIVATE ncnn)

add_executable(rvf-spynet-test tests/spynet_ncnn_test.cpp)
target_compile_features(rvf-spynet-test PRIVATE cxx_std_20)
target_compile_options(rvf-spynet-test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-spynet-test PRIVATE ncnn)

add_library(rvf-optical-flow STATIC src/spynet.cpp src/flow_warp.cpp)
target_include_directories(rvf-optical-flow PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(rvf-optical-flow PUBLIC cxx_std_20)
target_compile_options(rvf-optical-flow PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-optical-flow PUBLIC ncnn)

add_executable(rvf-spynet-flow-test tests/spynet_flow_test.cpp)
target_compile_options(rvf-spynet-flow-test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-spynet-flow-test PRIVATE rvf-optical-flow)

add_library(rvf-restoration STATIC src/recurrent_restorer.cpp src/attention_mask_descriptor.cpp src/gelu.cpp)
target_include_directories(rvf-restoration PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(rvf-restoration PUBLIC cxx_std_20)
target_compile_options(rvf-restoration PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-restoration PUBLIC rvf-optical-flow)

add_executable(rvf-sequence-test tests/sequence_test.cpp)
target_compile_options(rvf-sequence-test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-sequence-test PRIVATE rvf-restoration)

add_executable(rvf-gelu-test tests/gelu_test.cpp src/gelu.cpp)
target_include_directories(rvf-gelu-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(rvf-gelu-test PRIVATE cxx_std_20)
target_compile_options(rvf-gelu-test PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(rvf-gelu-test PRIVATE ncnn)
