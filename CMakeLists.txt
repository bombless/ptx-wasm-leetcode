cmake_minimum_required(VERSION 3.14)
project(ptx_vm VERSION 1.0.0 LANGUAGES CXX)

# Project information
set(PROJECT_AUTHOR "Zhenzhong Han <zhenzhong.han@qq.com>")
set(PROJECT_DESCRIPTION "A virtual machine implementation for executing NVIDIA PTX (Parallel Thread Execution) intermediate code")

# Require C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(MSVC)
    # Enable exceptions and RTTI for MSVC
    add_compile_options(/EHsc /GR)
else()
    # Enable exceptions and RTTI for GCC/Clang
    add_compile_options(-fexceptions -frtti)
endif()

# Options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_DOCUMENTATION "Build documentation" ON)

add_compile_options(-g)

# Add subdirectories
add_subdirectory(src/logger)
add_subdirectory(src/core)
add_subdirectory(src/decoder)
add_subdirectory(src/execution)
add_subdirectory(src/memory)
add_subdirectory(src/optimizer)
add_subdirectory(src/debugger)
add_subdirectory(src/host)
add_subdirectory(src/parser)
add_subdirectory(src/registers)

# Main executable
add_executable(ptx_vm
    src/main.cpp
)

# Link libraries to main executable
target_link_libraries(ptx_vm
    logger
    core
    decoder
    execution
    memory
    optimizer
    debugger
    host
    parser
    registers
)

# Include directories
target_include_directories(ptx_vm PRIVATE
    include
    src
)

# Compiler-specific options for main executable
if(MSVC)
    target_compile_options(ptx_vm PRIVATE /W4)
else()
    target_compile_options(ptx_vm PRIVATE -Wall -Wextra -pedantic)
endif()

# Build tests if enabled
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples if enabled
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(EMSCRIPTEN)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(EMCC_EXECUTABLE NAMES emcc REQUIRED)
    set(BARRACUDA_ROOT "${CMAKE_SOURCE_DIR}/third_party/BarraCUDA")

    if(NOT EXISTS "${BARRACUDA_ROOT}/src/nvidia/emit.c")
        message(FATAL_ERROR
            "BarraCUDA sources were not found at ${BARRACUDA_ROOT}. "
            "Initialize git submodules before configuring the web build: "
            "'git submodule update --init --recursive'."
        )
    endif()

    set(BARRACUDA_WEB_OUTPUT_DIR "${CMAKE_BINARY_DIR}/webapp")
    set(BARRACUDA_WEB_JS "${BARRACUDA_WEB_OUTPUT_DIR}/barracuda.js")
    set(PTX_WEB_INDEX_HTML "${BARRACUDA_WEB_OUTPUT_DIR}/index.html")
    set(LEETGPU_CATALOG_JS "${BARRACUDA_WEB_OUTPUT_DIR}/challenges.generated.js")
    set(CODEMIRROR_BUNDLE_JS "${BARRACUDA_WEB_OUTPUT_DIR}/codemirror_bundle.js")
    set(KATEX_CSS "${BARRACUDA_WEB_OUTPUT_DIR}/katex.min.css")
    set(KATEX_JS "${BARRACUDA_WEB_OUTPUT_DIR}/katex.min.js")
    set(KATEX_AUTORENDER_JS "${BARRACUDA_WEB_OUTPUT_DIR}/auto-render.min.js")
    set(KATEX_FONTS_DIR "${BARRACUDA_WEB_OUTPUT_DIR}/fonts")
    set(KATEX_FONTS_STAMP "${BARRACUDA_WEB_OUTPUT_DIR}/.katex-fonts.stamp")
    file(GLOB KATEX_FONT_FILES "${CMAKE_SOURCE_DIR}/web/fonts/*")
    set(BARRACUDA_WEB_SOURCES
        ${CMAKE_SOURCE_DIR}/web/barracuda_web_bridge.c
        ${BARRACUDA_ROOT}/src/fe/bc_err.c
        ${BARRACUDA_ROOT}/src/fe/preproc.c
        ${BARRACUDA_ROOT}/src/fe/lexer.c
        ${BARRACUDA_ROOT}/src/fe/parser.c
        ${BARRACUDA_ROOT}/src/fe/sema.c
        ${BARRACUDA_ROOT}/src/ir/bir.c
        ${BARRACUDA_ROOT}/src/ir/bir_lower.c
        ${BARRACUDA_ROOT}/src/ir/bir_mem2reg.c
        ${BARRACUDA_ROOT}/src/ir/bir_cfold.c
        ${BARRACUDA_ROOT}/src/ir/bir_dce.c
        ${BARRACUDA_ROOT}/src/nvidia/isel.c
        ${BARRACUDA_ROOT}/src/nvidia/emit.c
    )

    add_custom_command(
        OUTPUT "${BARRACUDA_WEB_JS}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${BARRACUDA_WEB_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E env
            EMSDK_PYTHON=${Python3_EXECUTABLE}
            ${EMCC_EXECUTABLE}
            -std=c99
            -O2
            -I${BARRACUDA_ROOT}/src
            -I${BARRACUDA_ROOT}/src/fe
            -I${BARRACUDA_ROOT}/src/ir
            -I${BARRACUDA_ROOT}/src/nvidia
            ${BARRACUDA_WEB_SOURCES}
            -sALLOW_MEMORY_GROWTH=1
            -sFORCE_FILESYSTEM=1
            -sINVOKE_RUN=0
            -sEXIT_RUNTIME=0
            -sMODULARIZE=1
            -sEXPORT_NAME=createBarraModule
            -sEXPORTED_FUNCTIONS=['_malloc','_free','_barracuda_compile_to_ptx','_barracuda_get_last_error']
            -sEXPORTED_RUNTIME_METHODS=['ccall','FS']
            -o "${BARRACUDA_WEB_JS}"
        DEPENDS ${BARRACUDA_WEB_SOURCES}
        COMMENT "Building BarraCUDA wasm compiler"
        VERBATIM
    )

    add_custom_target(barracuda_web DEPENDS "${BARRACUDA_WEB_JS}")

    add_custom_command(
        OUTPUT
            "${PTX_WEB_INDEX_HTML}"
            "${LEETGPU_CATALOG_JS}"
            "${CODEMIRROR_BUNDLE_JS}"
            "${KATEX_CSS}"
            "${KATEX_JS}"
            "${KATEX_AUTORENDER_JS}"
            "${KATEX_FONTS_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${BARRACUDA_WEB_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/web/index.html
            "${PTX_WEB_INDEX_HTML}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/web/challenges.generated.js
            "${LEETGPU_CATALOG_JS}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/web/codemirror_bundle.js
            "${CODEMIRROR_BUNDLE_JS}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/web/katex.min.css
            "${KATEX_CSS}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/web/katex.min.js
            "${KATEX_JS}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/web/auto-render.min.js
            "${KATEX_AUTORENDER_JS}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/web/fonts
            "${KATEX_FONTS_DIR}"
        COMMAND ${CMAKE_COMMAND} -E touch "${KATEX_FONTS_STAMP}"
        DEPENDS
            ${CMAKE_SOURCE_DIR}/web/index.html
            ${CMAKE_SOURCE_DIR}/web/challenges.generated.js
            ${CMAKE_SOURCE_DIR}/web/codemirror_bundle.js
            ${CMAKE_SOURCE_DIR}/web/katex.min.css
            ${CMAKE_SOURCE_DIR}/web/katex.min.js
            ${CMAKE_SOURCE_DIR}/web/auto-render.min.js
            ${KATEX_FONT_FILES}
        COMMENT "Copying browser UI assets"
        VERBATIM
    )

    add_custom_target(
        ptx_web_ui
        DEPENDS
            "${PTX_WEB_INDEX_HTML}"
            "${LEETGPU_CATALOG_JS}"
            "${CODEMIRROR_BUNDLE_JS}"
            "${KATEX_CSS}"
            "${KATEX_JS}"
            "${KATEX_AUTORENDER_JS}"
            "${KATEX_FONTS_STAMP}"
    )

    add_executable(ptx_web
        web/ptx_web_bridge.cpp
    )

    target_link_libraries(ptx_web
        logger
        core
        decoder
        execution
        memory
        optimizer
        debugger
        host
        parser
        registers
    )

    target_include_directories(ptx_web PRIVATE
        include
        src
    )

    set_target_properties(ptx_web PROPERTIES
        OUTPUT_NAME "ptx_web"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/webapp"
    )

    target_link_options(ptx_web PRIVATE
        "SHELL:-sALLOW_MEMORY_GROWTH=1"
        "SHELL:-sNO_EXIT_RUNTIME=1"
        "SHELL:-sFORCE_FILESYSTEM=1"
        "SHELL:-sDISABLE_EXCEPTION_CATCHING=0"
        "SHELL:-sEXPORTED_FUNCTIONS=['_malloc','_free','_ptxvm_load_program','_ptxvm_get_entry_count','_ptxvm_get_entry_name','_ptxvm_get_param_count','_ptxvm_get_param_name','_ptxvm_get_param_type','_ptxvm_is_param_pointer','_ptxvm_run_vector_add','_ptxvm_run_matrix_addition','_ptxvm_run_matrix_multiplication','_ptxvm_run_matrix_transpose','_ptxvm_run_reduction','_ptxvm_run_softmax','_ptxvm_run_softmax_attention','_ptxvm_run_color_inversion','_ptxvm_get_result_count','_ptxvm_get_result_value','_ptxvm_get_last_error']"
        "SHELL:-sEXPORTED_RUNTIME_METHODS=['ccall','FS','HEAPF32','HEAPU8']"
    )

    add_dependencies(ptx_web barracuda_web ptx_web_ui)
endif()
