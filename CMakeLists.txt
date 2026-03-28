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
        "SHELL:-sEXPORTED_FUNCTIONS=['_malloc','_free','_ptxvm_load_program','_ptxvm_get_entry_count','_ptxvm_get_entry_name','_ptxvm_get_param_count','_ptxvm_get_param_name','_ptxvm_get_param_type','_ptxvm_is_param_pointer','_ptxvm_run_vector_add','_ptxvm_get_result_count','_ptxvm_get_result_value','_ptxvm_get_last_error']"
        "SHELL:-sEXPORTED_RUNTIME_METHODS=['ccall','FS','HEAPF32']"
    )

    add_custom_command(TARGET ptx_web POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/web/index.html
            $<TARGET_FILE_DIR:ptx_web>/index.html
        COMMENT "Copying browser UI assets"
    )
endif()
