# cascade-router C++ core — ONNX benchmark + HTTP proxy
#
# See header comments in build_and_run.bat for Windows ONNX Runtime setup.
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.20)
project(cascade_router_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Header-only HTTP server
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
)

# Ultra-fast JSON parser
FetchContent_Declare(
    simdjson
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    GIT_TAG v3.10.1
)
set(SIMDJSON_DEVELOPMENT_CHECKS OFF CACHE BOOL "" FORCE)

# JSON mutation for upstream payload rewriting
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(httplib simdjson nlohmann_json)

# ---------------------------------------------------------------------------
# ONNX Runtime (prebuilt binaries — not fetched by CMake)
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of extracted ONNX Runtime package")

if(NOT ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    endif()
endif()

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT is not set.\n"
        "Pass -DONNXRUNTIME_ROOT=<path> or set the ONNXRUNTIME_ROOT environment variable.")
endif()

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED
)

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        "${ONNXRUNTIME_ROOT}/lib"
        "${ONNXRUNTIME_ROOT}/lib/${CMAKE_VS_PLATFORM_NAME}"
    REQUIRED
)

if(WIN32)
    find_file(
        ONNXRUNTIME_DLL
        NAMES onnxruntime.dll
        HINTS
            "${ONNXRUNTIME_ROOT}/lib"
            "${ONNXRUNTIME_ROOT}/bin"
            "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native"
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

function(cascade_link_onnx target_name)
    target_include_directories(${target_name} PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(${target_name} PRIVATE "${ONNXRUNTIME_LIBRARY}")

    if(WIN32 AND ONNXRUNTIME_DLL)
        add_custom_command(
            TARGET ${target_name} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${ONNXRUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying onnxruntime.dll beside ${target_name}"
        )
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(benchmark_onnx src/benchmark_onnx.cpp)
cascade_link_onnx(benchmark_onnx)

add_executable(proxy_server src/proxy_server.cpp)
target_include_directories(proxy_server PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(proxy_server PRIVATE httplib::httplib simdjson::simdjson nlohmann_json::nlohmann_json)
cascade_link_onnx(proxy_server)

if(WIN32)
    target_compile_definitions(proxy_server PRIVATE _WIN32_WINNT=0x0A00)
endif()
