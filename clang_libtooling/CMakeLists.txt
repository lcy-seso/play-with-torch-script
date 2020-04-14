cmake_minimum_required(VERSION 3.0)

project (clang_libtool CXX C)

set(TARGET_NAME clang-example)

find_package(Threads)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(Clang REQUIRED)
find_package(Clang REQUIRED clangTooling)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${Clang_DEFINITIONS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -std=c++11 -g -O0")

set(SRCS hello_world.cpp)

add_executable(${TARGET_NAME} ${SRCS})

llvm_map_components_to_libnames(llvm_libs support core irreader)

target_link_libraries(${TARGET_NAME}
    ${CMAKE_THREAD_LIBS_INIT}
    ${llvm_libs}
    clangTooling)
