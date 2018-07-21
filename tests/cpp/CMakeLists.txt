#
# Adapted from https://github.com/dmonopoly/gtest-cmake-example
#
cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME MigaTests LANGUAGES CXX CUDA)
project(${PROJECT_NAME})

# C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 11)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Miga lib
set(MIGA_SRC_DIR "../../package/miga/src")

set(MIGA_CPP_SRC
    "${MIGA_SRC_DIR}/Population.cpp"
    "${MIGA_SRC_DIR}/CPUPopulation.cpp"
)

set(MIGA_CUDA_SRC
    "${MIGA_SRC_DIR}/GPUPopulation.cu"
)

add_library(miga_lib STATIC ${MIGA_CPP_SRC} ${MIGA_CUDA_SRC})
include_directories(${MIGA_SRC_DIR})

# Compiler options
if(NOT CMAKE_CUDA_FLAGS)
  set(CMAKE_CUDA_FLAGS "-arch compute_30 -code sm_30,compute_30")
endif()
set_target_properties(miga_lib PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_source_files_properties(${MIGA_CUDA_SRC} PROPERTIES COMPILE_FLAGS "-cudart static -res-usage -maxrregcount 63 -Xcompiler -march=native")

if(MSVC)
    set_source_files_properties(${MIGA_CPP_SRC} PROPERTIES COMPILE_FLAGS "/W4 $<$<CONFIG:RELEASE>:/O2>")

    # Force static runtime libraries for msvc builds
    set(variables 
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_MINSIZEREL
    )
    foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        endif()
    endforeach()
else()
    # General unix flags
    set_source_files_properties(${MIGA_CPP_SRC} PROPERTIES COMPILE_FLAGS "-Wall -march=native $<$<CONFIG:RELEASE>:-O3>")

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set_source_files_properties(${MIGA_CPP_SRC} PROPERTIES COMPILE_FLAGS "-stdlib=libc++")
    endif()
endif()

# Testing
add_subdirectory(lib/gtest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Unit Tests
add_executable(all_tests main.cpp ga_tests.cpp)
# Google test libs
target_link_libraries(all_tests gtest gtest_main)
# Miga code
target_link_libraries(all_tests miga_lib)

if(APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET all_tests PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable
add_test(NAME miga-tests COMMAND all_tests)
