#
# Adapted from https://github.com/dmonopoly/gtest-cmake-example
# and https://github.com/kaizouman/gtest-cmake-example
#
cmake_minimum_required(VERSION 2.11)

set(PROJECT_NAME MigaTests)
project(${PROJECT_NAME})

# C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Compiler options
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options("-Wall" "$<$<CONFIG:RELEASE>:-O3>")

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options("-stdlib=libc++")
    else()
        # nothing special for gcc at the moment
    endif()
endif()

# Force static runtime libraries for msvc builds
if(MSVC)
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
endif()

# Miga sources
set(MIGA_SRC "../../package/miga/src")
set(MIGA_CPP 
    "${MIGA_SRC}/Population.cpp"
    "${MIGA_SRC}/CPUPopulation.cpp"
)

add_library(miga_lib ${MIGA_CPP})
include_directories(${MIGA_SRC})

# Testing
if (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
endif (APPLE)

add_subdirectory(lib/gtest)

enable_testing()

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Unit Tests
add_executable(all_tests main.cpp)
# Google test libs
target_link_libraries(all_tests gtest gtest_main)
# Miga code
target_link_libraries(all_tests miga_lib)

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable
add_test(NAME miga-tests COMMAND all_tests)
