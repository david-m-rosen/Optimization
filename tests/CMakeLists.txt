project(Optimization-Tests CXX)

# Find Eigen library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")
find_package(Eigen3 3.3.3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library (version ${EIGEN3_VERSION})")
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}\n")
else()
message(STATUS "Eigen library not found!")
endif()

# Add the Eigen include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Get GoogleTest
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/googletest.cmake)
fetch_googletest(
    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake
    ${PROJECT_BINARY_DIR}/googletest
    )

# Add GTest unit tests
enable_testing()

# UNIT TESTS
set(TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/TNT_unit_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GradientDescent_unit_test.cpp)

# INDIVIDUAL UNIT TESTS

add_executable(TNT_unit_test ${CMAKE_CURRENT_SOURCE_DIR}/TNT_unit_test.cpp)
target_link_libraries(TNT_unit_test Optimization gtest_main)

add_executable(GradientDescent_unit_test ${CMAKE_CURRENT_SOURCE_DIR}/GradientDescent_unit_test.cpp)
target_link_libraries(GradientDescent_unit_test Optimization gtest_main)

# RUN ALL UNIT TESTS
add_executable(RunAllUnitTests ${TEST_SRCS})
target_link_libraries(RunAllUnitTests Optimization gtest_main)




