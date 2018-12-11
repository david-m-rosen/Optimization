# Find Eigen library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
find_package(Eigen3 3.3.3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library (version ${EIGEN3_VERSION})")
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}\n")
else()
message(STATUS "Eigen library not found!")
endif()

# Add the Eigen include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Add GTest unit tests
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# GTest also requires pthreads
find_package(Threads REQUIRED)


# UNIT TESTS
set(TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TNT_unit_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GradientDescent_unit_test.cpp)

add_executable(TNT_unit_test ${CMAKE_CURRENT_SOURCE_DIR}/TNT_unit_test.cpp)
target_link_libraries(TNT_unit_test Optimization ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(GradientDescent_unit_test ${CMAKE_CURRENT_SOURCE_DIR}/GradientDescent_unit_test.cpp)
target_link_libraries(GradientDescent_unit_test Optimization ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(gtest_main  ${TEST_SRCS})
target_link_libraries(gtest_main Optimization ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
gtest_discover_tests(gtest_main)




