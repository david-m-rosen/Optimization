# Find Eigen library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
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
add_executable(TNT_unit_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/TNT_unit_test.cpp)
target_link_libraries(TNT_unit_test Optimization ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
gtest_discover_tests(TNT_unit_test)


# Add these to a custom target named $EXAMPLES_TARGET_NAME in the generated Makefile
add_custom_target(${TESTS_TARGET_NAME})
add_dependencies(${TESTS_TARGET_NAME} TNT_unit_test)






