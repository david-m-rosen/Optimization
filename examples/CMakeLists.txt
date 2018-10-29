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

# EXAMPLE EXECUTABLES

# Build unit test
add_executable(TNTUnitTest EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/TNTUnitTest.cpp )
target_link_libraries(TNTUnitTest Optimization)

# Build a minimal working example to demonstrate gradient descent applied to solve a simple quadratic minimization problem
add_executable(gradient_descent_example EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/gradient_descent_example.cpp)
target_link_libraries(gradient_descent_example Optimization)

# Build a minimal working example to demonstrate truncated-Newton trust-region optimization over the sphere S2
add_executable(Riemannian_optimization_example EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/Riemannian_optimization_example.cpp)
target_link_libraries(Riemannian_optimization_example Optimization)

# Build a minimal working example to demonstrate the use of the simplified Euclidean truncated-Newton trust-region interface to minimize the McCormick function
add_executable(smooth_optimization_example EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/smooth_optimization_example.cpp)
target_link_libraries(smooth_optimization_example Optimization)

# Build a minimal working example to the proximal gradient method on a group-sparse lasso problem
add_executable(ProximalGradientTest EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/ProximalGradientTest.cpp)
target_link_libraries(ProximalGradientTest Optimization)

# Add these to a custom target named $EXAMPLES_TARGET_NAME in the generated Makefile
add_custom_target(${EXAMPLES_TARGET_NAME})
add_dependencies(${EXAMPLES_TARGET_NAME} TNTUnitTest gradient_descent_example Riemannian_optimization_example smooth_optimization_example ProximalGradientTest)






