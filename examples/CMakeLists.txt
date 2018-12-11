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

# EXAMPLE EXECUTABLES

# Build a minimal working example to demonstrate truncated-Newton trust-region optimization over the sphere S2
add_executable(Riemannian_optimization_example ${CMAKE_CURRENT_SOURCE_DIR}/Riemannian_optimization_example.cpp)
target_link_libraries(Riemannian_optimization_example Optimization)

# Build a minimal working example to demonstrate the use of the simplified Euclidean truncated-Newton trust-region interface to minimize the McCormick function
add_executable(Euclidean_optimization_example ${CMAKE_CURRENT_SOURCE_DIR}/Euclidean_optimization_example.cpp)
target_link_libraries(Euclidean_optimization_example Optimization)

# Build a minimal working example to demonstrate the proximal gradient method on a group-sparse lasso problem
add_executable(proximal_gradient_example ${CMAKE_CURRENT_SOURCE_DIR}/proximal_gradient_example.cpp)
target_link_libraries(proximal_gradient_example Optimization)

# Build a minimal working example to demonstrate the use of the ADMM algorithm
add_executable(ADMM_example ${CMAKE_CURRENT_SOURCE_DIR}/ADMM_example.cpp)
target_link_libraries(ADMM_example Optimization)



