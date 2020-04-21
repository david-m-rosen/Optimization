/** This small unit test ensures that the extensions added to the Eigen
 * MatrixBase template class in the file TRSQPEigenExtensions.h are implemented
 * correctly
 *
 *  Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

// Add definitions from TRSQPEigenExtensions to the Eigen MatrixBase class
// Note that THIS PREPROCESSOR COMMAND MUST PRECEDE ANY EIGEN HEADERS
#define EIGEN_MATRIXBASE_PLUGIN                                                \
  "Optimization/Constrained/TRSQPEigenExtensions.h"

#include <Eigen/Dense>
#include <gtest/gtest.h>

class EigenExtensionsTest : public testing::Test {

protected:
  typedef Eigen::VectorXd Vector;

  Vector v, w;

  virtual void SetUp() {
    v.resize(3);
    v << 1.0, 1.27, 3.14;

    w.resize(3);
    w << -1.0, 6.28, -3.25;
  }
};

// Dimension
TEST_F(EigenExtensionsTest, Dim) { EXPECT_EQ(v.size(), v.dim()); }

// Maximum element
TEST_F(EigenExtensionsTest, Max) { EXPECT_EQ(v.max(), v.maxCoeff()); }

// Elementwise maximum with scalar
TEST_F(EigenExtensionsTest, MaxWithElement) {
  double s = 2;

  Vector m = v.max(s);

  for (size_t k = 0; k < v.size(); ++k)
    EXPECT_EQ(m(k), std::max(v(k), s));
}

// Elementwise maximum with another vector
TEST_F(EigenExtensionsTest, MaxWithVector) {
  double s = 2;

  Vector m = v.max(w);

  EXPECT_LT((m - v.cwiseMax(w)).norm(), 1e-6);
}

// Maximum element
TEST_F(EigenExtensionsTest, Min) { EXPECT_EQ(v.min(), v.minCoeff()); }

// Elementwise maximum with scalar
TEST_F(EigenExtensionsTest, MinWithElement) {
  double s = 2;

  Vector m = v.min(s);

  for (size_t k = 0; k < v.size(); ++k)
    EXPECT_EQ(m(k), std::min(v(k), s));
}

// Elementwise maximum with scalar
TEST_F(EigenExtensionsTest, HadamardProduct) {

  Vector p = v.hadamard_product(w);

  EXPECT_LT((p - v.cwiseProduct(w)).norm(), 1e-6);
}

// Elementwise maximum with scalar
TEST_F(EigenExtensionsTest, HadamardInverse) {

  Vector p = v.hadamard_product(w);

  EXPECT_LT((p - v.cwiseProduct(w)).norm(), 1e-6);
}

// Inner product
TEST_F(EigenExtensionsTest, InnerProduct) {

  EXPECT_EQ(v.inner_product(w), v.dot(w));
}
