#include "Optimization/LinearAlgebra/LOBPCG.h"

#include <iostream>

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Sparse>

#include "gtest/gtest.h"

// Typedef for the numerical type will use in the following tests
typedef double Scalar;

// Typedefs for Eigen matrix types
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor> SparseMatrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix> LinOp;

using namespace std;

class LOBPCGTest : public testing::Test {

protected:
  /// Test configuration

  // Dimension of linear operators
  size_t n = 500;

  // Block size
  size_t m = 5;

  // Number of desired eigenvalues k
  size_t nev = 3;

  /// Termination criteria
  size_t max_iters = n;
  double tau = 1e-6;

  /// Test operators

  Vector D;
  LinOp A, B, T;

  virtual void SetUp() {

    /// Set A to be multiplication by a diagonal matrix with a fixed spectrum
    // Set eigenvalues
    this->D = Vector::LinSpaced(n, -.5 * n, .5 * n);

    A = [this](const Matrix &X) -> Matrix { return this->D.asDiagonal() * X; };

    /// Set preconditioner T to be multiplication by the absolute values of D
    T = [this](const Matrix &X) -> Matrix {
      return this->D.cwiseAbs().asDiagonal() * X;
    };
  }
};

/// Test LOBPCG with standard eigenvalue problem and no preconditioning
TEST_F(LOBPCGTest, EigenvalueProblem) {

  Vector Theta;
  Matrix X;
  size_t num_iters;
  size_t num_converged;
  std::tie(Theta, X) = Optimization::LinearAlgebra::LOBPCG<Vector, Matrix>(
      A, std::optional<LinOp>(), std::optional<LinOp>(), n, m, nev, n,
      num_iters, num_converged, tau);

  /// Verify that the method reported the correct number of converged
  /// eigenvalues
  EXPECT_EQ(num_converged, nev);

  /// Verify that the estimated eigenvalues are correct to high accuracy
  Vector Lambda_true = D.head(nev);
  EXPECT_LT((Theta - Lambda_true).norm(), 1e-3);
}

/// Test LOBPCG with standard eigenvalue problem and a simple (diagonal) PSD
/// preconditioner
TEST_F(LOBPCGTest, PreconditionedEigenvalueProblem) {

  Vector Theta;
  Matrix X;
  size_t num_iters;
  size_t num_converged;
  std::tie(Theta, X) = Optimization::LinearAlgebra::LOBPCG<Vector, Matrix>(
      A, std::optional<LinOp>(), std::optional<LinOp>(T), n, m, nev, n,
      num_iters, num_converged, tau);

  /// Verify that the method reported the correct number of converged
  /// eigenvalues
  EXPECT_EQ(num_converged, nev);

  /// Verify that the estimated eigenvalues are correct to high accuracy
  Vector Lambda_true = D.head(nev);
  EXPECT_LT((Theta - Lambda_true).norm(), 1e-3);
}
