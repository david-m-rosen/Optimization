#include "Optimization/LinearAlgebra/LOBPCG.h"

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Sparse>

#include <limits>

#include "gtest/gtest.h"

// Typedef for the numerical type will use in the following tests
typedef double Scalar;

// Typedefs for Eigen matrix types
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor> SparseMatrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>
    LinearOperator;
typedef Optimization::LinearAlgebra::LOBPCGUserFunction<Vector, Matrix> StopFun;

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
  LinearOperator A, B, T;
  StopFun stop_fun;

  virtual void SetUp() {

    /// Set A to be multiplication by a diagonal matrix with a fixed spectrum
    // Set eigenvalues
    this->D = Vector::LinSpaced(n, -.5 * n, .5 * n);

    A = [this](const Matrix &X) -> Matrix { return this->D.asDiagonal() * X; };

    /// Set preconditioner T to be multiplication by the absolute values of D
    T = [this](const Matrix &X) -> Matrix {
      return this->D.cwiseAbs().asDiagonal() * X;
    };

    /// Set stopping function to be the relative error criterion
    stop_fun = [this](size_t i, const LinearOperator &A,
                      const std::optional<LinearOperator> &B,
                      const std::optional<LinearOperator> &T, size_t nev,
                      const Vector &Theta, const Matrix &X, const Vector &r) {
      Vector res_tols = this->tau * Theta.head(nev).cwiseAbs();
      return ((r.head(nev).array() <= res_tols.array()).count() == nev);
    };
  }
};

/// Test the SVQB helper function
TEST_F(LOBPCGTest, SVQB) {

  size_t m = 5;
  size_t nu = 3;

  // Construct a random m x m matrix
  Matrix L = Matrix::Random(m, m);

  // Construct symmetric positive-definite metric matrix M
  Matrix M = L * L.transpose();

  // Construct test basis matrix U
  Matrix U = Matrix::Random(m, nu);

  // Calculate M-orthonormalization of U
  Matrix V = Optimization::LinearAlgebra::SVQB<Vector, Matrix>(U, M * U);

  /// Check that V is indeed M-orthonormal
  Matrix R = V.transpose() * M * V - Matrix::Identity(nu, nu);
  EXPECT_LT(R.norm(), 1e-6);

  /// Check that U and V have the same range (i.e., they determine the same
  /// subspace).  Note that if S = range(U) = range(V), then range([U, V]) = S
  /// as well.  Therefore, it suffices to check that that rank of
  /// B := [U,V] is nu

  Matrix B(m, 2 * nu);
  B.leftCols(nu) = U;
  B.rightCols(nu) = V;

  // Compute rank-revealing QR factorization of B
  Eigen::ColPivHouseholderQR<Matrix> qr(B);

  EXPECT_EQ(qr.rank(), nu);
}

/// Test the SVQBdrop helper function
TEST_F(LOBPCGTest, SVQBdrop) {

  size_t m = 5;
  size_t nu = 3;

  // Construct a random m x m matrix
  Matrix L = Matrix::Random(m, m);

  // Construct symmetric positive-definite metric matrix M
  Matrix M = L * L.transpose();

  // Construct test basis matrix U
  Matrix U = Matrix::Random(m, nu + 1);

  // Make U rank-deficient by setting the final column to be a scalar multiple
  // of the penultimate
  U.col(nu) = -1 * U.col(nu - 1);

  // Calculate M-orthonormalization of U
  Matrix V = Optimization::LinearAlgebra::SVQBdrop<Vector, Matrix>(U, M * U);

  /// Check that the returned V has nu columns -- that is, that we truncated 1
  /// column due to U's rank-deficiency
  EXPECT_EQ(V.cols(), nu);

  /// Check that the returned V is indeed M-orthonormal
  Matrix R = V.transpose() * M * V - Matrix::Identity(nu, nu);
  EXPECT_LT(R.norm(), 1e-6);

  /// Check that U and V have the same range (i.e., they determine the same
  /// subspace).  Note that if S = range(U) = range(V), then range([U, V]) = S
  /// as well.  Therefore, it suffices to check that that rank of
  /// B := [U,V] is nu

  Matrix B(m, 2 * nu + 1);
  B.leftCols(nu + 1) = U;
  B.rightCols(nu) = V;

  // Compute rank-revealing QR factorization of B
  Eigen::ColPivHouseholderQR<Matrix> qr(B);

  EXPECT_EQ(qr.rank(), nu);
}

/// Test the orthoDrop helper function
TEST_F(LOBPCGTest, orthoDrop) {

  size_t m = 7;

  /// Construct symmetric positive-definite metric matrix M
  Matrix L = Matrix::Random(m, m);
  Matrix M = L * L.transpose();

  Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix> Mop =
      [&M](const Matrix &X) -> Matrix { return M * X; };

  /// Construct external M-orthonormal basis V
  size_t nv = 3;
  Matrix V = Matrix::Random(m, nv);

  // Orthonormalize V
  V = Optimization::LinearAlgebra::SVQB<Vector, Matrix>(V, M * V);

  // Verify that V is indeed M-orthonormal
  EXPECT_LT((V.transpose() * M * V - Matrix::Identity(nv, nv)).norm(), 1e-6);

  /// Construct (singular) input matrix U
  size_t nu = 3;
  Matrix U = Matrix::Random(m, nu + 1);

  // Make U rank-deficient by setting the final column to be a scalar multiple
  // of the penultimate column
  U.col(nu) = -1 * U.col(nu - 1);

  // Calculate orthonormalization of U against V
  Matrix W = Optimization::LinearAlgebra::orthoDrop<Vector, Matrix>(
      U, V,
      std::optional<
          Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>(Mop));

  /// Check that the concatenated basis B := [V, W] is indeed M-orthonormal
  Matrix B(m, nv + W.cols());
  B.leftCols(nv) = V;
  B.rightCols(W.cols()) = W;
  EXPECT_LT(
      (B.transpose() * M * B - Matrix::Identity(nv + W.cols(), nv + W.cols()))
          .norm(),
      1e-6);

  /// Verify that range([V, W]) contains range([U, V])

  // If B is M-orthonormal, then in particular it has full rank.  Moreover, if
  // range([V, W]) contains range([U, V]), then range([B]) = range([B, U])
  // => #cols B = rank([B]) = rank([B, U])

  Matrix S(m, B.cols() + U.cols());
  S.leftCols(B.cols()) = B;
  S.rightCols(U.cols()) = U;

  // Compute rank-revealing QR factorization of B
  Eigen::ColPivHouseholderQR<Matrix> qr(S);

  EXPECT_EQ(qr.rank(), B.cols());
}

/// Test the orthoDrop helper function *without* passing an optional metric
/// operator M
TEST_F(LOBPCGTest, orthoDropNoMetric) {

  size_t m = 7;

  /// Construct external orthonormal basis V
  size_t nv = 3;
  Matrix V = Matrix::Random(m, nv);

  // Orthonormalize V
  V = Optimization::LinearAlgebra::SVQB<Vector, Matrix>(V, V);

  // Verify that V is indeed orthonormal
  EXPECT_LT((V.transpose() * V - Matrix::Identity(nv, nv)).norm(), 1e-6);

  /// Construct (singular) input matrix U
  size_t nu = 3;
  Matrix U = Matrix::Random(m, nu + 1);

  // Make U rank-deficient by setting the final column to be a scalar multiple
  // of the penultimate column
  U.col(nu) = -1 * U.col(nu - 1);

  // Calculate orthonormalization of U against V
  Matrix W = Optimization::LinearAlgebra::orthoDrop<Vector, Matrix>(
      U, V,
      std::optional<
          Optimization::LinearAlgebra::SymmetricLinearOperator<Matrix>>(
          nullopt));

  /// Check that the concatenated basis B := [V, W] is indeed orthonormal
  Matrix B(m, nv + W.cols());
  B.leftCols(nv) = V;
  B.rightCols(W.cols()) = W;
  EXPECT_LT((B.transpose() * B - Matrix::Identity(nv + W.cols(), nv + W.cols()))
                .norm(),
            1e-6);

  /// Verify that range([V, W]) contains range([U, V])

  // If B is M-orthonormal, then in particular it has full rank.  Moreover, if
  // range([V, W]) contains range([U, V]), then range([B]) = range([B, U])
  // => #cols B = rank([B]) = rank([B, U])

  Matrix S(m, B.cols() + U.cols());
  S.leftCols(B.cols()) = B;
  S.rightCols(U.cols()) = U;

  // Compute rank-revealing QR factorization of B
  Eigen::ColPivHouseholderQR<Matrix> qr(S);

  EXPECT_EQ(qr.rank(), B.cols());
}

/// Test the basic Rayleigh-Ritz helper function
TEST_F(LOBPCGTest, RayleighRitz) {

  // Sample two symmetric matrices of appropriate dimension
  size_t n = 7;

  /// Construct symmetric positive-definite metric matrix A
  Matrix AL = Matrix::Random(n, n);
  Matrix A = AL * AL.transpose();

  // Construct symmetric positive-definite metric matrix B
  Matrix BL = Matrix::Random(n, n);
  Matrix B = BL * BL.transpose();

  Vector Theta;
  Matrix C;
  std::tie(Theta, C) =
      Optimization::LinearAlgebra::RayleighRitz<Vector, Matrix>(A, B);

  /// Verify that C'AC = Theta

  EXPECT_LT((C.transpose() * A * C - Matrix(Theta.asDiagonal())).norm(), 1e-8);

  /// Verify that C'BC = In
  EXPECT_LT((C.transpose() * B * C - Matrix::Identity(n, n)).norm(), 1e-8);
}

/// Test modified Rayleigh-Ritz method with useOrtho = true
TEST_F(LOBPCGTest, ModifiedRayleighRitzUseOrthoTrue) {

  // Sample two symmetric matrices of appropriate dimension
  size_t ns = 9;
  size_t nx = 4;
  size_t nc = 2;

  /// Construct symmetric positive-definite metric matrix A
  Matrix AL = Matrix::Random(ns, ns);
  Matrix A = AL * AL.transpose();

  // Construct symmetric positive-definite metric matrix B
  Matrix BL = Matrix::Random(ns, ns);
  Matrix B = BL * BL.transpose();

  const auto &[Thetax, Thetap, Cx, Cp, useOrtho] =
      Optimization::LinearAlgebra::ModifiedRayleighRitz<Vector, Matrix>(
          A, B, nx, nc, true);

  /// Verify that the return values have the correct dimensions
  EXPECT_EQ(Thetax.size(), nx);

  EXPECT_EQ(Thetap.rows(), nx - nc);
  EXPECT_EQ(Thetap.cols(), nx - nc);

  EXPECT_EQ(Cx.rows(), ns);
  EXPECT_EQ(Cx.cols(), nx);

  EXPECT_EQ(Cp.rows(), ns);
  EXPECT_EQ(Cp.cols(), nx - nc);

  // Reconstruct matrices Theta and C
  Matrix Theta = Matrix::Zero(2 * nx - nc, 2 * nx - nc);
  Theta.topLeftCorner(nx, nx) = Thetax.asDiagonal();
  Theta.bottomRightCorner(nx - nc, nx - nc) = Thetap;

  Matrix C(ns, 2 * nx - nc);
  C.leftCols(nx) = Cx;
  C.rightCols(nx - nc) = Cp;

  /// Verify that C'C = I
  EXPECT_LT(
      (C.transpose() * C - Matrix::Identity(2 * nx - nc, 2 * nx - nc)).norm(),
      1e-6);

  /// Verify that C'AC = Theta
  EXPECT_LT((C.transpose() * A * C - Theta).norm(), 1e-6);

  /// Verify that useOrtho == true
  EXPECT_TRUE(useOrtho);
}

/// Test modified Rayleigh-Ritz method with useOrtho = false and singular B
/// matrix
TEST_F(LOBPCGTest, ModifiedRayleighRitzUseOrthoFalseIllConditionedB) {

  // Sample two symmetric matrices of appropriate dimension
  size_t ns = 9;
  size_t nx = 4;
  size_t nc = 2;

  /// Construct symmetric positive-definite metric matrix A
  Matrix AL = Matrix::Random(ns, ns);
  Matrix A = AL * AL.transpose();

  // Construct matrix B = Id
  Matrix BL = Matrix::Random(ns, ns);
  // Multiply first column by 10^-8 to make this suuuper
  // ill-conditioned
  BL.col(0) *= 1e-8;

  Matrix B = BL * BL.transpose();

  const auto &[Thetax, Thetap, Cx, Cp, useOrtho] =
      Optimization::LinearAlgebra::ModifiedRayleighRitz<Vector, Matrix>(
          A, B, nx, nc, false);

  /// Verify that useOrtho has been set to true
  EXPECT_TRUE(useOrtho);
}

/// Test modified Rayleigh-Ritz method with useOrtho = false
TEST_F(LOBPCGTest, ModifiedRayleighRitzUseOrthoFalse) {

  // Sample two symmetric matrices of appropriate dimension
  size_t ns = 9;
  size_t nx = 4;
  size_t nc = 2;

  /// Construct symmetric positive-definite metric matrix A
  Matrix AL = Matrix::Random(ns, ns);
  Matrix A = AL * AL.transpose();

  // Construct symmetric positive-definite metric matrix B
  Matrix BL = Matrix::Random(ns, ns);
  Matrix B = BL * BL.transpose();

  const auto &[Thetax, Thetap, Cx, Cp, useOrtho] =
      Optimization::LinearAlgebra::ModifiedRayleighRitz<Vector, Matrix>(
          A, B, nx, nc, false);

  EXPECT_EQ(Thetax.size(), nx);

  EXPECT_EQ(Thetap.rows(), nx - nc);
  EXPECT_EQ(Thetap.cols(), nx - nc);

  EXPECT_EQ(Cx.rows(), ns);
  EXPECT_EQ(Cx.cols(), nx);

  EXPECT_EQ(Cp.rows(), ns);
  EXPECT_EQ(Cp.cols(), nx - nc);

  // Reconstruct matrices Theta and C
  Matrix Theta = Matrix::Zero(2 * nx - nc, 2 * nx - nc);
  Theta.topLeftCorner(nx, nx) = Thetax.asDiagonal();
  Theta.bottomRightCorner(nx - nc, nx - nc) = Thetap;

  Matrix C(ns, 2 * nx - nc);
  C.leftCols(nx) = Cx;
  C.rightCols(nx - nc) = Cp;

  /// Verify that C has the correct dimensions
  EXPECT_EQ(C.rows(), ns);
  EXPECT_EQ(C.cols(), 2 * nx - nc);

  /// Verify that C' * B * C = I
  EXPECT_LT((C.transpose() * B * C - Matrix::Identity(2 * nx - nc, 2 * nx - nc))
                .norm(),
            1e-6);

  /// Verify that C'AC = Theta
  EXPECT_LT((C.transpose() * A * C - Theta).norm(), 1e-6);

  /// Verify that useOrtho == false
  EXPECT_FALSE(useOrtho);
}

/// Test LOBPCG with standard eigenvalue problem and no preconditioning
TEST_F(LOBPCGTest, EigenvalueProblem) {

  Vector Theta;
  Matrix X;
  size_t num_iters;
  size_t num_converged;
  std::tie(Theta, X) = Optimization::LinearAlgebra::LOBPCG<Vector, Matrix>(
      A, std::optional<LinearOperator>(), std::optional<LinearOperator>(), n, m,
      nev, n, num_iters, num_converged, tau);

  /// Verify that the reported eigenvalues converged to the required tolerance
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
      A, std::optional<LinearOperator>(), std::optional<LinearOperator>(T), n,
      m, nev, n, num_iters, num_converged, tau);

  /// Verify that the method reported the correct number of converged
  /// eigenvalues
  EXPECT_EQ(num_converged, nev);

  /// Verify that the estimated eigenvalues are correct to high accuracy
  Vector Lambda_true = D.head(nev);
  EXPECT_LT((Theta - Lambda_true).norm(), 1e-3);
}
