/** This simple example demonstrates the use of the LOBPCG method to recover a
 * few of the algebraically-smallest eigenpairs of the eigenvalue problem:
 *
 * Ax = lambda*x
 *
 */

#include "Optimization/LinearAlgebra/LOBPCG.h"

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>

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

int main() {

  /// Test configuration
  // Dimension of linear operators
  size_t n = 20;

  // Block size to use for LOBPCG
  size_t m = 5;

  // Number of desired eigenvalues k to use for LOBPCG
  size_t nev = 3;

  /// Termination criteria
  size_t max_iters = 5 * n;
  double tau = 1e-2;

  /// Test operators

  /// Construct linear operator A as a diagonal operator with a difficult
  /// spectrum
  Vector Aspec = Eigen::pow(10, Vector::LinSpaced(n, -5, 6).array());

  LinearOperator A = [Aspec](const Matrix &X) -> Matrix {
    return Aspec.asDiagonal() * X;
  };

  /// Set stopping function to be the relative error criterion
  StopFun stop_fun = [tau](size_t i, const LinearOperator &A,
                           const std::optional<LinearOperator> &B,
                           const std::optional<LinearOperator> &T, size_t nev,
                           const Vector &Theta, const Matrix &X,
                           const Vector &r) -> bool {
    Vector res_tols = tau * Theta.head(nev).cwiseAbs();
    return ((r.head(nev).array() <= res_tols.array()).count() == nev);
  };

  cout << "Requested spectrum of operator A: " << Aspec.head(nev).transpose()
       << endl
       << endl;

  /// Run LOBPCG

  cout << "Running LOBPCG ... " << endl;

  Vector Theta;
  Matrix X;
  size_t num_iters;
  size_t num_converged;
  std::tie(Theta, X) = Optimization::LinearAlgebra::LOBPCG<Vector, Matrix>(
      A, std::optional<LinearOperator>(), std::optional<LinearOperator>(), n, m,
      nev, max_iters, num_iters, num_converged, 0.0,
      std::optional<StopFun>(stop_fun));

  cout << "LOBPCG terminated after " << num_iters << " iterations" << endl;
  cout << "Returned eigenvalue estimates: " << Theta.transpose() << endl;

  /// Calculate residuals of returned eigenvector estimates

  Matrix R = A(X) - X * Theta.asDiagonal();
  Vector rnorms = R.colwise().norm();
  Vector xnorms = X.colwise().norm();

  cout << "Residuals of returned vectors:  " << rnorms.transpose() << endl;
  cout << "Norms of eigenvector estimates: " << xnorms.transpose() << endl;
}
