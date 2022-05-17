/** This header file provides an implementation of the locally optimal block
 * preconditioned conjugate gradient (LOBPCG) method for computing a few
 * algebraically-smallest eigenpairs (lambda, x) of the symmetric generalized
 * eigenproblem:
 *
 * Ax = lambda * Bx
 *
 * Here A and B are assumed to be symmetric linear operators, with B
 * positive-definite.
 *
 * Our implementation generally follows the one given in the paper:
 *
 * "Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX) in Hypre
 * and PETSc", by A.V. Knyazev, M.E. Argentati, I. Lashuk, and E.E. Ovtchinnikov
 *
 * Note that in *THIS SPECIFIC FILE*, it is assumed that the template parameters
 * 'Matrix' and 'Vector' correspond to dense Matrix and Vector types in the
 * Eigen library.
 *
 * Copyright (C) 2022 by David M. Rosen (d.rosen@northeastern.edu)
 */

#pragma once

#include <optional>
#include <random>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "Optimization/LinearAlgebra/Concepts.h"

namespace Optimization {

namespace LinearAlgebra {

/** An alias template for a user-definable function that can be used to access
 * various interesting bits of information about the internal state of the
 * LOBPCG algorithm as it runs.  Here:
 * - i: Index of current iteration
 * - A: Symmetric linear operator for generalized eigenvalue problem
 * - B: Optional symmetric positive-definite linear operator for generalized
 *   eigenvalue problem
 * - T: Optional symmetric positive-definite preconditioning operator
 * - Theta: the current vector of eigenvalue estimates (Ritz values)
 * - nev:  The number of desired eigenpairs
 * - X: the current set of eigenvector estimates
 * - r: the current vector of residual norms
 *
 * This function is called once per iteration, after the Rayleigh-Ritz procedure
 * is used to update the current Ritz pairs and their corresponding residuals
 *
 * This function may also return the Boolean value 'true' in order to terminate
 * the LOBPCG algorithm; this provides a convenient means of implementing a
 * custom (user-definable) stopping criterion.
 */
template <typename Vector, typename Matrix, typename Scalar = double,
          typename... Args>
using LOBPCGUserFunction = std::function<bool(
    size_t i, const SymmetricLinearOperator<Matrix, Args...> &A,
    const std::optional<SymmetricLinearOperator<Matrix, Args...>> &B,
    const std::optional<SymmetricLinearOperator<Matrix, Args...>> &T,
    size_t nev, const Vector &Theta, const Matrix &X, const Vector &residuals,
    Args &... args)>;

/** This function estimates the smallest eigenpairs (lambda, x) of the
 * generalized symmetric eigenvalue problem:
 *
 * Ax = lambda * Bx
 *
 * using Knyazev's locally optimal block preconditioned conjugate gradient
 * (LOBPCG) method.  Here:
 *
 * - A is a symmetric linear operator
 * - B is an optional positive-definite symmetric operator.  The absence of this
 *   parameter indicates that B == I (i.e. that we are solving a standard
 *   symmetric eigenproblem)
 * - T is an optional positive-definite symmetric preconditioner that
 *   approximates A^-1 in the sense that the condition number kappa(TA) should
 *   be (much) smaller than kappa(A) itself.  The absence of this parameter
 *   indicates that T == I (i.e. that no preconditioning is applied)
 * - X0 is an n x m matrix containing an initial set of linearly-independent
 *   eigenvector estimates.
 * - nev is the number of desired eigenpairs; note that this number should be
 *   less than the block size m.
 * - max_iters is the maximum number of iterations to perform
 * - num_iters is a return value that gives the number of iterations
 *   executed by the algorithm
 * - num_converged is a return value that indicates the number of eigenpairs
 *   that have converged to the requested precision
 * - tau is the stopping tolerance for the convergence test described in
 *   Section 4.3 of the paper "A Robust and Efficient Implementation of LOBPCG";
 *   specifically, an eigenpair (lambda, x) is considered converged when:
 *
 *   ||Ax - lambda * Bx|| / (||A||_2 + lambda * ||B_2|| * ||x||) <= tau
 *
 * - user_function is an optional user-defined function that can be used to
 *   inspect various variables of interest as the algorithm runs, and
 *   (optionally) to define a custom stopping criterion for the algorithm.
 */
template <typename Vector, typename Matrix, typename Scalar = double,
          typename... Args>
std::pair<Vector, Matrix>
LOBPCG(const SymmetricLinearOperator<Matrix, Args...> &A,
       const std::optional<SymmetricLinearOperator<Matrix, Args...>> &B,
       const std::optional<SymmetricLinearOperator<Matrix, Args...>> &T,
       const Matrix &X0, size_t nev, size_t max_iters, size_t &num_iters,
       size_t &num_converged, Args &... args, Scalar tau = 1e-6,
       const std::optional<LOBPCGUserFunction<Vector, Matrix, Scalar, Args...>>
           &user_function = std::nullopt) {

  size_t n = X0.rows(); // Dimension of problem
  size_t m = X0.cols(); // Block size

  /// Input sanitation

  if (nev > m)
    throw std::invalid_argument("Block size m must be greater than or equal to "
                                "the number nev of desired eigenpairs");

  if (m > n)
    throw std::invalid_argument("Block size m must be less than or equal to "
                                "the dimension n of the problem");

  /// Preallocate memory

  // Get some useful references to elements of the result struct
  Matrix X = X0; // Matrix of eigenvector estimates
  Vector Theta;  // Vector of eigenvalue estimates (Ritz values)
  Vector r;      // Vector of residual norms

  /// Preallocate memory
  Matrix W(n, m);
  Matrix P(n, m);
  Matrix Q(n, m);
  Matrix AX(n, m);
  Matrix AW(n, m);
  Matrix AP(n, m);
  Matrix BX(n, m);
  Matrix BW(n, m);
  Matrix BP(n, m);
  Matrix R(m, m);
  Matrix Rinv(m, m);
  Matrix Im = Matrix::Identity(m, m);

  Matrix gramA, gramB;

  Matrix CX, CW, CP;

  Matrix V;

  /// Estimate 2-norms of A and B using a random matrix with standard Gaussian
  /// entries, as described in Section 4.3 of the paper "A Robust and Efficient
  /// Implementation of LOBPCG"

  // Sample a Gaussian
  std::default_random_engine gen;
  std::normal_distribution<Scalar> normal(0, 1.0);
  Matrix Omega(n, m);

  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < m; ++j)
      Omega(i, j) = normal(gen);

  Scalar A2normest = A(Omega).norm() / Omega.norm();
  Scalar B2normest = B ? (*B)(Omega).norm() / Omega.norm() : 1.0;

  /// START MAIN LOBPCG LOOP

  // Step 3:  B-orthonormalize X
  BX = B ? (*B)(X) : X;

  // Calculate upper-triangular factor R of Cholesky decomposition of X'BX = R'R
  R = (X.transpose() * BX).llt().matrixU();
  Rinv = R.template triangularView<Eigen::Upper>().solve(Im);
  X = X * Rinv;
  BX = BX * Rinv;
  AX = A(X);

  // Step 4: Compute initial Ritz vectors
  // Solve eigenproblem X'AX = V*Lambda*V'

  Eigen::SelfAdjointEigenSolver<Matrix> XtAX(X.transpose() * AX);
  Theta = XtAX.eigenvalues();
  V = XtAX.eigenvectors();

  X = X * V;
  AX = AX * V;
  BX = BX * V;

  for (num_iters = 0; num_iters < max_iters; ++num_iters) {
    // Step 7: Compute residuals W
    W = AX - BX * Theta.asDiagonal();

    // Calculate residual norms
    r = W.colwise().norm();

    /// TEST STOPPING CRITERIA
    // Here we apply the convergence test described in Section 4.3 of the paper
    // "A Robust and Efficient Implementation of LOBPCG"

    Vector tolerances =
        tau * (A2normest + B2normest * Theta.cwiseAbs().transpose().array()) *
        X.colwise().norm().array();

    // Test how many of the lowest k eigenpairs have converged
    num_converged =
        (r.head(nev).array() <= tolerances.head(nev).array()).count();

    // Call user-supplied function (if one was provided), and test for
    // user-defined stopping criterion
    if (user_function &&
        (*user_function)(num_iters, A, B, T, nev, Theta, X, r, args...))
      break;

    // Test whether the requested number of eigenpairs have converged
    if (num_converged == nev)
      break;

    // Step 9:  Apply preconditioner T (if one was supplied)
    if (T)
      W = (*T)(W);

    // Step 11:  Compute BW and B-orthonormalize W
    BW = B ? (*B)(W) : W;

    // Calculate upper-triangular factor R of Cholesky decomposition of W'BW =
    // R'R
    R = (W.transpose() * BW).llt().matrixU();
    Rinv = R.template triangularView<Eigen::Upper>().solve(Im);
    W = W * Rinv;
    BW = BW * Rinv;

    // Step 12:  Compute AW
    AW = A(W);

    if (num_iters > 0) {
      // Step 14:  B-orthonormalize P
      R = (P.transpose() * BP).llt().matrixU();
      Rinv = R.template triangularView<Eigen::Upper>().solve(Im);
      P = P * Rinv;

      // Step 15:  Update AP
      AP = AP * Rinv;
      BP = BP * Rinv;
    }

    /// Rayleigh-Ritz procedure

    if (num_iters > 0) {

      if (num_iters == 1) {
        gramA.resize(3 * m, 3 * m);
        gramB.resize(3 * m, 3 * m);
      }

      // Step 18:  Fill in the upper-triangular block of gramA
      gramA.block(0, 0, m, m) = Theta.asDiagonal();
      gramA.block(0, m, m, m) = X.transpose() * AW;
      gramA.block(0, 2 * m, m, m) = X.transpose() * AP;

      gramA.block(m, m, m, m) = W.transpose() * AW;
      gramA.block(m, 2 * m, m, m) = W.transpose() * AP;

      gramA.block(2 * m, 2 * m, m, m) = P.transpose() * AP;

      // Step 19:  Fill in the upper-triangular block of gramB
      gramB.block(0, 0, m, m) = Im;
      gramB.block(0, m, m, m) = X.transpose() * BW;
      gramB.block(0, 2 * m, m, m) = X.transpose() * BP;

      gramB.block(m, m, m, m) = Im;
      gramB.block(m, 2 * m, m, m) = W.transpose() * BP;

      gramB.block(2 * m, 2 * m, m, m) = Im;
    } else {
      gramA.resize(2 * m, 2 * m);
      gramB.resize(2 * m, 2 * m);

      // Step 21
      gramA.block(0, 0, m, m) = Theta.asDiagonal();
      gramA.block(0, m, m, m) = X.transpose() * AW;

      gramA.block(m, m, m, m) = W.transpose() * AW;

      // Step 22
      gramB.block(0, 0, m, m) = Im;
      gramB.block(0, m, m, m) = X.transpose() * BW;

      gramB.block(m, m, m, m) = Im;
    }

    // Step 24
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> genEigs(gramA.transpose(),
                                                             gramB.transpose());

    // Extract the smallest m eigenvalues and corresponding eigenvectors
    Theta = genEigs.eigenvalues().head(m);
    V = genEigs.eigenvectors().leftCols(m);

    if (num_iters > 0) {

      // Step 26
      CX = V.topRows(m);
      CW = V.middleRows(m, m);
      CP = V.bottomRows(m);

      // Step 27
      P = W * CW + P * CP;
      AP = AW * CW + AP * CP;
      BP = BW * CW + BP * CP;

      // Step 28
      X = X * CX + P;
      AX = AX * CX + AP;
      BX = BX * CX + BP;
    } else {
      // Step 30
      CX = V.topRows(m);
      CW = V.bottomRows(m);

      // Step 31
      P = W * CW;
      AP = AW * CW;
      BP = BW * CW;

      // Step 32
      X = X * CX + P;
      AX = AX * CX + AP;
      BX = BX * CX + BP;
    }
  } // for(num_iters ... )

  /// Finalize solution
  Theta.conservativeResize(nev);
  X.conservativeResize(Eigen::NoChange, nev);

  return std::make_pair(Theta, X);
}

/** This function estimates the smallest eigenpairs (lambda, x) of the
 * generalized symmetric eigenvalue problem:
 *
 * Ax = lambda * Bx
 *
 * using Knyazev's locally optimal block preconditioned conjugate gradient
 * (LOBPCG) method.  Here:
 *
 * - A is a symmetric linear operator
 * - B is an optional positive-definite symmetric operator.  The absence of this
 *   parameter indicates that B == I (i.e. that we are solving a standard
 *   symmetric eigenproblem)
 * - T is an optional positive-definite symmetric preconditioner that
 *   approximates A^-1 in the sense that the condition number kappa(TA) should
 *   be (much) smaller than kappa(A) itself.  The absence of this parameter
 *   indicates that T == I (i.e. that no preconditioning is applied)
 * - n is the dimension of the eigenvalue problem (order of A, B, T)
 * - m is the block size
 * - nev is the number of desired eigenpairs; note that this number should be
 *   less than the block size m.
 * - max_iters is the maximum number of iterations to perform
 * - num_iters is a return value that gives the number of iterations
 *   executed by the algorithm
 * - num_converged is a return value that indicates the number of eigenpairs
 *   that have converged to the requested precision
 * - tau is the stopping tolerance for the convergence test described in
 *   Section 4.3 of the paper "A Robust and Efficient Implementation of LOBPCG";
 *   specifically, an eigenpair (lambda, x) is considered converged when:
 *
 *   ||Ax - lambda * Bx|| / (||A||_2 + lambda * ||B_2|| * ||x||) <= tau
 *
 * - user_function is an optional user-defined function that can be used to
 *   inspect various variables of interest as the algorithm runs, and
 *   (optionally) to define a custom stopping criterion for the algorithm.
 */
template <typename Vector, typename Matrix, typename Scalar = double,
          typename... Args>
std::pair<Vector, Matrix>
LOBPCG(const SymmetricLinearOperator<Matrix, Args...> &A,
       const std::optional<SymmetricLinearOperator<Matrix, Args...>> &B,
       const std::optional<SymmetricLinearOperator<Matrix, Args...>> &T,
       size_t n, size_t m, size_t nev, size_t max_iters, size_t &num_iters,
       size_t &num_converged, Args &... args, Scalar tau = 1e-6,
       const std::optional<LOBPCGUserFunction<Vector, Matrix, Scalar, Args...>>
           &user_function = std::nullopt) {
  Matrix X0 = Matrix::Random(n, m);

  return LOBPCG<Vector, Matrix, Scalar, Args...>(A, B, T, X0, nev, max_iters,
                                                 num_iters, num_converged,
                                                 args..., tau, user_function);
}
} // namespace LinearAlgebra
} // namespace Optimization
