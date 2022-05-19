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
 * Our implementation follows the paper:
 *
 * "A Robust and Efficient Implementation of LOBPCG", by J.A. Duersch, M. Shao,
 * C. Yang, and M. Gu
 *
 * Note that in *THIS SPECIFIC FILE*, it is assumed that the template parameters
 * 'Matrix' and 'Vector' correspond to dense Matrix and Vector types in the
 * Eigen library.
 *
 * Copyright (C) 2022 by David M. Rosen (d.rosen@northeastern.edu)
 */

#pragma once

#include <limits>
#include <optional>
#include <random>
#include <tuple>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "Optimization/LinearAlgebra/Concepts.h"

namespace Optimization {

namespace LinearAlgebra {

/// The following are helper functions for the main LOBPCG function

/** Given an m x m symmetric positive-definite matrix M and an m x n basis
 * matrix U, this function computes and returns an M-orthonormalized basis
 * matrix V such that range(U) = range(V). */
template <typename Vector, typename Matrix>
Matrix SVQB(const Matrix &M, const Matrix &U) {

  // Construct pairwise inner product matrix U'*M*U
  Matrix UtMU = U.transpose() * M * U;

  // Construct diagonal scaling matrix D to normalize columns of U'MU
  Vector D = UtMU.diagonal().array().cwiseSqrt().cwiseInverse();

  // Calculate symmetric eigendecomposition of D * UtMU * D
  Eigen::SelfAdjointEigenSolver<Matrix> eigs(D.asDiagonal() * UtMU *
                                             D.asDiagonal());

  // Determine eigenvalue threshold t for stabilizing eigenvalues of Theta.  As
  // per the recommendation in the "Robust Implementation" paper, the drop
  // tolerance tau_replace should be a small multiple of the machine precision
  // times the dimension of U
  typename Matrix::Scalar tau_replace =
      100 * U.rows() * std::numeric_limits<typename Matrix::Scalar>::epsilon();

  // Threshold for dropping an eigenvector from the returned basis
  typename Matrix::Scalar t =
      tau_replace * eigs.eigenvalues().cwiseAbs().maxCoeff();

  // Replace any eigenvalue less than t with t
  Vector Theta = eigs.eigenvalues().cwiseMax(t);

  return U * D.asDiagonal() * eigs.eigenvectors() *
         (Theta.cwiseSqrt().cwiseInverse()).asDiagonal();
}

/** Given an m x m symmetric positive-definite matrix M and an m x n
 * matrix U, this function computes and returns an M-orthonormalized basis
 * matrix V such that range(U) = range(V).  Unlike the basic SVQB method, this
 * function does *NOT* assume that the input matrix U is a basis (that is, U may
 * be numerically singular).  In the event that U is numerically rank-deficient,
 * this function will return a nonsingular and well-conditioned basis matrix V
 * for the subspace S = range(U) obtained after *truncating* some of U's
 * columns.
 */
template <typename Vector, typename Matrix>
Matrix SVQBdrop(const Matrix &M, const Matrix &U) {

  // Construct pairwise inner product matrix U'*M*U
  Matrix UtMU = U.transpose() * M * U;

  // Construct diagonal scaling matrix D to normalize columns of U'MU
  Vector D = UtMU.diagonal().array().cwiseSqrt().cwiseInverse();

  // Calculate symmetric eigendecomposition of D * UtMU * D
  Eigen::SelfAdjointEigenSolver<Matrix> eigs(D.asDiagonal() * UtMU *
                                             D.asDiagonal());

  // Determine eigenvalue threshold t for dropping columns from U.  As per the
  // recommendation in the "Robust Implementation" paper, the drop tolerance
  // tau_drop should be a small multiple of the machine precision times the
  // dimension of U
  typename Matrix::Scalar tau_drop =
      100 * U.rows() * std::numeric_limits<typename Matrix::Scalar>::epsilon();

  // Threshold for dropping an eigenvector from the returned basis
  typename Matrix::Scalar t =
      tau_drop * eigs.eigenvalues().cwiseAbs().maxCoeff();

  // Get the indices of the eigenvalues that are greater than the cutoff
  // threshold
  Eigen::Array<bool, Eigen::Dynamic, 1> indicators =
      (eigs.eigenvalues().array() > t);

  // Extract the indices of the columns to keep
  std::vector<int> J;
  J.reserve(indicators.size());
  for (int i = 0; i < indicators.size(); ++i)
    if (indicators(i))
      J.emplace_back(i);

  // Compute and return U * D * V(:, J) * Theta(J, J)
  return U * D.asDiagonal() * eigs.eigenvectors()(Eigen::all, J) *
         eigs.eigenvalues()(J).cwiseSqrt().cwiseInverse().asDiagonal();
}

/** Given an m x m symmetric positive-definite matrix M, an m x nu matrix U, and
 * an external M-orthonormal basis V, this function calculates and returns an
 * M-orthonormal basis W such that the following conditions are satisfied:
 *
 * - [U, W] is an M-orthonormal basis for the subspace range([U, W])
 * - range([V, W]) contains range([U, V])
 *
 * The basis W is obtained by M-orthonormalizing the input matrix U against the
 * external basis V, and dropping columns from U (if necessary) to obtain a
 * well-conditioned basis.
 */
template <typename Vector, typename Matrix>
Matrix orthoDrop(const Matrix &M, const Matrix &U, const Matrix &V) {

  // As per the recommendation in the "Robust Implementation" paper, the drop
  // tolerance tau_ortho should be a small multiple of the machine precision
  typename Matrix::Scalar tau_ortho =
      100 * std::numeric_limits<typename Matrix::Scalar>::epsilon();

  // Initialize candidate basis
  Matrix W = U;

  // Calculate and cache product MV
  const Matrix MV = M * V;
  double MVnorm = MV.norm(); // Frobenius norm of  MV

  // Cache variable for product MW
  Matrix MW = M * W;

  for (int i = 0; i < 3; ++i) {

    // Outer loop: M-orthogonalize W against V
    W = W - V * (V.transpose() * M * W);

    for (int j = 0; j < 3; ++j) {

      // Inner loop: M-orthonormalize W
      if (j == 0)
        W = SVQB<Vector, Matrix>(M, W);
      else
        W = SVQBdrop<Vector, Matrix>(M, W);

      // Update cache variable MW
      MW = M * W;

      // Test stopping condition: ||W'MU - I|| < ||MW|| * ||W|| * t_ortho
      // Note that in the following expressions we use the FROBENIUS norm,
      // rather than the SPECTRAL norm, in order to avoid calculating (another)
      // expensive eigendecomposition of W'MW.  Note that in consequence, we
      // scale tau_ortho by W.size(), the total number of elements in W
      if ((W.transpose() * MW - Matrix::Identity(W.cols(), W.cols())).norm() <
          MW.norm() * W.norm() * W.size() * tau_ortho)
        break;
    } // inner loop

    // Test stopping condition: ||V'MW|| < ||MV||* ||W|| * t_ortho
    // Note that in the following expressions we use the FROBENIUS norm,
    // rather than the SPECTRAL norm, in order to avoid calculating (another)
    // expensive singular value decomposition of W'MW.  Note that in
    // consequence, we scale tau_ortho by W.size(), the total number of elements
    // in W
    if ((W.transpose() * MV).norm() < MVnorm * W.norm() * W.size() * tau_ortho)
      break;
  }

  return W;
}

/** This function implements the basic Rayleigh-Ritz procedure: Given two
 * symmetric n x n matrices A and B, with B positive-definite, this function
 * computes and returns an n x n matrix C and a vector of eigenvalues Theta
 * satisfying
 *
 * C'AC = Theta
 * C'BC = I_n
 *
 * In a basic LOBPCG implementation, A would be replaced by S'AS and B would be
 * replaced by S'BS, where S is a basis matrix for the search space.
 */
template <typename Vector, typename Matrix>
std::pair<Vector, Matrix> RayleighRitz(const Matrix &A, const Matrix &B) {
  // Compute diagonal scaling matrix to equilibrate B
  Vector D = A.diagonal().cwiseSqrt().cwiseInverse();

  Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> eig(
      D.asDiagonal() * A * D.asDiagonal(), D.asDiagonal() * B * D.asDiagonal());

  return std::make_pair(eig.eigenvalues(), D.asDiagonal() * eig.eigenvectors());
}

/** This function implements the modified Rayleigh-Ritz procedure (with improved
 * basis selection) described in Algorithm 7 of the "Robust Implementation"
 * paper.  Here:
 *
 * - StAS and StBS are the matrix products S'AS and S'BS, where S is a n x ns
 *   basis matrix for the LOBPCG search space
 * - nx is the number of desired eigenpairs
 * - nc is the number of converged eigenpairs from the previous
 *   iteration; this is used to implement the "soft locking" procedure for
 *   converged eigenpairs
 * - useOrtho is a Boolean value indicating whether S'BS = I
 *
 * This function returns a matrix C = [Cx, Cp] of dimension ns x (2*nx - nc)
 * and a matrix Theta of Ritz values satisfying
 *
 * C'(S'AS)C = Theta, C'(S'BS)C = I
 */
template <typename Vector, typename Matrix>
std::tuple<Matrix, Matrix, bool>
ModifiedRayleighRitz(const Matrix &StAS, const Matrix &StBS, size_t nx,
                     size_t nc, bool useOrtho) {

  double tau_skip = 1e16;

  // Get dimension of search space
  size_t ns = StAS.rows();

  // Preallocate output matrix C
  Matrix C(ns, 2 * nx - nc);

  // A vector to cache the original Ritz values
  Vector ritz_vals;

  // Preallocate output matrix Theta of appropriate dimension
  Matrix Theta = Matrix::Zero(2 * nx - nc, 2 * nx - nc);

  // Preallocate orthogonal matrix Q1p^T from LQ factorization
  Matrix Q1pT;

  bool useOrtho_out = useOrtho;

  if (useOrtho) {

    useOrtho_out = true;
    // Compute symmetric eigendecomposition S' A S = Z Theta Z'

    // NB:  Here is it important that the eigenvalues are sorted in *INCREASING
    // ORDER* -- this means that the sought (algebraically-smallest) eigenpairs
    // will be ordered *FIRST*
    Eigen::SelfAdjointEigenSolver<Matrix> eig(StAS);

    // Cache eigenvalues
    ritz_vals = eig.eigenvalues();

    // Partition Z
    //
    //
    //  Z = [Z1 Z1p]
    //      [Z1 Z2p]
    //
    // and construct an LQ factorization L1p * Q1p = Z1p  Note that this is
    // equivalent to the QR factorization:
    //
    // Q1p^T * L1p^T = Z1p^T

    const auto &Z = eig.eigenvectors();

    // Compute QR factorization of Z1p^T
    Eigen::HouseholderQR<Matrix> qr(Z.topRightCorner(nx, ns - nx).transpose());

    Q1pT = qr.householderQ();

    // Set Cx
    C.leftCols(nx) = Z.leftCols(nx);
    C.rightCols(nx - nc) = Z.rightCols(ns - nx) * Q1pT.leftCols(nx - nc);

  } else {
    // Extract diagonal equilibration vector
    Vector D = StBS.diagonal().cwiseSqrt().cwiseInverse();

    // Calculate equilibrated matrix
    Matrix DStBSD = D.asDiagonal() * StBS * D.asDiagonal();

    // Calculate conditioning of DStBSD using SVD
    Eigen::JacobiSVD<Matrix> svd(DStBSD);
    double cond =
        svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();

    if (cond > tau_skip) {
      // Set useOrtho_out = true and exit
      return std::make_tuple(Theta, C, true);
    }

    // Solve generalized eigenvalue problem
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> eig(
        D.asDiagonal() * StAS * D.asDiagonal(), DStBSD);

    // Extract and cache eigenvalues
    ritz_vals = eig.eigenvalues();

    // Partition Z
    //
    //
    //  Z = [Z1 Z1p]
    //      [Z1 Z2p]
    //
    // and construct an LQ factorization L1p * Q1p = Z1p  Note that this is
    // equivalent to the QR factorization:
    //
    // Q1p^T * L1p^T = Z1p^T

    const auto &Z = eig.eigenvectors();

    // Compute QR factorization of Z1p^T
    Eigen::HouseholderQR<Matrix> qr(Z.topRightCorner(nx, ns - nx).transpose());

    Q1pT = qr.householderQ();

    // Set Cx
    C.leftCols(nx) = D.asDiagonal() * Z.leftCols(nx);
    C.rightCols(nx - nc) =
        D.asDiagonal() * Z.rightCols(ns - nx) * Q1pT.leftCols(nx - nc);
  }

  // Construct and return block diagonal matrix Theta
  Theta.topLeftCorner(nx, nx) = ritz_vals.head(nx).asDiagonal();
  Theta.bottomRightCorner(nx - nc, nx - nc) =
      Q1pT.leftCols(nx - nc).transpose() *
      ritz_vals.tail(ns - nx).asDiagonal() * Q1pT.leftCols(nx - nc);

  return std::make_tuple(Theta, C, useOrtho_out);
}

/** An alias template for a user-definable function that can be used to
 * access various interesting bits of information about the internal state
 * of the LOBPCG algorithm as it runs.  Here:
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
 * This function is called once per iteration, after the Rayleigh-Ritz
 * procedure is used to update the current Ritz pairs and their
 * corresponding residuals
 *
 * This function may also return the Boolean value 'true' in order to
 * terminate the LOBPCG algorithm; this provides a convenient means of
 * implementing a custom (user-definable) stopping criterion.
 */
template <typename Vector, typename Matrix, typename Scalar = double,
          typename... Args>
using LOBPCGUserFunction = std::function<bool(
    size_t i, const SymmetricLinearOperator<Matrix, Args...> &A,
    const std::optional<SymmetricLinearOperator<Matrix, Args...>> &B,
    const std::optional<SymmetricLinearOperator<Matrix, Args...>> &T,
    size_t nev, const Vector &Theta, const Matrix &X, const Vector &residuals,
    Args &...args)>;

/** This function estimates the smallest eigenpairs (lambda, x) of the
 * generalized symmetric eigenvalue problem:
 *
 * Ax = lambda * Bx
 *
 * using Knyazev's locally optimal block preconditioned conjugate gradient
 * (LOBPCG) method.  Here:
 *
 * - A is a symmetric linear operator
 * - B is an optional positive-definite symmetric operator.  The absence of
 *   this parameter indicates that B == I (i.e. that we are solving a standard
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
 *   Section 4.3 of the paper "A Robust and Efficient Implementation of
 * LOBPCG"; specifically, an eigenpair (lambda, x) is considered converged
 * when:
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
       size_t &num_converged, Args &...args, Scalar tau = 1e-6,
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
  /// entries, as described in Section 4.3 of the paper "A Robust and
  /// Efficient Implementation of LOBPCG"

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

  // Calculate upper-triangular factor R of Cholesky decomposition of X'BX =
  // R'R
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
    // Here we apply the convergence test described in Section 4.3 of the
    // paper "A Robust and Efficient Implementation of LOBPCG"

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
 * - B is an optional positive-definite symmetric operator.  The absence of
 * this parameter indicates that B == I (i.e. that we are solving a standard
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
 *   Section 4.3 of the paper "A Robust and Efficient Implementation of
 * LOBPCG"; specifically, an eigenpair (lambda, x) is considered converged
 * when:
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
       size_t &num_converged, Args &...args, Scalar tau = 1e-6,
       const std::optional<LOBPCGUserFunction<Vector, Matrix, Scalar, Args...>>
           &user_function = std::nullopt) {
  Matrix X0 = Matrix::Random(n, m);

  return LOBPCG<Vector, Matrix, Scalar, Args...>(A, B, T, X0, nev, max_iters,
                                                 num_iters, num_converged,
                                                 args..., tau, user_function);
}
} // namespace LinearAlgebra
} // namespace Optimization
