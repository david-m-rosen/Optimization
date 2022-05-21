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

/** Given an m x m symmetric positive-definite linear operator M and an m x n
 * basis matrix U, this function computes and returns an M-orthonormalized basis
 * matrix V such that range(U) = range(V).
 *
 * Note that this function does not access M directly, but instead accepts as
 * input the product MU, which is assumed to be calculated externally.
 */
template <typename Vector, typename Matrix>
Matrix SVQB(const Matrix &U, const Matrix &MU) {

  // Construct pairwise inner product matrix U'*M*U
  Matrix UtMU = U.transpose() * MU;

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

/** Given an m x m symmetric positive-definite linear operator M and an m x n
 * matrix U, this function computes and returns an M-orthonormalized basis
 * matrix V such that range(U) = range(V).  Unlike the basic SVQB method, this
 * function does *NOT* assume that the input matrix U is a basis (that is, U may
 * be numerically singular).  In the event that U is numerically rank-deficient,
 * this function will return a nonsingular and well-conditioned basis matrix V
 * for the subspace S = range(U) obtained after *truncating* some of U's
 * columns.
 *
 * Note that this function does not access M directly, but instead accepts as
 * input the product MU, which is assumed to be calculated externally.
 */
template <typename Vector, typename Matrix>
Matrix SVQBdrop(const Matrix &U, const Matrix &MU) {

  // Construct pairwise inner product matrix U'*M*U
  Matrix UtMU = U.transpose() * MU;

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

/** Given an m x m symmetric positive-definite operator M, an m x nu matrix U,
 * and an external M-orthonormal basis V, this function calculates and returns
 * an M-orthonormal basis W such that the following conditions are satisfied:
 *
 * - [U, W] is an M-orthonormal basis for the subspace range([U, W])
 * - range([V, W]) contains range([U, V])
 *
 * The basis W is obtained by B-orthonormalizing the input matrix U against the
 * external basis V, and dropping columns from U (if necessary) to obtain a
 * well-conditioned basis.
 *
 * Note that the following function accepts the operator M in the form of a
 * std::optional<SymmetricLinearOperator>; if this optional is not set, it is
 * interpreted as M = I.
 *
 */

template <typename Vector, typename Matrix, typename... Args>
Matrix
orthoDrop(const Matrix &U, const Matrix &V,
          const std::optional<SymmetricLinearOperator<Matrix, Args...>> &M,
          Args &...args) {

  // As per the recommendation in the "Robust Implementation" paper, the drop
  // tolerance tau_ortho should be a small multiple of the machine precision
  typename Matrix::Scalar tau_ortho =
      100 * std::numeric_limits<typename Matrix::Scalar>::epsilon();

  // Initialize candidate basis
  Matrix W = U;

  // Calculate and cache product MV
  const Matrix MV = M ? (*M)(V) : V;
  double MVnorm = MV.norm(); // Frobenius norm of  MV

  // Cache for product matrix MW
  Matrix MW;

  for (int i = 0; i < 3; ++i) {

    // Outer loop: M-orthogonalize W against V
    W = W - V * (MV.transpose() * W);

    for (int j = 0; j < 3; ++j) {

      // Update product MW
      MW = M ? (*M)(W) : W;

      // Test stopping condition: ||W'MU - I|| < ||MW|| * ||W|| * t_ortho
      // Note that in the following expressions we use the FROBENIUS norm,
      // rather than the SPECTRAL norm, in order to avoid calculating (another)
      // expensive eigendecomposition of W'MW.  Note that in consequence, we
      // scale tau_ortho by W.size(), the total number of elements in W
      if ((W.transpose() * MW - Matrix::Identity(W.cols(), W.cols())).norm() <
          MW.norm() * W.norm() * W.size() * tau_ortho)
        break;

      // Inner loop: M-orthonormalize W
      W = (j == 0 ? SVQB<Vector, Matrix>(W, MW)
                  : SVQBdrop<Vector, Matrix>(W, MW));

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
 * computes and returns a pair of n x n matrices [Theta, C] satisfying
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
  Vector D = B.diagonal().cwiseSqrt().cwiseInverse();

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
 * This function returns a tuple [Thetax, Thetap, Cx, Cp, useOrtho] consisting
 * of:
 *
 * - A vector Thetax consisting of updated Ritz values
 * - A matrix Thetap consisting of partial inner products
 * - An ns x nx transformation matrix Cx for updating the eigenvector estimates
 * - An ns x (nx - nc) transformation matrix Cp for updating the search
 *
 * Defining C = [Cx, Cp] and Theta Diag(Thetax, Thetap), the return values for
 * this function should satisfy the equations:
 *
 * C'(S'AS)C = Theta, C'(S'BS)C = I
 */
template <typename Vector, typename Matrix>
std::tuple<Vector, Matrix, Matrix, Matrix, bool>
ModifiedRayleighRitz(const Matrix &StAS, const Matrix &StBS, size_t nx,
                     size_t nc, bool useOrtho) {

  double tau_skip = 1e16;

  // Get dimension of search space
  size_t ns = StAS.rows();

  // Preallocate outputs
  Vector Thetax(nx);
  Matrix Thetap = Matrix::Zero(nx - nc, nx - nc);

  // Preallocate output matrix C
  Matrix Cx(ns, nx);
  Matrix Cp(ns, nx - nc);

  // A vector to cache the original Ritz values
  Vector ritz_vals;

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
    Cx = Z.leftCols(nx);
    Cp = Z.rightCols(ns - nx) * Q1pT.leftCols(nx - nc);

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
      return std::make_tuple(Thetax, Thetap, Cx, Cp, true);
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
    Cx = D.asDiagonal() * Z.leftCols(nx);
    Cp = D.asDiagonal() * Z.rightCols(ns - nx) * Q1pT.leftCols(nx - nc);
  }

  // Construct and return block diagonal matrix Theta
  Thetax = ritz_vals.head(nx);
  Thetap = Q1pT.leftCols(nx - nc).transpose() *
           ritz_vals.tail(ns - nx).asDiagonal() * Q1pT.leftCols(nx - nc);

  return std::make_tuple(Thetax, Thetap, Cx, Cp, useOrtho_out);
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
 * - X0 is an m x nx matrix containing an initial set of linearly-independent
 *   eigenvector estimates.
 * - nev is the number of desired eigenpairs; note that this number should be
 *   less than the block size nx.
 * - max_iters is the maximum number of iterations to perform
 * - num_iters is a return value that gives the number of iterations
 *   executed by the algorithm
 * - nc is a return value that indicates the number of eigenpairs
 *   that have converged to the requested precision
 * - tau is the stopping tolerance for the convergence test described in
 *   Section 4.3 of the paper "A Robust and Efficient Implementation of
 *   LOBPCG"; specifically, an eigenpair (lambda, x) is considered converged
 *   when:
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
       size_t &nc, Args &...args, Scalar tau = 1e-6,
       const std::optional<LOBPCGUserFunction<Vector, Matrix, Scalar, Args...>>
           &user_function = std::nullopt) {

  size_t m = X0.rows();  // Dimension of problem
  size_t nx = X0.cols(); // Block size
  size_t Pcols = 0;

  // Column dimension of current search space basis matrix S
  size_t ns;

  /// Input sanitation

  if (nev > nx)
    throw std::invalid_argument(
        "Block size nx must be greater than or equal to "
        "the number nev of desired eigenpairs");

  if (nx > m)
    throw std::invalid_argument("Block size nx must be less than or equal to "
                                "the dimension m of the problem");

  /// Preallocate memory

  // Get some useful references to elements of the result struct
  Matrix X = X0; // Matrix of eigenvector estimates

  // In our implementation of LOBPCG, Theta is a matrix of the form
  //
  // Theta = [Thetax,    0  ]
  //         [  0     Thetap]
  //
  // where:
  //  - Thetax is a diagonal matrix of dimension nx containing estimated Ritz
  //    pairs
  //  - Thetap is a block matrix of size (nx - nc) x (nx - nc)
  Vector Thetax;
  Matrix Thetap;

  // Change of basis matrix C = [Cx, Cp] used to B-orthonormalize subspace basis
  // S and (partially) diagonalize S'AS.  This matrix satisfies:
  // - C'(S'BS)C = I
  // - C'(S'AS)C = Theta
  Matrix Cx, Cp;

  // Cache variables for various matrix products
  Matrix AX, BX;

  // Matrix of residuals A*X - B*Theta
  Matrix R;

  // Matrix of preconditioned residuals -- these will be orthonormalized to get
  // the matrix of search directions W
  Matrix TR;

  // Matrix of *orthonormalized* preconditioned search directions
  Matrix W;

  // Search space basis matrix S = [X, P, W]
  Matrix S(m, 3 * nx); // We preallocate the maximum needed size

  // Cache variables for products AS and BS
  // NB:  We initialize these to their maximum possible size
  Matrix AS(m, 3 * nx);
  Matrix BS(m, 3 * nx);

  // Gram matrices S'AS, B'AS
  Matrix StAS(3 * nx, 3 * nx);
  Matrix StBS(3 * nx, 3 * nx);

  // Vector of residual norms: ri = ||Ri||, the norm of the ith column of R
  Vector r;

  /// Estimate 2-norms of A and B using a random matrix with standard Gaussian
  /// entries, as described in Section 4.3 of the paper "A Robust and Efficient
  /// Implementation of LOBPCG".  These quantities are used in the
  /// scale-invariant backward-stable termination criterion.

  // Sample a Gaussian
  std::default_random_engine gen;
  std::normal_distribution<Scalar> normal(0, 1.0);
  Matrix Omega(m, nx);

  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < nx; ++j)
      Omega(i, j) = normal(gen);

  Scalar A2normest = A(Omega).norm() / Omega.norm();
  Scalar B2normest = B ? (*B)(Omega).norm() / Omega.norm() : 1.0;

  /// INITIALIZATION

  AX = A(X);
  BX = B ? (*B)(X) : X;

  // B-orthonormalize columns of X using standard Rayleigh Ritz procedure
  std::tie(Thetax, Cx) =
      RayleighRitz<Vector, Matrix>(X.transpose() * AX, X.transpose() * BX);

  // In-place update AX and BX
  AX = AX * Cx;
  BX = BX * Cx;

  // Compute residuals
  R = AX - BX * Thetax.asDiagonal();

  // Initialize S
  S.leftCols(nx) = X;

  // Initialize number of converged eigenpairs
  nc = 0;

  bool useOrtho_prev = false;
  bool useOrtho = false;

  /// MAIN LOOP

  for (num_iters = 0; num_iters < max_iters; ++num_iters) {

    /// LOOP INVARIANTS: At the start of each iteration, the following hold:
    ///
    /// Pcols is initialized
    ///
    /// S = [X, P, *], where X and P are the values to be used in the current
    /// iteration
    ///
    /// X, AX, BX, and Thetax are initialized to their values for the current
    /// iteration
    ///
    /// Residuals R = AX - BX*Thetax are initialized to their values for the
    /// current iteration
    ///
    /// useOrtho = useOrtho_prev

    /// COMPUTE ORTHONORMALIZED BASIS OF PRECONDITIONED SEARCH DIRECTIONS W

    // Compute preconditioned residuals (search directions), if a preconditioner
    // was supplied
    TR = T ? (*T)(R) : R;

    // B-orthonormalize the preconditioned search residuals TR against the
    // partial search space basis X, P
    W = useOrtho ? orthoDrop<Vector, Matrix, Args...>(
                       TR, S.leftCols(nx + Pcols), B, args...)
                 : TR;

    // Set final block of search space basis matrix S = [X, P, *] to W
    S.middleCols(nx + Pcols, W.cols()) = W;

    // Set search space dimension
    ns = nx + Pcols + W.cols();

    // Update AS, BS
    AS.leftCols(nx) = AX;
    AS.middleCols(nx, ns - nx) = A(S.middleCols(nx, ns - nx));

    BS.leftCols(nx) = BX;
    BS.middleCols(nx, ns - nx) =
        B ? (*B)(S.middleCols(nx, ns - nx)) : S.middleCols(nx, ns - nx);

    /// Update Gram matrices
    StAS.topLeftCorner(ns, ns) = S.leftCols(ns).transpose() * AS.leftCols(ns);
    StBS.topLeftCorner(ns, ns) = S.leftCols(ns).transpose() * BS.leftCols(ns);

    std::tie(Thetax, Thetap, Cx, Cp, useOrtho) =
        ModifiedRayleighRitz<Vector, Matrix>(StAS.topLeftCorner(ns, ns),
                                             StBS.topLeftCorner(ns, ns), nx, nc,
                                             useOrtho_prev);

    if (useOrtho && !useOrtho_prev) {
      // useOrtho just "switched on" for the first time

      // B-orthonormalize preconditioned residuals TR against partial search
      // space basis [X, P]
      W = orthoDrop<Vector, Matrix, Args...>(TR, S.leftCols(nx + Pcols), B,
                                             args...);

      // Update search space basis S with this new set of B-orthonormalized
      // preconditioned search directions W
      S.middleCols(nx + Pcols, W.cols()) = W;

      // Update search space dimension
      ns = nx + Pcols + W.cols();

      // Update AS, BS with this new choice of W
      AS.middleCols(nx + Pcols, W.cols()) = A(W);
      BS.middleCols(nx + Pcols, W.cols()) = B ? (*B)(W) : W;

      /// Update Gram matrices -- this is a bit wasteful, since we're
      /// recomputing the *ENTIRE* Gram matrices, but since this block executes
      /// AT MOST once per algorithm execution, it's probably fine XD
      StAS.topLeftCorner(ns, ns) = S.leftCols(ns).transpose() * AS.leftCols(ns);
      StBS.topLeftCorner(ns, ns) = S.leftCols(ns).transpose() * BS.leftCols(ns);

      std::tie(Thetax, Thetap, Cx, Cp, useOrtho) =
          ModifiedRayleighRitz<Vector, Matrix>(StAS.topLeftCorner(ns, ns),
                                               StBS.topLeftCorner(ns, ns), nx,
                                               nc, useOrtho);
    }

    // Cache the current value of useOrtho
    useOrtho_prev = useOrtho;

    // Update eigenvector estimates X
    X = S.leftCols(ns) * Cx;

    // Update P-block of S
    S.middleCols(nx, Cp.cols()) = S.leftCols(ns) * Cp;
    Pcols = Cp.cols();

    // Update AX and BX
    AX = A(X);
    BX = B ? (*B)(X) : X;

    // Update residuals
    R = AX - BX * Thetax.asDiagonal();

    /// TEST STOPPING CRITERIA

    // Calculate residual norms
    r = R.colwise().norm();

    // Here we apply the convergence test described in Section 4.3 of
    // the paper "A Robust and Efficient Implementation of LOBPCG"

    Vector tolerances =
        tau * (A2normest + B2normest * Thetax.cwiseAbs().transpose().array()) *
        X.colwise().norm().array();

    // Calculate vector of Boolean indicators for the convergence of
    // each eigenpair
    const auto converged =
        (r.head(nev).array() <= tolerances.head(nev).array());

    // Find the largest number k such that converged[i] = true for
    // all i <= k; this gives the number of converged eigenvectors.
    // Note that this may be less than the *TOTAL* number of
    // eigenpairs for which converged = true; this difference
    // accounts for the fact that eigenpairs must be soft-locked IN
    // ORDER: that is, we may apply soft locking only to the first
    // *CONTINGUOUS BLOCK* of converged eigenpairs
    for (nc = 0; nc < nev; ++nc)
      if (!converged[nc])
        break;

    // Call user-supplied function (if one was provided), and test
    // for user-defined stopping criterion
    if (user_function &&
        (*user_function)(num_iters, A, B, T, nev, Thetax, X, r, args...))
      break;

    // Test whether the requested number of eigenpairs have converged
    if (nc == nev)
      break;

  } // for(num_iters ... )

  /// Finalize solution
  Thetax.conservativeResize(nev);
  X.conservativeResize(Eigen::NoChange, nev);

  return std::make_pair(Thetax, X);
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
 * - m is the dimension of the eigenvalue problem (order of A, B, T)
 * - nx is the block size
 * - nev is the number of desired eigenpairs; note that this number should be
 *   less than the block size m.
 * - max_iters is the maximum number of iterations to perform
 * - num_iters is a return value that gives the number of iterations
 *   executed by the algorithm
 * - num_converged is a return value that indicates the number of eigenpairs
 *   that have converged to the requested precision
 * - tau is the stopping tolerance for the convergence test described in
 *   Section 4.3 of the paper "A Robust and Efficient Implementation of
 *   LOBPCG"; specifically, an eigenpair (lambda, x) is considered converged
 *   when:
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
       size_t m, size_t nx, size_t nev, size_t max_iters, size_t &num_iters,
       size_t &nc, Args &...args, Scalar tau = 1e-6,
       const std::optional<LOBPCGUserFunction<Vector, Matrix, Scalar, Args...>>
           &user_function = std::nullopt) {
  Matrix X0 = Matrix::Random(m, nx);

  return LOBPCG<Vector, Matrix, Scalar, Args...>(
      A, B, T, X0, nev, max_iters, num_iters, nc, args..., tau, user_function);
}

} // namespace LinearAlgebra
} // namespace Optimization
