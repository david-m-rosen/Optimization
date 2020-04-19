/** This unit test exercises the functionality of the iterative linear-algebraic
 * methods in the IterativeSolvers file */

#include "Optimization/LinearAlgebra/IterativeSolvers.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>

#include <gtest/gtest.h>

// Typedef for the numerical type will use in the following tests
typedef double Scalar;

// Typedefs for Eigen matrix types
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
typedef Eigen::SparseMatrix<Scalar, Eigen::ColMajor> SparseMatrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

/// Dimensions of test problems
constexpr size_t small_dim = 3;
constexpr size_t large_dim = 1000;
constexpr size_t num_constraints = 100;

/// Test tolerances
Scalar eps_abs = 1e-6;
Scalar eps_rel = 1e-6;

/// Test fixture for testing the Steihaug-Toint preconditioned conjugate
/// gradient solver
class STPCGUnitTest : public testing::Test {
protected:
  /// Set up variables and functions used in the following tests

  // Inner product between Vectors
  Optimization::LinearAlgebra::InnerProduct<Vector> inner_product;

  // Input vector  g -- right-hand side of the equation Hv = -g we're trying to
  // solve in each test instance
  Vector small_g;
  Vector large_g;

  // Positive-definite Hessian matrices
  DiagonalMatrix small_P;
  DiagonalMatrix large_P;

  // Negative-definite Hessian matrices
  DiagonalMatrix small_N;

  // Preconditioning matrices
  DiagonalMatrix small_M;
  DiagonalMatrix large_M;

  // Hessian operators corresponding to P and N
  Optimization::LinearAlgebra::SymmetricLinearOperator<Vector> small_Pop;
  Optimization::LinearAlgebra::SymmetricLinearOperator<Vector> large_Pop;
  Optimization::LinearAlgebra::SymmetricLinearOperator<Vector> small_Nop;

  // Preconditioning operators
  Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector> small_MinvOp;
  Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector> large_MinvOp;

  /// method

  // Trust region radius
  Scalar Delta;

  // Truncation stopping criteria
  Scalar kappa;
  Scalar theta;
  size_t max_iterations;

  // Additional returns from STPCG
  Scalar update_step_norm;
  size_t num_iterations;

  virtual void SetUp() {

    // Inner product
    inner_product = [](const Vector &X, const Vector &Y) { return X.dot(Y); };

    // Right-hand side g
    small_g.resize(small_dim);
    small_g << 21, -.4, 19;
    large_g.resize(large_dim);
    large_g.setRandom();

    /// Termination criteria for the Steihaug-Toint truncated conjugate gradient
    /// solver

    /// Positive-definite Hessian matrices

    // Positive-definite matrix P
    // Manually-constructed small positive-definite matrix
    small_P.resize(small_dim);
    small_P.diagonal() << 1000, 100, 1;
    small_Pop = [&](const Vector &X) { return small_P * X; };

    // Sample a large PD Hessian (note that here we are using the fact that
    // Eigen's Random operator returns values in the range [-1, 1]
    large_P =
        (2000 * Vector::Ones(large_dim) + 1000 * Vector::Random(large_dim))
            .asDiagonal();

    large_Pop = [&](const Vector &X) { return large_P * X; };

    /// Negative-definite Hessian matrices

    small_N = -1 * small_P;
    small_Nop = [&](const Vector &X) { return small_N * X; };

    /// Preconditioning operators

    // Manually-constructed small positive-definite preconditioner
    small_M = DiagonalMatrix(small_dim);
    small_M.diagonal() << 100, 10, 1;
    small_MinvOp = [&](const Vector &v) -> std::pair<Vector, Vector> {
      return std::make_pair(small_M.inverse() * v, Vector());
    };

    // Randomly-sampled large positive-definite preconditioner
    large_M =
        (2000 * Vector::Ones(large_dim) + 1000 * Vector::Random(large_dim))
            .asDiagonal();

    large_MinvOp = [&](const Vector &v) -> std::pair<Vector, Vector> {
      return std::make_pair(large_M.inverse() * v, Vector());
    };
  }
};

/// Test STPCG with a small positive-definite Hessian, extremely tight
/// (effectively exact) stopping tolerance, and a numerically-infinite
/// trust-region radius, to ensure that the underlying CG method is implemented
/// properly
TEST_F(STPCGUnitTest, ExactSTPCG) {

  kappa = 1e-8;
  theta = .999;
  Delta = std::numeric_limits<Scalar>::max();

  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      small_g, small_Pop, inner_product, update_step_norm, num_iterations,
      Delta, small_dim, kappa, theta);

  // Compute ground-truth solution
  Vector s_gt = -(small_P.inverse() * small_g);

  // Compute error in returned step s
  Scalar error = (s - s_gt).norm();

  // Check the solution
  EXPECT_LT(error, eps_abs);

  // Check that the reported update step norm is correct
  EXPECT_LT(fabs(update_step_norm - s.norm()) / s.norm(), eps_rel);
}

/// Test STPCG with a small negative-definite Hessian, extremely tight
/// (effectively exact) stopping tolerance, and a numerically-infinite
/// trust-region radius, to ensure that the step to the boundary of the
/// trust-region is computed properly
TEST_F(STPCGUnitTest, ExactSTPCGwithNegativeCurvature) {

  kappa = 1e-8;
  theta = .999;
  Delta = 1000;

  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      small_g, small_Nop, inner_product, update_step_norm, num_iterations,
      Delta, small_dim, kappa, theta);

  // Compute ground-truth solution
  Vector s_gt = -((Delta / small_g.norm()) * small_g);

  // Compute error in returned step s
  Scalar error = (s - s_gt).norm();

  // Check the solution
  EXPECT_LT(error, eps_abs);

  // Check that the reported update step norm is correct
  EXPECT_LT(fabs(update_step_norm - s.norm()) / s.norm(), eps_rel);
}

/// Test STPCG using a positive-definite Hessian and a (nontrivial)
/// positive-definite preconditioning operator
TEST_F(STPCGUnitTest, ExactSTPCGwithPreconditioning) {

  kappa = 1e-8;
  theta = .999;
  Delta = std::numeric_limits<Scalar>::max();

  std::experimental::optional<
      Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector>>
      precon_op(small_MinvOp);

  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      small_g, small_Pop, inner_product, update_step_norm, num_iterations,
      Delta, small_dim, kappa, theta, precon_op);

  // Compute ground-truth solution
  Vector s_gt = -(small_P.inverse() * small_g);

  // Compute error in returned step s
  Scalar error = (s - s_gt).norm();

  // Check the solution
  EXPECT_LT(error, eps_abs);

  // Check that the reported update step norm is correct
  Scalar s_M_norm = sqrt(s.dot(small_M * s));
  EXPECT_LT(fabs((update_step_norm - s_M_norm) / s_M_norm), eps_rel);
}

/// Test STPCG using a negative-definite Hessian and a (nontrivial)
/// positive-definite preconditioning operator
TEST_F(STPCGUnitTest, ExactSTPCGwithNegativeCurvatureAndPreconditioning) {

  kappa = 1e-8;
  theta = .999;
  Delta = 1000;

  std::experimental::optional<
      Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector>>
      precon_op(small_MinvOp);

  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      small_g, small_Nop, inner_product, update_step_norm, num_iterations,
      Delta, small_dim, kappa, theta, precon_op);

  /// Compute ground-truth solution
  // Compute initial search direction
  Vector p = -(small_M.inverse() * small_g);
  // Compute M-norm of initial search direction
  Scalar p_M_norm = sqrt(p.dot(small_M * p));
  // Scale this search direction so that |alpha*p|_M = Delta
  Vector s_gt = (Delta / p_M_norm) * p;

  // Compute error in returned step s
  Scalar error = (s - s_gt).norm();

  // Check the solution
  EXPECT_LT(error, eps_abs);

  // Check that the reported update step norm is correct
  Scalar s_M_norm = sqrt(s.dot(small_M * s));
  EXPECT_LT(fabs(update_step_norm - s_M_norm) / s_M_norm, eps_rel);
}

/// Test a large-scale instance of the problem using truncation
TEST_F(STPCGUnitTest, STPCGwithTruncation) {

  kappa = .1;
  theta = .7;
  Delta = 1000;

  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      large_g, large_Pop, inner_product, update_step_norm, num_iterations,
      Delta, small_dim, kappa, theta);

  // Compute norm of residual
  Scalar r_norm = (large_g + large_P * s).norm();

  // Compute ratio of the final residual to the initial residual
  Scalar relative_residual = r_norm / large_g.norm();

  // Check that the final residual achieves the targeted relative reduction
  EXPECT_LT(relative_residual, kappa);

  // Check that the reported update step norm is correct
  EXPECT_LT(fabs(update_step_norm - s.norm()) / s.norm(), eps_rel);
}

/// Test a large-scale instance of the problem using preconditioning and
/// truncation
TEST_F(STPCGUnitTest, STPCGwithPreconditioningAndTruncation) {

  kappa = .1;
  theta = .7;
  Delta = 1000;

  std::experimental::optional<
      Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector>>
      precon_op(large_MinvOp);

  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      large_g, large_Pop, inner_product, update_step_norm, num_iterations,
      Delta, large_dim, kappa, theta, precon_op);

  // Compute the Minv-norm of the intial residual large_g
  Scalar gMinv_norm = sqrt(large_g.dot(large_M.inverse() * large_g));

  // Compute Minv-norm of the final residual r := g + Hs
  Vector r = large_g + large_P * s;
  Scalar rMinv_norm = sqrt(r.dot(large_M.inverse() * r));

  // Compute ratio of the final residual to the initial residual
  Scalar relative_residual = rMinv_norm / gMinv_norm;

  // Check that the final residual achieves the targeted relative reduction
  EXPECT_LT(relative_residual, kappa);

  // Check that the reported update step norm is correct
  // Compute M-norm of returned update step
  Scalar s_M_norm = sqrt(s.dot(large_M * s));
  EXPECT_LT(fabs((update_step_norm - s_M_norm) / s_M_norm), eps_rel);
}

/// Test STPCG on a large problem with an equality constraint, using constraint
/// preconditiong, and with an extremely tight (effectively exact) stopping
/// tolerance and a numerically-infinite trust-region radius, in order to ensure
/// that the underlying method is implemented properly
TEST_F(STPCGUnitTest, ExactProjectedSTPCG) {

  kappa = 1e-8;
  theta = .7;
  Delta = std::numeric_limits<Scalar>::max();

  // Sample a random constraint operator A
  Matrix A = 1000 * Matrix::Random(num_constraints, large_dim);

  /// Compute the solution of this (strictly convex) quadratic program by
  /// solving the KKT system directly:
  ///
  /// [H A'][s] = [-g]
  /// [A 0 ][l] = [0]

  // Somewhat hacky but simple: construct KKT system matrix as a dense matrix by
  // assigning to its blocks
  Matrix Kdense =
      Matrix::Zero(large_dim + num_constraints, large_dim + num_constraints);
  Kdense.topLeftCorner(large_dim, large_dim) = large_P;
  Kdense.topRightCorner(large_dim, num_constraints) = A.transpose();
  Kdense.bottomLeftCorner(num_constraints, large_dim) = A;

  // ... and then sparsify
  SparseMatrix K = Kdense.sparseView();
  K.makeCompressed();

  // Construct right-hand side vector for KKT system
  Vector rhs = Vector::Zero(large_dim + num_constraints);
  rhs.head(large_dim) = -large_g;

  // Solve KKT system using LU factorization
  Eigen::UmfPackLU<SparseMatrix> Kfact(K);
  Vector z = Kfact.solve(rhs);

  Vector s_gt = z.head(large_dim);
  Vector lambda_star = z.tail(num_constraints);

  // Verify solution by checking the KKT conditions
  EXPECT_LT((large_g + large_P * s_gt + A.transpose() * lambda_star).norm(),
            eps_abs);
  EXPECT_LT((A * s_gt).norm(), eps_abs);

  /// Construct constraint preconditioning operator Mc, satisfying:
  ///
  /// Mc(s) = (x, l), where (x,l) is the solution of:
  ///
  /// [M A'][x] = [s]
  /// [A 0 ][l] = [0]

  Matrix Mcdense(large_dim + num_constraints, large_dim + num_constraints);
  Mcdense.topLeftCorner(large_dim, large_dim) = large_M;
  Mcdense.topRightCorner(large_dim, num_constraints) = A.transpose();
  Mcdense.bottomLeftCorner(num_constraints, large_dim) = A;

  SparseMatrix Mc = Mcdense.sparseView();
  Mc.makeCompressed();

  // Construct and cache factorization of Pc
  Eigen::UmfPackLU<SparseMatrix> Mcfact(Mc);
  Vector w = Vector::Zero(large_dim + num_constraints);

  // Construct preconditioning operator
  Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector> McOp =
      [&Mcfact, &w](const Vector &r) -> std::pair<Vector, Vector> {
    // Construct and solve constraint preconditioning system
    w.head(large_dim) = r;
    Vector z = Mcfact.solve(w);

    return std::make_pair(z.head(large_dim), z.tail(num_constraints));
  };

  std::experimental::optional<
      Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector>>
      precon_op(McOp);

  /// Construct transpose constraint operator
  Matrix At = A.transpose();
  Optimization::LinearAlgebra::LinearOperator<Vector, Vector> At_func =
      [&At](const Vector &x) -> Vector { return At * x; };

  std::experimental::optional<
      Optimization::LinearAlgebra::LinearOperator<Vector, Vector>>
      At_op(At_func);

  /// Solve linearly-constrained trust-region subproblem *exactly* (i.e.,
  /// assuming an infinite trust-region radius and high-precision tolerance
  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      large_g, large_Pop, inner_product, update_step_norm, num_iterations,
      Delta, 5 * large_dim, kappa, theta, precon_op, At_op);

  // Verify that the computed update step lies in the null space of the
  // constraint operator A
  EXPECT_LT((A * s).norm(), eps_abs);

  // Verify that s agrees with the primal component of the primal-dual solution
  // z of the KKT system
  EXPECT_LT((s - s_gt).norm() / s_gt.norm(), eps_rel);

  // Verify that the M-norm of s is computed correctly
  Scalar s_M_norm = sqrt(s.dot(large_M * s));
  EXPECT_LT(fabs(s_M_norm - update_step_norm) / s_M_norm, eps_rel);
}

///// Test truncated STPCG on a large problem with an equality constraint, using
///// constraint preconditioning constraint
TEST_F(STPCGUnitTest, TruncatedPreconditionedProjectedSTPCG) {

  kappa = .1;
  theta = .7;
  Delta = std::numeric_limits<Scalar>::max();

  // Sample a random constraint operator A
  Matrix A = 1000 * Matrix::Random(num_constraints, large_dim);

  /// Construct constraint preconditioning operator Mc, satisfying:
  ///
  /// Mc(s) = (x, l), where (x,l) is the solution of:
  ///
  /// [M A'][x] = [s]
  /// [A 0 ][l] = [0]

  Matrix Mcdense(large_dim + num_constraints, large_dim + num_constraints);
  Mcdense.topLeftCorner(large_dim, large_dim) = large_M;
  Mcdense.topRightCorner(large_dim, num_constraints) = A.transpose();
  Mcdense.bottomLeftCorner(num_constraints, large_dim) = A;

  SparseMatrix Mc = Mcdense.sparseView();
  Mc.makeCompressed();

  // Construct and cache factorization of Pc
  Eigen::UmfPackLU<SparseMatrix> Mcfact(Mc);
  Vector w = Vector::Zero(large_dim + num_constraints);

  // Construct preconditioning operator
  Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector> McOp =
      [&Mcfact, &w](const Vector &r) -> std::pair<Vector, Vector> {
    // Construct and solve constraint preconditioning system
    w.head(large_dim) = r;
    Vector z = Mcfact.solve(w);

    return std::make_pair(z.head(large_dim), z.tail(num_constraints));
  };

  std::experimental::optional<
      Optimization::LinearAlgebra::STPCGPreconditioner<Vector, Vector>>
      precon_op(McOp);

  /// Construct transpose constraint operator
  Matrix At = A.transpose();
  Optimization::LinearAlgebra::LinearOperator<Vector, Vector> At_func =
      [&At](const Vector &x) -> Vector { return At * x; };

  std::experimental::optional<
      Optimization::LinearAlgebra::LinearOperator<Vector, Vector>>
      At_op(At_func);

  /// Approximately solve linearly-constrained trust-region subproblem
  Vector s = Optimization::LinearAlgebra::STPCG<Vector, Vector>(
      large_g, large_Pop, inner_product, update_step_norm, num_iterations,
      Delta, 5 * large_dim, kappa, theta, precon_op, At_op);

  /// Verify that the target fractional reduction in the P-norm of the residual
  /// is satisfied by the returned update
  std::pair<Vector, Vector> Pr0 = McOp(large_g);
  Scalar r0_Pnorm = sqrt(large_g.dot(Pr0.first));

  Vector rk = large_g + large_P * s;
  std::pair<Vector, Vector> Prk = McOp(rk);
  Scalar rk_Pnorm = sqrt(rk.dot(Prk.first));
  EXPECT_LT(rk_Pnorm / r0_Pnorm, kappa);

  /// Verify that the M-norm of s is computed correctly
  Scalar s_M_norm = sqrt(s.dot(large_M * s));

  EXPECT_LT(fabs(s_M_norm - update_step_norm) / s_M_norm, eps_rel);
}
