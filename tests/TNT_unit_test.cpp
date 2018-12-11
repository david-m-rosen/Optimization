/** This unit test exercises the functionality of the Riemannian
 * truncated-Newton trust-region method */

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "Optimization/Smooth/TNT.h"

// Typedef for the numerical type will use in the following tests
typedef double Scalar;

// Typedefs for Eigen matrix types
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Variable;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Tangent;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> DiagonalMatrix;

// Threshold for considering two values to be numerically equal in the following
// tests
Scalar TNT_test_error_threshold = 1e-6;

// Dimensions of small and large tests problems
constexpr size_t small_dim = 3;
constexpr size_t large_dim = 1000;

/// Test fixture for testing the Steihaug-Toint preconditioned conjugate
/// gradient solver used to (approximately) solve the trust-region subproblem at
/// each iteration of the truncated-Newton trust-region optimization method
class STPCGUnitTest : public testing::Test {
protected:
  /// Set up variables and functions used in the following tests

  // Standard Euclidean inner product for vectors
  Optimization::Smooth::RiemannianMetric<Variable, Tangent, Scalar> metric =
      [](const Variable &X, const Tangent &V1, const Tangent &V2) {
        return V1.dot(V2);
      };

  // Variable representing the current iterate at which we are evaluating the
  // Hessian operator -- this is not actually used in most of the following
  // tests
  Variable X;

  // Input vector  g -- right-hand side of the equation Hv = -g we're trying to
  // solve in each test instance
  Eigen::Matrix<Scalar, small_dim, 1> small_g;
  Eigen::Matrix<Scalar, large_dim, 1> large_g;

  // Positive-definite Hessian matrix
  Eigen::DiagonalMatrix<Scalar, small_dim> small_P;
  Eigen::DiagonalMatrix<Scalar, large_dim> large_P;

  // Negative-definite Hessian matrix
  Eigen::DiagonalMatrix<Scalar, small_dim> small_N;

  // Preconditioning matrix M
  Eigen::DiagonalMatrix<Scalar, small_dim> small_M;
  Eigen::DiagonalMatrix<Scalar, large_dim> large_M;

  // Hessian operators corresponding to P and N
  Optimization::Smooth::LinearOperator<Variable, Tangent> small_Pop;
  Optimization::Smooth::LinearOperator<Variable, Tangent> large_Pop;

  Optimization::Smooth::LinearOperator<Variable, Tangent> small_Nop;

  // Preconditioning operator: multiplies v by M^-1
  Optimization::Smooth::LinearOperator<Variable, Tangent> small_MinvOp;
  Optimization::Smooth::LinearOperator<Variable, Tangent> large_MinvOp;

  // Null-op (identity) preconditioner
  std::experimental::optional<
      Optimization::Smooth::LinearOperator<Variable, Tangent>>
      Id_precon;

  /// Arguments to Steihaug-Toint truncated preconditioned conjugate gradient
  /// method

  // Trust region radius
  Scalar Delta;

  // Truncation stopping criteria
  Scalar kappa_fgr;
  Scalar theta;

  // Additional returns from STPCG
  Scalar update_step_norm;
  size_t num_iterations;

  virtual void SetUp() {

    /// Positive-definite Hessian matrices

    // Positive-definite matrix P
    // Manually-constructed small positive-definite matrix
    small_P.diagonal() << 1000, 100, 1;
    small_Pop = [&](const Variable &X, const Tangent &V) {
      return small_P * V;
    };

    // Sample a large PD Hessian (note that here we are using the fact that
    // Eigen's Random operator returns values in the range [-1, 1]
    large_P =
        (2000 * Tangent::Ones(large_dim) + 1000 * Tangent::Random(large_dim))
            .asDiagonal();

    large_Pop = [&](const Variable &X, const Tangent &V) {
      return large_P * V;
    };

    /// Negative-definite Hessian matrices

    // Manually-constructed small negative-definite matrix N
    small_N = -1.0 * small_P;
    small_Nop = [&](const Variable &X, const Tangent &V) {
      return small_N * V;
    };

    /// Positive-definite preconditioning operators

    // Manually-constructed small positive-definite preconditioner
    small_M.diagonal() << 100, 10, 1;
    small_MinvOp = [&](const Variable &X, const Tangent &V) -> Tangent {
      return (small_M.inverse() * V);
    };

    // Randomly-sampled large positive-definite preconditioner
    large_M =
        (2000 * Tangent::Ones(large_dim) + 1000 * Tangent::Random(large_dim))
            .asDiagonal();
    large_MinvOp = [&](const Variable &X, const Tangent &V) -> Tangent {
      return (large_M.inverse() * V);
    };

    // Right-hand side g
    small_g << 21, -.4, 19;
    large_g.setRandom();

    /// Termination criteria for the Steihaug-Toint truncated conjugate gradient
    /// solver

    // Trust-region radius
    Delta = 100;
    kappa_fgr = .1;
    theta = 1;
  }
};

/// Tests for the Steihaug-Toint truncated preconditioned conjugate gradient
/// method

TEST_F(STPCGUnitTest, ExactSTPCG) {

  /// First, run the conjugate gradient method with a small PD matrix
  /// and an *exact* stopping criterion and an infinite trust region radius, to
  /// ensure that the underlying method is implemented properly

  Tangent V = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, small_g, small_Pop, metric, Id_precon, update_step_norm,
      num_iterations, std::numeric_limits<Scalar>::max(), small_dim, 0);

  Scalar error = (V + small_P.inverse() * small_g).norm();

  // Check the solution
  EXPECT_NEAR(error, 0, TNT_test_error_threshold);

  // Check that the reported update step norm is correct
  EXPECT_NEAR(update_step_norm, V.norm(), TNT_test_error_threshold);
}

TEST_F(STPCGUnitTest, STPCGwithNegativeCurvature) {

  /// Test STPCG with small negative-definite diagonal Hessian to ensure that
  /// the step to the boundary of the trust-region is computed properly

  Tangent V = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, small_g, small_Nop, metric, Id_precon, update_step_norm,
      num_iterations, Delta, small_dim, 0);

  // Since N is negative-definite, the negative-curvature-handling subroutine of
  // the STPCG method should compute a solution that moves in the direction of
  // -g all the way to the boundary of the trust-region radius (i.e., until its
  // norm is Delta)
  Tangent target_solution = -(Delta / small_g.norm()) * small_g;
  Scalar error = (target_solution - V).norm();

  EXPECT_NEAR(error, 0, TNT_test_error_threshold);
  EXPECT_NEAR(update_step_norm, target_solution.norm(),
              TNT_test_error_threshold);
}

TEST_F(STPCGUnitTest, STPCGwithPreconditioning) {

  /// Now test using a positive-definite Hessian and a (nontrivial)
  /// positive-definite preconditioning operator

  Tangent v = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, small_g, small_Pop, metric, small_MinvOp, update_step_norm,
      num_iterations, std::numeric_limits<Scalar>::max(), small_dim, 0);

  // Error in exact solution of equation Hv = -g
  Scalar error = (small_P * v + small_g).norm();

  // Error in norm estimate under the M-norm: sqrt(v' * M * v)
  Scalar v_M_norm = std::sqrt(v.dot(small_M * v));

  EXPECT_NEAR(error, 0, TNT_test_error_threshold);
  EXPECT_NEAR(update_step_norm, v_M_norm, TNT_test_error_threshold);
}

TEST_F(STPCGUnitTest, STPCGwithNegativeCurvatureAndPreconditioning) {

  /// Test with a negative-definite Hessian and positive-definite
  /// preconditioning operator

  Tangent V = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, small_g, small_Nop, metric, small_MinvOp, update_step_norm,
      num_iterations, Delta, small_dim, 0);

  // Compute M-norm of initial search direction p = -M^{-1} * g:
  Tangent p = -(small_M.inverse() * small_g);
  Scalar p_M_norm = std::sqrt(p.dot(small_M * p));
  Tangent p_scaled = (Delta / p_M_norm) * p;

  Scalar error = (V - p_scaled).norm();
  Scalar norm_error = fabs(update_step_norm - std::sqrt(V.dot(small_M * V)));

  EXPECT_NEAR(error, 0, TNT_test_error_threshold);
  EXPECT_NEAR(norm_error, 0, TNT_test_error_threshold);
}

TEST_F(STPCGUnitTest, SmallSTPCGwithTruncation) {

  /// Test STPCG with a small positive-definite Hessian and truncation

  Tangent V_truncated = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, small_g, small_Pop, metric, Id_precon, update_step_norm,
      num_iterations, Delta, small_dim, kappa_fgr, theta);

  // Compute the norm of the predicted gradient g_pred = H * v + g at the
  // truncated solution
  Scalar truncated_grad_norm = (small_P * V_truncated + small_g).norm();

  // Get the relative error |gpred| / |g| of the predicted gradient
  Scalar truncate_grad_norm_relative_error =
      truncated_grad_norm / small_g.norm();

  // Compute the error the value returned for the norm of the update step
  Scalar V_truncated_norm_error = fabs(update_step_norm - V_truncated.norm());

  EXPECT_LE(truncate_grad_norm_relative_error, kappa_fgr);
  EXPECT_NEAR(V_truncated_norm_error, 0, TNT_test_error_threshold);
}

TEST_F(STPCGUnitTest, LargeSTPCGwithTruncation) {

  /// Test STPCG with a large positive-definite Hessian and truncation

  Tangent V_truncated = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, large_g, large_Pop, metric, Id_precon, update_step_norm,
      num_iterations, Delta, large_dim, kappa_fgr, theta);

  Scalar truncated_grad_norm = (large_P * V_truncated + large_g).norm();
  Scalar truncate_grad_norm_relative_error =
      truncated_grad_norm / large_g.norm();

  Scalar V_truncated_norm_error = fabs(update_step_norm - V_truncated.norm());

  EXPECT_LE(truncate_grad_norm_relative_error, kappa_fgr);
  EXPECT_NEAR(V_truncated_norm_error, 0, TNT_test_error_threshold);
}

TEST_F(STPCGUnitTest, SmallSTPCGwithPreconditioningAndTruncation) {

  /// Test preconditioned STPCG with a small positive-definite Hessian and
  /// truncation

  Tangent V_truncated = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, small_g, small_Pop, metric, small_MinvOp, update_step_norm,
      num_iterations, std::numeric_limits<Scalar>::max(), small_dim, kappa_fgr,
      theta);

  Scalar truncated_grad_norm = (small_P * V_truncated + small_g).norm();
  Scalar truncate_grad_norm_relative_error =
      truncated_grad_norm / small_g.norm();

  Scalar V_truncated_M_norm = std::sqrt(V_truncated.dot(small_M * V_truncated));
  Scalar V_truncated_norm_error = fabs(update_step_norm - V_truncated_M_norm);

  EXPECT_LE(truncate_grad_norm_relative_error, kappa_fgr);
  EXPECT_NEAR(V_truncated_norm_error, 0, TNT_test_error_threshold);
}

TEST_F(STPCGUnitTest, LargeSTPCGwithPreconditioningAndTruncation) {

  /// Test preconditioned STPCG with a large positive-definite Hessian and
  /// truncation

  Tangent V_truncated = Optimization::Smooth::STPCG<Variable, Tangent, Scalar>(
      X, large_g, large_Pop, metric, large_MinvOp, update_step_norm,
      num_iterations, std::numeric_limits<Scalar>::max(), large_dim, kappa_fgr,
      theta);

  Scalar truncated_grad_norm = (large_P * V_truncated + large_g).norm();
  Scalar truncate_grad_norm_relative_error =
      truncated_grad_norm / large_g.norm();

  Scalar V_truncated_M_norm = std::sqrt(V_truncated.dot(large_M * V_truncated));
  Scalar V_truncated_norm_error = fabs(update_step_norm - V_truncated_M_norm);

  EXPECT_LE(truncate_grad_norm_relative_error, kappa_fgr);
  EXPECT_NEAR(V_truncated_norm_error, 0, TNT_test_error_threshold);
}

/// Test cases for the end-to-end truncated-Newton trust-region algorithm

TEST(TNTUnitTest, EuclideanTNTRosenbrock) {
  /// Test minimization of the Rosenbrock function
  ///
  /// f(x,y) = (a - x)^2 + b(y - x^2)^2
  ///
  /// whose (global) minimizer is f(a,a^2) = 0,
  ///
  /// using the simplified Euclidean interface

  // Rosenbrock function parameters
  Scalar a = 1;
  Scalar b = 100;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix;

  // Objective: f(x,y) = (a - x)^2  + b(y - x^2)^2
  Optimization::Objective<Vector, Scalar> F = [a, b](const Vector &x) {
    return std::pow(a - x(0), 2) + b * std::pow((x(1) - std::pow(x(0), 2)), 2);
  };

  // Euclidean gradient operator: returns the Euclidean gradient nablaF(X) at
  // each X df/dx = -2(a-x) - 4bx(y-x^2) df / dy = 2b(y-x^2)
  Optimization::Smooth::EuclideanVectorField<Vector> nabla_F =
      [a, b](const Vector &x) {
        Vector df;
        df(0) = -2 * (a - x(0)) - 4 * b * x(0) * (x(1) - std::pow(x(0), 2));
        df(1) = 2 * b * (x(1) - std::pow(x(0), 2));
        return df;
      };

  // Euclidean Hessian constructor: Returns the Hessian operator H(X) at X
  // H(X) = [2 - 4by + 12bx^2   -4bx
  //      -4bx               2b]
  Optimization::Smooth::EuclideanLinearOperatorConstructor<Vector> HC =
      [a, b](const Vector &x) {
        // Compute Euclidean Hessian at X
        Matrix H;
        H(0, 0) = 2 - 4 * b * x(1) + 12 * b * std::pow(x(0), 2);
        H(0, 1) = -4 * b * x(0);
        H(1, 0) = H(0, 1);
        H(1, 1) = 2 * b;

        // Construct and return Euclidean Hessian operator: this is a function
        // that accepts as input a point X and tangent vector V, and returns
        // H(X)[V], the value of the Hessian operator at X applied to V
        Optimization::Smooth::EuclideanLinearOperator<Vector> Hess =
            [H](const Vector &x, const Vector &v) { return H * v; };

        return Hess;
      };

  /// Sample intial point

  Vector x0(.1, .1);
  Vector x_min(a, a * a);

  /// Run TNT optimizer

  // Set TNT options
  Optimization::Smooth::TNTParams<Scalar> tnt_params;
  tnt_params.stepsize_tolerance = 0; // Allow arbitrarily small stepsizes

  Optimization::Smooth::TNTResult<Vector, Scalar> result =
      Optimization::Smooth::EuclideanTNT<Vector, Scalar>(
          F, nabla_F, HC, x0, std::experimental::nullopt, tnt_params);

  // Check function value
  EXPECT_NEAR(result.f, 0, TNT_test_error_threshold);

  // Check gradient value
  EXPECT_NEAR(result.grad_f_x_norm, 0, TNT_test_error_threshold);

  // Check final solution
  EXPECT_NEAR((result.x - x_min).norm(), 0, TNT_test_error_threshold);
}

TEST(TNTUnitTest, RiemannianTNTSphere) {
  ///  We will minimize the function f(X; P) := | X - P |^2, where P in S^2 is
  ///  a fixed point on the sphere
  ///
  /// In the example that follows, we will treat the point P as a fixed (i.e.
  /// unoptimized) parameter of the objective f, and pass this to the TNT
  /// optimization library as a user-supplied optional argument.

  typedef Eigen::Matrix<Scalar, 3, 1> Vector;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix;

  Vector P = {0.0, 0.0, 1.0}; // P is the north pole

  /// Set up function handles

  /// Utility function: projection of the vector V \in R^3 onto the tangent
  /// space T_X(S^2) of S^2 at the point
  auto project = [](const Vector &X, const Vector &V) -> Vector {
    return V - X.dot(V) * X;
  };

  /// Objective
  Optimization::Objective<Vector, Scalar, Vector> F =
      [](const Vector &X, const Vector &P) { return (X - P).squaredNorm(); };

  /// Gradient
  Optimization::Smooth::VectorField<Vector, Vector, Vector> grad_F =
      [&project](const Vector &X, const Vector &P) {
        // Euclidean gradient
        Vector nabla_f = 2 * (X - P);

        // Compute Riemannian gradient from Euclidean one
        return project(X, nabla_f);
      };

  /// Riemannian Hessian constructor: Returns the Riemannian Hessian operator
  /// H(X): T_X(S^2) -> T_X(S^2) at X
  Optimization::Smooth::LinearOperatorConstructor<Vector, Vector, Vector> HC =
      [&project, &grad_F](const Vector &X, Vector &P) {
        // Euclidean Hessian matrix
        Matrix EucHess = 2 * Matrix::Identity();

        // Return Riemannian Hessian-vector product operator using the
        // Euclidean Hessian
        Optimization::Smooth::LinearOperator<Vector, Vector, Vector> Hessian =
            [&project, EucHess, &grad_F](const Vector &X, const Vector &Xdot,
                                         Vector &P) -> Vector {
          return project(X, EucHess * Xdot) - X.dot(grad_F(X, P)) * Xdot;
        };

        return Hessian;
      };

  /// Riemannian metric on S^2: this is just the usual inner-product on R^3
  Optimization::Smooth::RiemannianMetric<Vector, Vector, Scalar, Vector>
      metric = [](const Vector &X, const Vector &V1, const Vector &V2,
                  const Vector &P) { return V1.dot(V2); };

  /// Projection-based retraction operator for S^2
  Optimization::Smooth::Retraction<Vector, Vector, Vector> retract =
      [](const Vector &X, const Vector &V, const Vector &P) {
        return (X + V).normalized();
      };

  /// Set initial point

  // X0 will be a point on the equator
  Vector X0 = {-0.5, -0.5, -0.707107};

  /// Run TNT optimizer
  // Set TNT options
  Optimization::Smooth::TNTParams<Scalar> tnt_params;
  tnt_params.stepsize_tolerance = 0;

  Optimization::Smooth::TNTResult<Vector, Scalar> result =
      Optimization::Smooth::TNT<Vector, Vector, Scalar, Vector>(
          F, grad_F, HC, metric, retract, X0, P, std::experimental::nullopt,
          tnt_params);

  EXPECT_NEAR(result.f, 0, TNT_test_error_threshold);
  EXPECT_NEAR(result.grad_f_x_norm, 0, TNT_test_error_threshold);
  EXPECT_NEAR((result.x - P).norm(), 0, TNT_test_error_threshold);
}
