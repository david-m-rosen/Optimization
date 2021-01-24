/** This unit test exercises the functionality of the Riemannian
 * truncated-Newton trust-region method */

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "Optimization/Riemannian/TNT.h"

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

/// Test cases for the truncated-Newton trust-region algorithm

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
  Optimization::Riemannian::EuclideanVectorField<Vector> nabla_F =
      [a, b](const Vector &x) {
        Vector df;
        df(0) = -2 * (a - x(0)) - 4 * b * x(0) * (x(1) - std::pow(x(0), 2));
        df(1) = 2 * b * (x(1) - std::pow(x(0), 2));
        return df;
      };

  // Euclidean Hessian constructor: Returns the Hessian operator H(X) at X
  // H(X) = [2 - 4by + 12bx^2   -4bx
  //      -4bx               2b]
  Optimization::Riemannian::EuclideanLinearOperatorConstructor<Vector> HC =
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
        Optimization::Riemannian::EuclideanLinearOperator<Vector> Hess =
            [H](const Vector &x, const Vector &v) { return H * v; };

        return Hess;
      };

  /// Sample intial point

  Vector x0(.1, .1);
  Vector x_min(a, a * a);

  /// Run TNT optimizer

  // Set TNT options
  Optimization::Riemannian::TNTParams<Scalar> tnt_params;
  tnt_params.stepsize_tolerance = 0; // Allow arbitrarily small stepsizes

  Optimization::Riemannian::TNTResult<Vector, Scalar> result =
      Optimization::Riemannian::EuclideanTNT<Vector, Scalar>(
          F, nabla_F, HC, x0, std::experimental::nullopt, tnt_params);

  // Check function value
  EXPECT_NEAR(result.f, 0, TNT_test_error_threshold);

  // Check gradient value
  EXPECT_NEAR(result.gradfx_norm, 0, TNT_test_error_threshold);

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
  Optimization::Riemannian::VectorField<Vector, Vector, Vector> grad_F =
      [&project](const Vector &X, const Vector &P) {
        // Euclidean gradient
        Vector nabla_f = 2 * (X - P);

        // Compute Riemannian gradient from Euclidean one
        return project(X, nabla_f);
      };

  /// Riemannian Hessian constructor: Returns the Riemannian Hessian operator
  /// H(X): T_X(S^2) -> T_X(S^2) at X
  Optimization::Riemannian::LinearOperatorConstructor<Vector, Vector, Vector>
      HC = [&project, &grad_F](const Vector &X, Vector &P) {
        // Euclidean Hessian matrix
        Matrix EucHess = 2 * Matrix::Identity();

        // Return Riemannian Hessian-vector product operator using the
        // Euclidean Hessian
        Optimization::Riemannian::LinearOperator<Vector, Vector, Vector>
            Hessian = [&project, EucHess, &grad_F](const Vector &X,
                                                   const Vector &Xdot,
                                                   Vector &P) -> Vector {
          return project(X, EucHess * Xdot) - X.dot(grad_F(X, P)) * Xdot;
        };

        return Hessian;
      };

  /// Riemannian metric on S^2: this is just the usual inner-product on R^3
  Optimization::Riemannian::RiemannianMetric<Vector, Vector, Scalar, Vector>
      metric = [](const Vector &X, const Vector &V1, const Vector &V2,
                  const Vector &P) { return V1.dot(V2); };

  /// Projection-based retraction operator for S^2
  Optimization::Riemannian::Retraction<Vector, Vector, Vector> retract =
      [](const Vector &X, const Vector &V, const Vector &P) {
        return (X + V).normalized();
      };

  /// Set initial point

  // X0 will be a point on the equator
  Vector X0 = {-0.5, -0.5, -0.707107};

  /// Run TNT optimizer
  // Set TNT options
  Optimization::Riemannian::TNTParams<Scalar> tnt_params;
  tnt_params.stepsize_tolerance = 0;

  Optimization::Riemannian::TNTResult<Vector, Scalar> result =
      Optimization::Riemannian::TNT<Vector, Vector, Scalar, Vector>(
          F, grad_F, HC, metric, retract, X0, P, std::experimental::nullopt,
          tnt_params);

  EXPECT_NEAR(result.f, 0, TNT_test_error_threshold);
  EXPECT_NEAR(result.gradfx_norm, 0, TNT_test_error_threshold);
  EXPECT_NEAR((result.x - P).norm(), 0, TNT_test_error_threshold);
}
