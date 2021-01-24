/** This unit test exercises the functionality of the Riemannian
 * truncated-Newton least-squares method */

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "Optimization/Riemannian/TNLS.h"

class TNLSUnitTest : public testing::Test {
protected:
  // Typedef for the numerical type will use in the following tests
  typedef double Scalar;

  // Typedefs for Eigen matrix types
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

  /// Absolute tolerance for equality
  Scalar eps_abs = 1e-6;

  // Boolean variable controlling whether to print output -- useful for quickly
  // checking failed tests
  bool verbose = false;

  // We consider fitting a simple curve of the form
  //
  // f(x; beta) := sin(omega * x + phi)
  //
  // to data

  Scalar omega = M_PI_2;
  Scalar phi = M_PI_4;

  size_t m = 100; // Number of test points

  // Vector of inputs
  Vector x;

  // Vector of outputs
  Vector y;

  // Cache variable to store the Jacobian of F
  Matrix J;

  // Initialization for nonlinear fitting
  Vector beta0;

  /// Function handles

  // Vector valued residual function: computes and returns
  //
  // F(beta)_i := y - f(x; beta)
  Optimization::Riemannian::Mapping<Vector, Vector> F;

  // Jacobian constructor

  Optimization::Riemannian::JacobianPairFunction<Vector, Vector, Vector>
      JacFunc;

  virtual void SetUp() override {

    // Set of test points x
    x = Vector::LinSpaced(m, -M_PI, M_PI);

    // Evaluate *noiseless* outputs y = sin(omega*x + phi) + eps
    y = (omega * x + Vector::Constant(m, phi)).array().sin();

    // Allocate storage for Jacobian
    J.resize(m, 2);

    /// Construct vector function:
    ///
    /// F: R^2 -> R^m
    /// F(omega,phi) = y - sin(omega*x + phi)

    F = [this](const Vector &beta) -> Vector {
      return this->y -
             Vector((beta(0) * x + Vector::Constant(this->x.size(), beta(1)))
                        .array()
                        .sin());
    };

    JacFunc = [this](const Vector &beta)
        -> std::pair<
            Optimization::Riemannian::Jacobian<Vector, Vector, Vector>,
            Optimization::Riemannian::JacobianAdjoint<Vector, Vector, Vector>> {
      /// First, compute Jacobian matrix and store in cache variable J

      // Set column of J corresponding to phi (= beta(1))
      this->J.col(1) =
          -(beta(0) * this->x + Vector::Constant(this->x.size(), beta(1)))
               .array()
               .cos();

      // Set column of J corresponding to w (= beta(0))
      this->J.col(0) = this->J.col(1).cwiseProduct(this->x);

      /// Construct Jacobian operators

      // Jacobian operator
      Optimization::Riemannian::Jacobian<Vector, Vector, Vector> gradF =
          [this](const Vector &x, const Vector &v) -> Vector {
        return this->J * v;
      };

      // Jacobian adjoint operator
      Optimization::Riemannian::JacobianAdjoint<Vector, Vector, Vector> gradFt =
          [this](const Vector &x, const Vector &v) -> Vector {
        return this->J.transpose() * v;
      };

      /// Pass these back as a pair
      return std::make_pair(gradF, gradFt);
    };

    // Set initial point beta0
    beta0.resize(2);
    beta0 << 1.0, 1.0;
  }
};

/// Test finding the root of a nonlinear system of equations by fitting to
/// *noiseless* data y
TEST_F(TNLSUnitTest, RootFinding) {

  Optimization::Riemannian::TNLSParams<Scalar> params;
  params.relative_decrease_tolerance = 0;
  params.gradient_tolerance = 0;
  params.stepsize_tolerance = 0;
  params.Delta_tolerance = 0;
  params.root_tolerance = eps_abs;
  params.verbose = verbose;

  Optimization::Riemannian::TNLSResult<Vector> result =
      Optimization::Riemannian::EuclideanTNLS<Vector>(F, JacFunc, beta0,
                                                      params);

  /// Verify that the method reported success, and terminated on a solution of
  /// sufficient small residual norm
  EXPECT_EQ(result.status, Optimization::Riemannian::TNLSStatus::Root);

  /// Verify that the residual at the returned estimate is sufficiently small
  EXPECT_LT(F(result.x).norm(), eps_abs);
}

/// Now fit these parameters to *noisy* data
TEST_F(TNLSUnitTest, LeastSquaresParameterFitting) {

  // Sample a random noise vector
  Vector z = .1 * Vector::Random(m);

  // Add this the outputs y
  y += z;

  Optimization::Riemannian::TNLSParams<Scalar> params;
  params.relative_decrease_tolerance = 0;
  params.gradient_tolerance = eps_abs;
  params.stepsize_tolerance = 0;
  params.Delta_tolerance = 1e-10;
  params.verbose = verbose;

  Optimization::Riemannian::TNLSResult<Vector> result =
      Optimization::Riemannian::EuclideanTNLS<Vector>(F, JacFunc, beta0,
                                                      params);

  // Extract solution
  const Vector &beta = result.x;

  // Evaluate residual at solution
  Vector Fbeta = F(beta);

  // Evaluate Jacobian at solution
  Optimization::Riemannian::Jacobian<Vector, Vector, Vector> gradF;
  Optimization::Riemannian::JacobianAdjoint<Vector, Vector, Vector> gradFt;
  std::tie(gradF, gradFt) = JacFunc(beta);

  Vector gradLbeta = 2 * gradFt(beta, Fbeta);

  /// Verify that the method reported success, and terminated on a solution
  /// of sufficient small residual norm
  EXPECT_EQ(result.status, Optimization::Riemannian::TNLSStatus::Gradient);

  /// Verify that the gradient of the least-squares objective at the returned
  /// estimate is sufficiently small
  EXPECT_LT(gradLbeta.norm(), eps_abs);

  /// Verify that the residual at the returned solution is *strictly smaller*
  /// than the norm of the noise that was added (i.e., the residual at the
  /// planted signal)
  EXPECT_LT(Fbeta.norm(), z.norm());
}
