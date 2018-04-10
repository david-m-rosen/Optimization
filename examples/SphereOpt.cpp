/** This simple example demonstrates how to use the TNT truncated-Newton
 * trust-region library to perform optimization over the sphere S^2 in R^3.*/

#include "Optimization/Smooth/TNT.h"

#include <Eigen/Dense>

#include <random>
#include <time.h>

typedef Eigen::Vector3d Vector;

using namespace std;
int main() {

  /// SET UP OPTIMIZATION

  ///  We will minimize the function f(X; P) := || X - P ||^2, where P in S^2 is
  ///  a fixed point on the sphere
  ///
  /// In the example that follows, we will treat the point P as a fixed (i.e.
  /// unoptimized) paramter of the objective f, and pass this to the TNT
  /// optimization library as a user-supplied optional argument.

  Vector P = {0.0, 0.0, 1.0}; // p is the north pole

  /// SET UP FUNCTION HANDLES

  /// Utility function: projection of the vector V \in R^3 onto the tangent
  /// space T_X(S^2) of S^2 at the point
  auto project = [](const Vector &X, const Vector &V) -> Vector {
    return V - X.dot(V) * X;
  };

  /// Objective
  Optimization::Objective<Vector, Vector> F =
      [](const Vector &X, const Vector &Y) { return (X - Y).squaredNorm(); };

  /// Local quadratic model function:  this constructs the (Riemannian) gradient
  /// and model Hessian operator for f at x.
  Optimization::Smooth::QuadraticModel<Vector, Vector, Vector> QM = [&project](
      const Vector &X, Vector &grad,
      Optimization::Smooth::LinearOperator<Vector, Vector, Vector> &Hessian,
      const Vector &P) {

    // Euclidean gradient
    Eigen::Vector3d nabla_f = 2 * (X - P);

    // Euclidean Hessian matrix
    Eigen::Matrix3d H = 2 * Eigen::Matrix3d::Identity();

    // Compute Riemannian gradient from Euclidean one
    grad = project(X, nabla_f);

    // Return Riemannian Hessian-vector product operator using the Euclidean
    // Hessian
    Hessian = [&project, H, nabla_f](const Vector &X, const Vector &Xdot,
                                     const Vector &P) {
      return project(X, H * Xdot) - X.dot(nabla_f) * Xdot;
    };
  };

  /// Riemannian metric on S^2: this is just the usual inner-product on R^3
  Optimization::Smooth::RiemannianMetric<Vector, Vector, Vector> metric =
      [](const Vector &X, const Vector &V1, const Vector &V2, const Vector &P) {
        return V1.dot(V2);
      };

  /// Projection-based retraction operator for S^2
  Optimization::Smooth::Retraction<Vector, Vector, Vector> retract =
      [](const Vector &X, const Vector &V, const Vector &P) {
        return (X + V).normalized();
      };

  /// SAMPLE INITIAL POINT

  cout << "Target point: P = " << endl << P << endl << endl;

  cout << "Sampling an initial point x0 on the sphere S^2 by perturbing P ... "
       << endl;

  // Construct a random initial point by slightly "fuzzing" a
  double epsilon = .75;
  // Unit tangent vector in T_P(S^2)
  Vector V = project(P, Vector::Random()).normalized();
  // Move away from x0 along v
  Vector X0 = retract(P, epsilon * V, P);

  cout << "X0 = " << endl << X0 << endl << endl;

  cout << "Initial distance between X0 and P: " << (X0 - P).norm() << endl
       << endl;

  /// RUN TNT OPTIMIZER!

  cout << "Running TNT optimizer!" << endl << endl;

  // Set TNT options
  Optimization::Smooth::TNTParams params;
  params.verbose = true;
  // params.Delta0 = .75;

  Optimization::Smooth::TNTResult<Vector> result =
      Optimization::Smooth::TNT<Vector, Vector, Vector>(
          F, QM, metric, retract, X0, P, std::experimental::nullopt, params);

  cout << "x_final = " << endl << result.x << endl << endl;

  cout << "Distance between p and x_final: " << (P - result.x).norm() << endl;
}
