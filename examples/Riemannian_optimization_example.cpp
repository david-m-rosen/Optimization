/** This simple example demonstrates how to use Riemannian gradient descent and
 * truncated-Newton trust-region (TNT) algorithms to solve a simple optimization
 * problem over the sphere S^2 in R^3. */

#include <Eigen/Dense>

#include <fstream>
#include <time.h>

#include "Optimization/Smooth/GradientDescent.h"
#include "Optimization/Smooth/TNT.h"

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

  /// Gradient
  Optimization::Smooth::VectorField<Vector, Vector, Vector> grad_F =
      [&project](const Vector &X, const Vector &P) {

        // Euclidean gradient
        Eigen::Vector3d nabla_f = 2 * (X - P);

        // Compute Riemannian gradient from Euclidean one
        return project(X, nabla_f);
      };

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

    // Return Riemannian Hessian-vector product operator using the
    // Euclidean Hessian
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

  /// SET INITIAL POINT

  Vector X0 = {sqrt(1.0 / 2), 0.0, -sqrt(1.0 / 2)};

  cout << "Target point: P = " << endl << P << endl << endl;
  cout << "X0 = " << endl << X0 << endl << endl;

  cout << "Initial distance between X0 and P: " << (X0 - P).norm() << endl
       << endl;

  /// RUN GRADIENT DESCENT OPTIMIZER!

  cout << "RUNNING GRADIENT DESCENT OPTIMIZER!" << endl << endl;

  // Set gradient descent options
  Optimization::Smooth::GradientDescentParams gd_params;
  gd_params.max_iterations = 1000;
  gd_params.verbose = true;
  gd_params.log_iterates = true;

  Optimization::Smooth::GradientDescentResult<Vector> gd_result =
      Optimization::Smooth::GradientDescent<Vector, Vector, Vector>(
          F, grad_F, metric, retract, X0, P, gd_params);

  cout << "Gradient descent final estimate for x =  " << endl
       << gd_result.x << endl
       << endl;

  cout << "Distance between p and x_final: " << (P - gd_result.x).norm() << endl
       << endl;

  /// RUN TNT OPTIMIZER!

  cout << "RUNNING TNT OPTIMIZER!" << endl << endl;

  // Set TNT options
  Optimization::Smooth::TNTParams tnt_params;
  tnt_params.verbose = true;
  tnt_params.log_iterates = true;

  Optimization::Smooth::TNTResult<Vector> tnt_result =
      Optimization::Smooth::TNT<Vector, Vector, Vector>(
          F, QM, metric, retract, X0, P, std::experimental::nullopt,
          tnt_params);

  cout << "x_final = " << endl << tnt_result.x << endl << endl;

  cout << "Distance between p and x_final: " << (P - tnt_result.x).norm()
       << endl
       << endl;

  /// SAVE STATE TRACES TO DISK

  cout << "Saving state traces" << endl;

  std::string gd_function_values_filename = "gd_function_values.txt";
  std::string gd_iterates_filename = "gd_iterates.txt";

  cout << "Saving sequence of function values to file: "
       << gd_function_values_filename << endl;
  ofstream gd_function_values_file(gd_function_values_filename);
  for (auto v : gd_result.objective_values)
    gd_function_values_file << v << " ";
  gd_function_values_file.close();

  cout << "Saving sequence of iterates to file: " << gd_iterates_filename
       << endl;
  ofstream gd_iterates_file(gd_iterates_filename);
  for (auto x : gd_result.iterates)
    gd_iterates_file << x.transpose() << endl;
  gd_iterates_file.close();
  std::string tnt_function_values_filename = "tnt_function_values.txt";
  std::string tnt_iterates_filename = "tnt_iterates.txt";

  cout << "Saving sequence of function values to file: "
       << tnt_function_values_filename << endl;
  ofstream tnt_function_values_file(tnt_function_values_filename);
  for (auto v : tnt_result.objective_values)
    tnt_function_values_file << v << " ";
  tnt_function_values_file.close();

  cout << "Saving sequence of iterates to file: " << tnt_iterates_filename
       << endl;
  ofstream tnt_iterates_file(tnt_iterates_filename);
  for (auto x : tnt_result.iterates)
    tnt_iterates_file << x.transpose() << endl;
  tnt_iterates_file.close();
}
