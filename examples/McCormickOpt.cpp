/** This simple example demonstrates how to use the simplified Euclidean
 * truncated-Newton trust-region interface to minimize the McCormick function:
 *
 * f(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
 *
 * whose minimizer is
 *
 * f(-.54719, -1.54719) = -1.9133
 */

#include "Optimization/Smooth/TNT.h"

#include <Eigen/Dense>

#include <random>
#include <time.h>

typedef Eigen::Vector2d Vector;

using namespace std;
using namespace Optimization;
using namespace Smooth;

int main() {

  /// SET UP OPTIMIZATION PROBLEM

  // Objective
  Objective<Vector> F = [](const Vector &x) {
    return sin(x(0) + x(1)) + (x(0) - x(1)) * (x(0) - x(1)) - 1.5 * x(0) +
           2.5 * x(1) + 1;
  };

  // Convenient utility function: Compute and return the Hessian matrix
  auto compute_Hessian = [](const Vector &x) -> Eigen::Matrix2d {
    Eigen::Matrix2d H;

    H(0, 0) = -sin(x(0) + x(1)) + 2;
    H(0, 1) = -sin(x(0) + x(1)) - 2;
    H(1, 0) = H(0, 1);
    H(1, 1) = -sin(x(0) + x(1)) + 2;

    return H;
  };

  // Local quadratic model
  EuclideanQuadraticModel<Vector> QM = [&compute_Hessian](
      const Vector &x, Vector &grad, EuclideanLinearOperator<Vector> &HessOp) {

    // Compute gradient
    grad(0) = cos(x(0) + x(1)) + 2 * (x(0) - x(1)) - 1.5;
    grad(1) = cos(x(0) + x(1)) - 2 * (x(0) - x(1)) + 2.5;

    HessOp = [&compute_Hessian](const Vector &x, const Vector &v) -> Vector {
      return compute_Hessian(x) * v;
    };
  };

  /// SAMPLE INITIAL POINT

  Vector x0(.1, .1);
  Vector x_opt(-0.54719, -1.54719);

  cout << "Initial point: x0 = " << endl << x0 << endl << endl;

  /// RUN TNT OPTIMIZER!

  // Test Hessian

  cout << "Running TNT optimizer!" << endl << endl;

  // Set TNT options
  Optimization::Smooth::TNTParams params;
  params.verbose = true;

  TNTResult<Vector> result =
      EuclideanTNT<Vector>(F, QM, x0, std::experimental::nullopt, params);

  cout << "Final objective value: = " << result.f << endl;
  cout << "True global minimum: -1.9133" << endl;
  cout << "Error in final objective: " << result.f + 1.9133 << endl << endl;

  cout << "Estimated minimizer: " << endl << result.x << endl << endl;
  cout << "True global minimizer: " << endl << x_opt << endl << endl;
  cout << "Error in minimizer estimate: " << (result.x - x_opt).norm() << endl
       << endl;
}
