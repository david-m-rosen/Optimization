/** This simple example demonstrates how to run gradient descent on a quadratic
 * test function using the simplified (Euclidean) interface */

#include <Eigen/Dense>
#include <fstream>

#include "Optimization/Smooth/GradientDescent.h"

using namespace std;
using namespace Optimization;
using namespace Optimization::Smooth;

int main() {
  /// SET UP TEST FUNCTION HANDLES

  /**  Minimize a simple quadratic of the form
   *
   * min || A *  x - b ||^2
   *
   */

  typedef Eigen::Vector2d Vector;

  Eigen::Matrix2d A = Eigen::Matrix2d::Random();
  Eigen::Vector2d b = Eigen::Vector2d::Random();

  A << 250, .5, .5, 1;

  cout << "Testing minimization of the randomly-sampled quadratic |Ax - b|^2, "
          "with:"
       << endl
       << "A = " << endl
       << A << endl
       << "b = " << endl
       << b << endl
       << endl;

  // Define f
  Objective<Vector> f = [&](const Vector &x) {
    return (A * x - b).squaredNorm();
  };

  // Gradient of f
  EuclideanVectorField<Vector> grad_f = [&](const Vector &x) {
    return 2 * A.transpose() * (A * x - b);
  };

  /// Optimization setup
  GradientDescentParams params;
  params.max_iterations = 100000;
  params.log_iterates = true;
  params.verbose = true;

  /// Initialization

  Vector x0 = Vector::Random();

  GradientDescentResult<Vector> result =
      EuclideanGradientDescent(f, grad_f, x0, params);

  cout << "Final result: " << endl
       << "x = " << endl
       << result.x << endl
       << endl;
}
