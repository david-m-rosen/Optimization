#include <Eigen/Dense>

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

  typedef Eigen::Vector2d Variable;
  typedef Eigen::Vector2d Tangent;

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
  Objective<Variable> f = [&](const Variable &x) {
    return (A * x - b).squaredNorm();
  };

  // Gradient of f
  VectorField<Variable, Tangent> grad_f = [&](const Variable &x) {
    return 2 * A.transpose() * (A * x - b);
  };

  // Riemannian metric
  RiemannianMetric<Variable, Tangent> metric =
      [&](const Variable &x, const Tangent &v1, const Tangent &v2) {
        return v1.dot(v2);
      };

  // Retraction operator
  Retraction<Variable, Tangent> retract =
      [&](const Tangent &x, const Tangent &v) { return x + v; };

  /// Optimization setup
  GradientDescentParams params;
  params.max_iterations = 100000;
  params.verbose = true;

  /// Initialization

  Variable x0 = Variable::Random();

  GradientDescentResult<Variable> result =
      GradientDescent(f, grad_f, metric, retract, x0, params);

  cout << "Final result: " << endl
       << "x = " << endl
       << result.x << endl
       << endl;
}
