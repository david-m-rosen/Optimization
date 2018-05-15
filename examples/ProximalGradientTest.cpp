#include "Optimization/Convex/ProximalGradient.h"

#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace Optimization;
using namespace Optimization::Convex;
int main() {
  /// SET UP TEST FUNCTION HANDLES

  /**  Let's start by solving a group-sparse lasso problem of the form
   *
   * min || A *  x - b ||^2 + mu * || x ||_2
   *
   */

  typedef Eigen::VectorXd Variable;
  Eigen::Matrix2d A;
  A << 1000, 0, 0, 1.0;
  Variable b = Eigen::Vector2d(1, 1);
  double mu = 100;

  InnerProduct<Variable> inner_product =
      [&](const Variable &v1, const Variable &v2) { return v1.dot(v2); };

  // Define f and grad_f
  Objective<Variable> f = [&](const Variable &x) {
    return (A * x - b).squaredNorm();
  };

  GradientOperator<Variable> grad_f = [&](const Variable &x) {
    return 2 * A.transpose() * (A * x - b);
  };

  // Define g and prox_{lambda g}
  Objective<Variable> g = [&](const Variable &x) { return mu * x.norm(); };

  ProximalOperator<Variable> prox_g = [&](const Variable &x, double lambda) {
    /** The proximal operator for the weighted L2 norm is the block-soft
     * thresholding operator:
     *
     * (1 - mu*lambda / |x|)_{+} * x
     *
     * (cf. Section 6.5.4 of Parikh and Boyd's "Proximal Algorithms"
     */

    return max(1 - mu * lambda / sqrt(x.dot(x)), 0.0) * x;
  };

  /// INITIALIZATION
  Variable x0(2);
  x0(0) = 4;
  x0(1) = 4;

  cout << "Solving group LASSO problem: " << endl
       << endl
       << "min | Ax - b |^2 + mu * |x|" << endl
       << endl
       << "with A = " << endl
       << A << endl
       << endl
       << "b =  " << endl
       << b << endl
       << "and mu = " << mu << endl
       << endl;

  cout << "Initial value of x = " << endl << x0 << endl << endl;

  /// OPTIMIZE!

  /// Set optimization algorithm parameters
  ProximalGradientParams params;
  params.verbose = true;
  params.max_iterations = 1000000;
  // params.adaptive_restart = false;
  params.mode = ACCELERATED;
  params.composite_gradient_tolerance = 1e-4;

  cout << "Optimizing!" << endl << endl;
  ProximalGradientResult<Variable> result = ProximalGradient<Variable>(
      f, grad_f, g, prox_g, inner_product, x0, params);

  cout << "Final result:" << endl;
  cout << "F(x) = " << result.f << endl;
  cout << "x = " << endl << result.x << endl << endl;
}
