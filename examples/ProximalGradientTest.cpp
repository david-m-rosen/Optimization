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
  // A = Eigen::Matrix2d::Identity();
  Variable b = Eigen::Vector2d(1, 1);
  double mu = 100;

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

  // We will use this user-supplied function to record the sequence of iterates
  // visited by the method
  std::vector<Variable> iterates;
  ProximalGradientUserFunction<Variable> record_iterates =
      [&](double t, const Variable &x, double F, double r, unsigned int ls,
          const Variable &dx, double dF) { iterates.push_back(x); };

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
  params.max_iterations = 10000;
  // params.adaptive_restart = false;
  params.mode = ACCELERATED;

  cout << "Optimizing!" << endl << endl;
  ProximalGradientResult<Variable> result = ProximalGradient<Variable>(
      f, grad_f, g, prox_g, x0, params, record_iterates);

  cout << "Final result:" << endl;
  cout << "F(x) = " << result.f << endl;
  cout << "x = " << endl << result.x << endl << endl;

  // Now write output to disk
  string function_values_filename = "function_values.txt";
  cout << "Saving function values to file: " << function_values_filename
       << endl;
  ofstream function_values_file(function_values_filename);
  for (auto v : result.objective_values)
    function_values_file << v << " ";
  function_values_file << endl;
  function_values_file.close();

  string iterates_filename = "iterates.txt";
  cout << "Saving iterates to file: " << iterates_filename << endl;
  ofstream iterates_file(iterates_filename);
  for (auto v : iterates)
    iterates_file << v.transpose() << endl;
  iterates_file.close();
}
