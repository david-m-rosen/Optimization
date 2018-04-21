/** This simple example demonstrates how to use the simplified Euclidean
 * interfaces for the gradient descent and truncated Newton trust-region (TNT)
 * algorithms to minimize the McCormick function:
 *
 * f(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
 *
 * whose minimizer is
 *
 * f(-.54719, -1.54719) = -1.9133
 */

#include "Optimization/Smooth/GradientDescent.h"
#include "Optimization/Smooth/TNT.h"

#include <Eigen/Dense>

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

  // Gradient
  EuclideanVectorField<Vector> grad_F = [](const Vector &x) {
    return Vector(cos(x(0) + x(1)) + 2 * (x(0) - x(1)) - 1.5,
                  cos(x(0) + x(1)) - 2 * (x(0) - x(1)) + 2.5);
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

  // Preconditioner
  //
  // NB:  This preconditioning operator works by evaluating the Hessian matrix
  // Hess(x), computing a positive-definite approximation of its inverse using a
  // modified eigendecomposition, and then applying this PD approximation to
  // precondition the input vector v, EACH TIME IT IS CALLED.
  //
  // Obviously, there are much more efficient ways of implementing this
  // operation (including making use of the variadic template arguments to
  // define cache variables that can save/reuse intermediate parts of the
  // computation, such as the eigendecomposition, etc.)  The simpler (*much*
  // less efficient) implementation used here is employed for the sake of
  // clarity of exposition, since the computational inefficiency of this
  // approach is not a serious limitation on this small (2D example)

  EuclideanLinearOperator<Vector> precon =
      [&compute_Hessian](const Vector &x, const Vector &v) -> Vector {

    // Compute Hessian matrix H at x
    Eigen::Matrix2d H = compute_Hessian(x);

    // Compute eigendecomposition of H
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(
        H, Eigen::ComputeEigenvectors);

    // Modify eigenvalues of H (if necessary) to ensure that they are
    // greater than a minimum (positive) constant lambda_min
    double lambda_min = .01;
    Vector lambdas = eigensolver.eigenvalues();
    Vector lambdas_plus =
        lambdas.array().max(Eigen::Array2d::Constant(lambda_min));

    // Now use these to compute an approximate inverse Hessian
    Vector lambdas_plus_inv = lambdas_plus.cwiseInverse();

    // Compute approximate inverse Hessian
    Eigen::Matrix2d M = eigensolver.eigenvectors() *
                        lambdas_plus_inv.asDiagonal() *
                        eigensolver.eigenvectors().transpose();

    // Apply M to precondition v
    return M * v;
  };

  /// SAMPLE INITIAL POINT

  Vector x0(-.5, -.5);
  Vector x_target(-0.54719, -1.54719);

  cout << "Initial point: x0 = " << endl << x0 << endl << endl;

  /// RUN GRADIENT DESCENT OPTIMIZER!
  GradientDescentParams gd_params;
  gd_params.verbose = true;

  GradientDescentResult<Vector> gd_result =
      EuclideanGradientDescent<Vector>(F, grad_F, x0, gd_params);

  cout << "Final objective value: = " << gd_result.f << endl;
  cout << "Target global minimum: -1.9133" << endl;
  cout << "Error in final objective: " << gd_result.f + 1.9133 << endl << endl;

  cout << "Estimated minimizer: " << endl << gd_result.x << endl << endl;
  cout << "Target minimizer: " << endl << x_target << endl << endl;
  cout << "Error in minimizer estimate: " << (gd_result.x - x_target).norm()
       << endl
       << endl;

  /// RUN TNT OPTIMIZER!

  cout << "Running TNT optimizer!" << endl << endl;

  // Set TNT options
  Optimization::Smooth::TNTParams tnt_params;
  tnt_params.verbose = true;

  TNTResult<Vector> tnt_result =
      EuclideanTNT<Vector>(F, QM, x0, precon, tnt_params);

  cout << "Final objective value: = " << tnt_result.f << endl;
  cout << "Target global minimum: -1.9133" << endl;
  cout << "Error in final objective: " << tnt_result.f + 1.9133 << endl << endl;

  cout << "Estimated minimizer: " << endl << tnt_result.x << endl << endl;
  cout << "Target minimizer: " << endl << x_target << endl << endl;
  cout << "Error in minimizer estimate: " << (tnt_result.x - x_target).norm()
       << endl
       << endl;
}
