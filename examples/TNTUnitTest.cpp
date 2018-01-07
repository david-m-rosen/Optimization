#include <Eigen/Dense>

#include "Optimization/Smooth/TNT.h"

using namespace std;

double test_error_threshold = 1e-6;
bool passed_test;
double numerical_diff_stepsize = 1e-5;

// The use of floats seems to be causing a loss of precision here ...
double numerical_diff_relative_error_threshold = 1e-3;

typedef Eigen::VectorXd Variable;
typedef Eigen::VectorXd Tangent;

int main() {
  // Standard Euclidean inner product for vectors
  Optimization::Smooth::RiemannianMetric<Variable, Tangent> metric =
      [](const Variable &X, const Tangent &V1, const Tangent &V2) {
        return V1.dot(V2);
      };

  /// Test Steihaug-Toint truncated conjugate-gradient method here

  // Dummy placeholder vector to represent the iterate at which we are
  // constructing the linear model
  Variable X;

  // Null-op preconditioner = identity
  std::experimental::optional<
      Optimization::Smooth::LinearOperator<Variable, Tangent>>
      Id(std::experimental::nullopt);

  /// First, test with PSD Hessian, infinite trust-region radius, and *exact*
  /// stopping criterion to ensure that the
  /// underlying PCG methd is implemented properly

  Eigen::DiagonalMatrix<double, 3> P(1, 2, 3);
  Optimization::Smooth::LinearOperator<Variable, Tangent> POp =
      [&](const Variable &X, const Tangent &V) { return P * V; };

  // Sample an initial gradient vector
  Tangent g = Tangent::Random(3);

  cout << "TESTING STEIHAUG-TOINT TRUNCATED CONJUGATE GRADIENT METHOD" << endl
       << endl;

  cout << "TEST BASIC PCG: " << endl;
  cout << "Testing STPCG with positive-definite diagonal Hessian H = " << endl
       << Eigen::MatrixXd(P) << endl
       << "randomly-sampled gradient g = " << endl
       << g.transpose() << endl
       << "and *exact* termination criteria." << endl
       << endl;

  double update_step_norm;
  unsigned int num_iterations;
  Tangent V0 = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g, POp, metric, Id, update_step_norm, num_iterations,
      std::numeric_limits<double>::max(), g.size(), 0);

  double V0_error = (V0 + P.inverse() * g).norm();
  double V0_norm_error = fabs(V0.norm() - update_step_norm);
  cout << "Error in estimated update step: " << V0_error << endl;
  cout << "Error in reported norm of update step: " << V0_norm_error << endl
       << endl
       << endl;
  assert(V0_error < test_error_threshold);
  assert(V0_norm_error < test_error_threshold);

  /// Test STPCG with negative-definite diagonal Hessian to ensure that the step
  /// to the boundary of the trust - region
  /// radius is computed properly
  double Delta = 100;

  Eigen::MatrixXd N = -1.0 * P;
  Optimization::Smooth::LinearOperator<Variable, Tangent> NOp =
      [&](const Variable &X, const Tangent &V) { return N * V; };
  cout << "TEST NEGATIVE CURVATURE HANDLING: " << endl;
  cout << "Testing STPCG with negative-definite diagonal Hessian H = " << endl
       << N << endl
       << "randomly-sampled gradient g = " << endl
       << g.transpose() << endl
       << "and *exact* termination criteria (to exercise negative curvature "
          "detection and scaling to the boundary)."
       << endl;

  Tangent V1 = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g, NOp, metric, Id, update_step_norm, num_iterations, Delta, g.size(),
      0);
  Tangent target_solution = -(Delta / g.norm()) * g;

  double V1_error = (target_solution - V1).norm();
  double V1_norm_error = fabs(V1.norm() - update_step_norm);

  cout << "Norm of estimated solution: " << V1.norm() << endl;
  cout << "Delta = " << Delta << endl << endl;
  cout << "Error in estimated update step: " << V1_error << endl;
  cout << "Error in reported norm of update step: " << V1_norm_error << endl
       << endl;
  assert(V1_error < test_error_threshold);
  assert(V1_norm_error < test_error_threshold);

  /// Now test using preconditioning
  Eigen::DiagonalMatrix<double, 3> M(9.0, 10.0, 11.0);

  Eigen::MatrixXd Minv = M.inverse();

  Optimization::Smooth::LinearOperator<Variable, Tangent> MinvOp =
      [&](const Variable &X, const Tangent &V) { return Minv * V; };

  cout << "TESTING PCG: " << endl;
  cout << "Testing STPCG with positive-definite diagonal Hessian H = " << endl
       << Eigen::MatrixXd(P) << endl
       << "randomly-sampled gradient g = " << endl
       << g.transpose() << endl
       << "preconditioner M^{-1}: " << endl
       << Minv << endl
       << "and *exact* termination criteria." << endl
       << endl;
  Tangent V2 = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g, POp, metric, MinvOp, update_step_norm, num_iterations,
      std::numeric_limits<double>::max(), g.size(), 0);

  double V2_error = (P * V2 + g).norm();
  double V2_norm_error = fabs(update_step_norm - sqrt(V2.dot(M * V2)));
  cout << "Error in estimated update step: " << V2_error << endl;
  cout << "Error in reported M-norm of update step: " << V2_norm_error << endl
       << endl;
  assert(V2_error < test_error_threshold);
  assert(V2_norm_error < test_error_threshold);

  cout << "TESTING PCG WITH NEGATIVE CURVATURE HANDLING: " << endl;
  cout << "Testing STPCG with negative-definite diagonal Hessian H = " << endl
       << N << endl
       << "randomly-sampled gradient g = " << endl
       << g.transpose() << endl
       << "preconditioner M^{-1}: " << endl
       << Minv << endl
       << "and *exact* termination criteria." << endl
       << endl;

  Tangent V3 = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g, NOp, metric, MinvOp, update_step_norm, num_iterations, Delta,
      g.size(), 0);

  // Compute M-norm of initial search direction p = -M^{-1} * g:
  Tangent p = -Minv * g;
  double p_M_norm = sqrt(p.dot(M * p));
  Tangent p_scaled = (Delta / p_M_norm) * p;

  double V3_error = (V3 - p_scaled).norm();
  double V3_norm_error = fabs(update_step_norm - sqrt(V3.dot(M * V3)));
  cout << "Delta = " << Delta << endl;
  cout << "M-norm of returned update step: " << sqrt(V3.dot(M * V3)) << endl;
  cout << "Reported M-norm of returned update step: " << update_step_norm
       << endl;
  cout << "Error in estimated update step: " << V3_error << endl;
  cout << "Error in reported M-norm of update step: " << V3_norm_error << endl
       << endl;
  assert(V3_error < test_error_threshold);
  assert(V3_norm_error < test_error_threshold);

  cout << "TESTING PCG WITH TRUNCATION CRITERIA:" << endl;
  constexpr unsigned int dim = 1000;
  Eigen::DiagonalMatrix<double, dim> H_big =
      (2 * Eigen::VectorXd::Ones(dim) + Eigen::VectorXd::Random(dim))
          .asDiagonal();
  Optimization::Smooth::LinearOperator<Variable, Tangent> H_bigOp =
      [&](const Variable &X, const Tangent &V) { return H_big * V; };

  Tangent g_big = Eigen::VectorXd::Random(dim);

  cout << "Testing STPCG with positive-definite diagonal Hessian of dimension "
       << dim << ", randomly-sampled gradient, and inexact stopping "
                 "(truncation) criteria."
       << endl;
  Tangent V4_exact = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g_big, H_bigOp, metric, Id, update_step_norm, num_iterations, Delta,
      dim, 0);

  double V4_exact_error = (H_big * V4_exact + g_big).norm();
  double V4_exact_norm_error = fabs(update_step_norm - V4_exact.norm());
  cout << "Error in exact (no early-truncation) solution: " << V4_exact_error
       << endl;
  cout << "Error in reported norm of exact (no early-truncation) solution: "
       << V4_exact_norm_error << endl
       << endl;
  assert(V4_exact_error < test_error_threshold);
  assert(V4_exact_norm_error < test_error_threshold);

  double kappa_fgr = .1;
  double theta = 1;

  Eigen::VectorXd V4_truncated = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g_big, H_bigOp, metric, Id, update_step_norm, num_iterations, Delta,
      dim, kappa_fgr, theta);

  double final_gradnorm = (H_big * V4_truncated + g_big).norm();
  double V4_truncated_relative_error = final_gradnorm / g_big.norm();
  double V4_truncated_norm_error = fabs(update_step_norm - V4_truncated.norm());
  cout << "Relative error in early-truncation solution = "
       << V4_truncated_relative_error
       << " (should be less than kappa_fgr = " << kappa_fgr << ")" << endl;
  cout << "Error in reported norm of update step: " << V4_truncated_norm_error
       << endl
       << endl;

  assert(V4_truncated_relative_error < kappa_fgr);
  assert(V4_truncated_norm_error < test_error_threshold);

  /// Now apply a randomly-sampled positive-definite diagonal preconditioner as
  /// well

  Eigen::VectorXd diag =
      3 * Eigen::VectorXd::Ones(dim) + Eigen::VectorXd::Random(dim);
  Eigen::DiagonalMatrix<double, dim> P_big = diag.asDiagonal();
  Optimization::Smooth::LinearOperator<Variable, Tangent> P_bigOp =
      [&](const Variable &X, const Tangent &V) { return P_big * V; };

  Eigen::VectorXd diag_inv = diag.cwiseInverse();

  cout << "Testing STPCG with positive-definite diagonal Hessian of dimension "
       << dim << ", randomly-sampled gradient, randomly-sampled PSD diagonal "
                 "preconditioner, and inexact stopping  "
                 "(truncation) criteria."
       << endl;
  Tangent V5 = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g_big, H_bigOp, metric, P_bigOp, update_step_norm, num_iterations,
      Delta, dim, kappa_fgr, theta);

  double V5_residual_norm = (H_big * V5 + g_big).norm();
  double V5_relative_error = V5_residual_norm / g_big.norm();
  double V5_norm_error =
      fabs(update_step_norm - sqrt(V5.dot(diag_inv.cwiseProduct(V5))));

  cout << "Relative error in early-truncation solution: " << V5_relative_error
       << endl;
  cout << "Error in reported M-norm of solution: " << V5_norm_error << endl
       << endl;
  assert(V5_relative_error < kappa_fgr);
  assert(V5_norm_error < test_error_threshold);

  cout << "Testing STPCG with a randomly-sampled (probably indefinite) Hessian "
          "of dimension "
       << dim << ", randomly-sampled gradient, randomly-sampled PSD diagonal "
                 "preconditioner, and inexact stopping  "
                 "(truncation) criteria."
       << endl;

  Eigen::DiagonalMatrix<double, dim> I_big =
      (5 * Eigen::VectorXd::Ones(dim)).asDiagonal();
  Optimization::Smooth::LinearOperator<Variable, Tangent> I_bigOp =
      [&](const Variable &X, const Tangent &V) { return I_big * V; };

  Tangent V6 = Optimization::Smooth::STPCG<Variable, Tangent>(
      X, g_big, I_bigOp, metric, P_bigOp, update_step_norm, num_iterations,
      Delta, dim, kappa_fgr, theta);

  double V6_residual_norm = (I_big * V6 + g_big).norm();
  double V6_relative_error = V6_residual_norm / g_big.norm();
  double V6_norm_error =
      fabs(update_step_norm - sqrt(V6.dot(diag_inv.cwiseProduct(V6))));

  cout << "Relative error in early-truncation solution: " << V6_relative_error
       << endl;
  cout << "Error in reported M-norm of solution: " << V6_norm_error << endl
       << endl;
  assert(V6_relative_error < kappa_fgr);
  assert(V6_norm_error < test_error_threshold);

  cout << "END STEIHAUG-TOINT CONJUGATE GRADIENT TESTS" << endl
       << endl
       << endl
       << endl;

  /// TEST TRUNCATED-NEWTON TRUST-REGION METHOD HERE!

  cout << "TESTING TRUNCATED NEWTON TRUST-REGION METHOD" << endl << endl;

  // Parameters for Rosenbrock test function
  double a = 1.0;
  double b = 100;

  // Construct 2D Rosenbrock objective function
  Optimization::Objective<Variable, Eigen::Matrix2d> Rosenbrock =
      [&](const Variable &x, Eigen::Matrix2d &Hess) {
        double x2 = x(0) * x(0);
        double a_minus_x = a - x(0);
        double y_minus_x2 = x(1) - x2;

        return a_minus_x * a_minus_x + b * y_minus_x2 * y_minus_x2;
      };

  // Construct 2D Rosenbrock quadratic model
  Optimization::Smooth::QuadraticModel<Variable, Tangent, Eigen::Matrix2d>
      quadratic_model =
          [&](const Variable &x, Tangent &grad,
              Optimization::Smooth::LinearOperator<Variable, Tangent,
                                                   Eigen::Matrix2d> &HessOp,
              Eigen::Matrix2d &Hess) {
            double x2 = x(0) * x(0);
            double a_minus_x = a - x(0);
            double y_minus_x2 = x(1) - x2;

            // Evaluate gradient
            grad.resize(2);
            grad(0) = -2 * a_minus_x - 4 * b * x(0) * y_minus_x2;
            grad(1) = 2 * b * y_minus_x2;

            // Evaluate Hessian
            Hess(0, 0) = 12 * b * x2 - 4 * b * x(1) + 2;
            Hess(0, 1) = -4 * b * x(0);
            Hess(1, 0) = Hess(0, 1);
            Hess(1, 1) = 2 * b;

            HessOp = [&Hess](const Variable &x, const Tangent &v,
                             Eigen::Matrix2d &H) { return Hess * v; };
          };

  // Redefinition of metric to conform to the required signature (must accept an
  // Eigen::Matrix2d& as a cache variable)
  Optimization::Smooth::RiemannianMetric<Variable, Tangent, Eigen::Matrix2d>
      Rosenbrock_metric = [](const Variable &X, const Tangent &V1,
                             const Tangent &V2,
                             Eigen::Matrix2d &Hess) { return V1.dot(V2); };

  // Construct preconditioner for 2D Rosenbrock function
  Optimization::Smooth::LinearOperator<Variable, Tangent, Eigen::Matrix2d>
      precon = [](const Variable &x, const Tangent &v, Eigen::Matrix2d &H) {
        return H.inverse() * v;
      };

  Optimization::Smooth::Retraction<Variable, Tangent, Eigen::Matrix2d>
      Euclidean_retraction = [](const Variable &x, const Tangent &v,
                                Eigen::Matrix2d Hess) { return x + v; };

  /// TEST OPTIMIZATION ON ROSENBROCK FUNCTION

  double scale = 5;
  unsigned int num_samples = 1;

  cout << "Testing truncated-Newton trust-region method on the 2D Rosenbrock "
          "function "
       << endl
       << endl;
  cout << "Sampling " << num_samples << " random starting points on the "
       << scale << " x " << scale << " grid ... " << endl;

  Optimization::Smooth::TNTParams params;
  params.Delta0 = 100;
  params.preconditioned_gradient_tolerance = 1e-6;
  params.verbose = true;

  double max_returned_val = -1;
  double max_preconditioned_returned_val = -1;

  for (unsigned int i = 0; i < num_samples; i++) {
    Variable x0 = scale * Eigen::VectorXd::Random(2);

    Eigen::Matrix2d Hess;

    /// Optimize without preconditioning

    if (params.verbose)
      cout << "Optimizing without preconditioning: " << endl << endl;

    Optimization::Smooth::TNTResult<Variable> result =
        Optimization::Smooth::TNT<Variable, Tangent, Eigen::Matrix2d>(
            Rosenbrock, quadratic_model, Rosenbrock_metric,
            Euclidean_retraction, x0, Hess, std::experimental::nullopt, params);

    max_returned_val =
        (result.f > max_returned_val ? result.f : max_returned_val);

    if (params.verbose)
      cout << endl << "Final objective value: " << result.f << endl << endl;

    assert(result.f < 1e-2);

    /// Optimize with preconditioning

    if (params.verbose)
      cout << "Optimizing with preconditioning: " << endl << endl;

    result = Optimization::Smooth::TNT<Variable, Tangent, Eigen::Matrix2d>(
        Rosenbrock, quadratic_model, Rosenbrock_metric, Euclidean_retraction,
        x0, Hess, precon, params);

    max_preconditioned_returned_val =
        (result.f > max_preconditioned_returned_val
             ? result.f
             : max_preconditioned_returned_val);

    if (params.verbose)
      cout << endl << "Final objective value: " << result.f << endl << endl;

    assert(result.f < 1e-2);
  }

  cout << "Maximum value of function at stopping across all non-preconditioned "
          "runs: "
       << max_returned_val << endl;
  cout << "Maximum value of function at stopping across all preconditioned "
          "runs: "
       << max_preconditioned_returned_val << endl
       << endl;

  cout << endl << "All tests passed" << endl;
}
