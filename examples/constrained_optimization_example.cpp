/** This example demonstrates how to use the TRSQP nonlinear programming
 * algorithm to solve a nonconvex nonlinear program
 *
 * Copyright (c) David M. Rosen (dmrosen@mit.edu)
 */
#include <fstream>
#include <type_traits> // To get static casting of enum types

// Add functions to Eigen's MatrixBase template class that are required to
// instantiate the TRSQP templates
#define EIGEN_MATRIXBASE_PLUGIN                                                \
  "Optimization/Constrained/TRSQPEigenExtensions.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;

/// Basic data types
typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

#include "Optimization/Constrained/TRSQP.h"

/// TRSQP TYPEDEFS

/// Function types
typedef Optimization::Objective<Vector, Scalar> ObjFunction;
typedef Optimization::Constrained::PairFunction<Vector, Vector, Vector>
    ConstraintFunction;
typedef Optimization::Constrained::VectorFunction<Vector> GradientFunction;
typedef Optimization::Constrained::PairFunction<Vector, SparseMatrix,
                                                SparseMatrix>
    JacobianFunction;
typedef Optimization::Constrained::HessianFunction<Vector, Vector, Vector,
                                                   SparseMatrix>
    HessFunction;

typedef Optimization::Constrained::AugmentedSystemSolverFunction<
    Vector, Vector, Vector, SparseMatrix, SparseMatrix>
    AugmentedSystemFunction;

typedef Optimization::Constrained::KKTSystemSolverFunction<
    Vector, Vector, Vector, SparseMatrix, SparseMatrix, SparseMatrix>
    KKTSystemFunction;

typedef Optimization::Constrained::PrimalDualStrategyFunction<
    Vector, Vector, Vector, SparseMatrix, SparseMatrix, SparseMatrix, Scalar>
    PDStrategyFunction;

/// Linear system solvers

AugmentedSystemFunction augmented_system_solver =
    [](const Optimization::Constrained::Pair<SparseMatrix, SparseMatrix> &Ax,
       const Vector &s, bool new_coefficient_matrix,
       const Optimization::Constrained::Pair<Vector, Vector> &b,
       const Optimization::Constrained::Pair<Vector, Vector> &c,
       Optimization::Constrained::Pair<Vector, Vector> &v,
       Optimization::Constrained::Pair<Vector, Vector> &y) {
      /// Get problem size

      size_t n = b.first.size();
      size_t me = c.first.size();
      size_t mi = c.second.size();
      size_t m = me + mi;

      size_t D = n + me + 2 * mi;

      /// Construct scaled augmented Jacobian Ahat
      Matrix Ahat = Matrix::Zero(m, n + mi);
      if (me > 0)
        Ahat.topLeftCorner(me, n) = Ax.first;
      if (mi > 0) {
        Ahat.bottomLeftCorner(mi, n) = Ax.second;
        Ahat.bottomRightCorner(mi, mi) = s.asDiagonal();
      }

      /// Construct augmented system matrix M
      Matrix M = Matrix::Zero(D, D);
      M.topLeftCorner(n + mi, n + mi).setIdentity();
      M.topRightCorner(n + mi, m) = Ahat.transpose();
      M.bottomLeftCorner(m, n + mi) = Ahat;

      /// Construct right-hand side vector for augmented linear system
      Vector rhs(D);
      rhs.head(n) = b.first;
      if (mi > 0)
        rhs.segment(n, mi) = b.second;
      if (me > 0)
        rhs.segment(n + mi, me) = c.first;
      if (mi > 0)
        rhs.segment(n + me + mi, mi) = c.second;

      /// SOLVE LINEAR SYSTEM!
      Eigen::FullPivHouseholderQR<Matrix> Mfact(M);
      Vector x = Mfact.solve(rhs);

      /// Extract and return solution components
      v.first = x.head(n);

      if (mi > 0)
        v.second = x.segment(n, mi);
      if (me > 0)
        y.first = x.segment(n + mi, me);
      if (mi > 0)
        y.second = x.segment(n + me + mi, mi);
    };

// Augmented linear system solver
KKTSystemFunction kkt_system_solver =
    [](const SparseMatrix &H, const Vector &Sigma,
       const Optimization::Constrained::Pair<SparseMatrix, SparseMatrix> &Ax,
       bool new_coefficient_matrix,
       const Optimization::Constrained::Pair<Vector, Vector> &b,
       const Optimization::Constrained::Pair<Vector, Vector> &c,
       Optimization::Constrained::Pair<Vector, Vector> &v,
       Optimization::Constrained::Pair<Vector, Vector> &y) -> void {
  /// Get problem size
  size_t n = b.first.size();
  size_t me = c.first.size();
  size_t mi = c.second.size();
  size_t m = me + mi;

  size_t D = n + me + 2 * mi;

  /// Construct augmented Hessian W
  Matrix W = Matrix::Zero(n + mi, n + mi);
  W.topLeftCorner(n, n) = H;
  if (mi > 0)
    W.bottomRightCorner(mi, mi) = Sigma.asDiagonal();

  /// Construct augmented Jacobian Abar
  Matrix Abar = Matrix::Zero(m, n + mi);
  if (me > 0)
    Abar.topLeftCorner(me, n) = Ax.first;
  if (mi > 0) {
    Abar.bottomLeftCorner(mi, n) = Ax.second;
    Abar.bottomRightCorner(mi, mi).setIdentity();
  }

  /// Construct KKT system matrix K

  Matrix K = Matrix::Zero(D, D);
  K.topLeftCorner(n + mi, n + mi) = W;
  K.topRightCorner(n + mi, m) = Abar.transpose();
  K.bottomLeftCorner(m, n + mi) = Abar;

  /// Construct right-hand side vector for augmented linear system
  Vector rhs(D);
  rhs.head(n) = b.first;
  if (mi > 0)
    rhs.segment(n, mi) = b.second;
  if (me > 0)
    rhs.segment(n + mi, me) = c.first;
  if (mi > 0)
    rhs.segment(n + me + mi, mi) = c.second;

  /// SOLVE LINEAR SYSTEM!

  // Compute symmetric eigendecomposition of K
  Eigen::SelfAdjointEigenSolver<Matrix> Kfact(K);

  // Solve linear system: K = V D V' => K^-1 = V D^-1 V'
  // x = V * D^-1 * V'
  Vector x = Kfact.eigenvectors() *
             Kfact.eigenvalues().cwiseInverse().asDiagonal() *
             Kfact.eigenvectors().transpose() * rhs;

  /// Extract and return solution components
  v.first = x.head(n);

  if (mi > 0)
    v.second = x.segment(n, mi);
  if (me > 0)
    y.first = x.segment(n + mi, me);
  if (mi > 0)
    y.second = x.segment(n + me + mi, mi);
};

PDStrategyFunction pd_strategy =
    [](size_t k, double t, const Vector &x, const Vector &s,
       const Optimization::Constrained::Pair<Vector, Vector> &lambda, Scalar fx,
       const Vector &gradfx, const SparseMatrix &Hx, const Vector &Sigma,
       const Optimization::Constrained::Pair<Vector, Vector> &cx,
       const Optimization::Constrained::Pair<SparseMatrix, SparseMatrix> &Ax,
       Scalar mu, Scalar Delta,
       Optimization::Constrained::TRSQPStepType prev_step_type, size_t CG_iters,
       const Optimization::Constrained::Pair<Vector, Vector> &d,
       bool prev_step_accepted) -> bool {
  // Super simple strategy: always attempt primal-dual step computation
  return true;
};

int main() {

  /** We solve:
   *
   * min f(x,y)
   *
   * s.t. 1/x - y <= 0
   *      x + sin(x) - y = 0
   *
   * where f(x,y) is the six-hump camel function:
   *
   * f(x,y) = (4 - 2.1x^2 + x^4/3)x^2 + xy + (-4 + 4y^2)y^2
   *        = x^6 / 3 - 2.1x^4 + 4x^2 + xy - 4y^2 + 4y^4
   */

  /// Objective data

  /// Objective
  ObjFunction f = [](const Vector &v) -> Scalar {
    const Scalar &x = v(0);
    Scalar x2 = x * x;
    Scalar x4 = x2 * x2;
    Scalar x6 = x2 * x4;

    const Scalar &y = v(1);
    Scalar y2 = y * y;
    Scalar y4 = y2 * y2;

    return x6 / 3 - 2.1 * x4 + 4 * x2 + x * y - 4 * y2 + 4 * y4;
  };

  /// Gradient of objective
  GradientFunction gradf = [](const Vector &v) -> Vector {
    const Scalar &x = v(0);
    Scalar x2 = x * x;
    Scalar x3 = x2 * x;
    Scalar x5 = x3 * x2;

    const Scalar &y = v(1);
    Scalar y2 = y * y;
    Scalar y3 = y2 * y;

    Vector g(2);

    // Gradient of f wrt x
    g(0) = 2 * x5 - 8.4 * x3 + 8 * x + y;

    // Gradient of f wrt y
    g(1) = 16 * y3 - 8 * y + x;

    return g;
  };

  /// Constraint function
  ConstraintFunction c =
      [](const Vector &v) -> Optimization::Constrained::Pair<Vector, Vector> {
    const Scalar &x = v(0);
    const Scalar &y = v(1);

    // Equality constraint: x + sin(x) - y = 0
    Vector ce(1);
    ce(0) = x + sin(x) - y;

    // Inequality constraint: 1/x - y <= 0
    Vector ci(1);
    ci(0) = 1 / x - y;

    return Optimization::Constrained::Pair<Vector, Vector>(ce, ci);
  };

  /// Jacobian function
  JacobianFunction A = [](const Vector &v)
      -> Optimization::Constrained::Pair<SparseMatrix, SparseMatrix> {
    const Scalar &x = v(0);
    Scalar x2 = x * x;
    const Scalar &y = v(1);

    // Jacobian of equality constraint: x + sin(x) - y = 0
    SparseMatrix Ae(1, 2);
    Ae.insert(0, 0) = 1 + cos(x);
    Ae.insert(0, 1) = -1;

    // Jacobian of inequality constraint: 1/x - y <= 0
    SparseMatrix Ai(1, 2);
    Ai.insert(0, 0) = -1 / x2;
    Ai.insert(0, 1) = -1;

    return Optimization::Constrained::Pair<SparseMatrix, SparseMatrix>(Ae, Ai);
  };

  /// Hessian of Lagrangian wrt x
  HessFunction H =
      [](const Vector &v,
         const Optimization::Constrained::Pair<Vector, Vector> &lambda)
      -> SparseMatrix {
    const Scalar &x = v(0);
    Scalar x2 = x * x;
    Scalar x3 = x2 * x;
    Scalar x4 = x2 * x2;

    const Scalar &y = v(1);
    Scalar y2 = y * y;

    // Construct Hessian of objective f
    Eigen::MatrixXd Hessf(2, 2);
    Hessf(0, 0) = 10 * x4 - 25.2 * x2 + 8;
    Hessf(0, 1) = 1;
    Hessf(1, 0) = 1;
    Hessf(1, 1) = 48 * y2 - 8;

    // Construct Hessian of equality constraint function: x + sin(x) - y = 0
    Eigen::MatrixXd Hessce = Eigen::MatrixXd::Zero(2, 2);
    Hessce(0, 0) = -sin(x);

    // Construct Hessian of inequality constraint function: 1/x - y <= 0
    Eigen::MatrixXd Hessci = Eigen::MatrixXd::Zero(2, 2);
    Hessci(0, 0) = 2 / x3;

    Eigen::MatrixXd Hdense =
        Hessf + lambda.first(0) * Hessce + lambda.second(0) * Hessci;

    return Hdense.sparseView();
  };

  /// INITIALIZATION

  // Set initial vector x0
  Vector x0(2);
  x0 << 5, 7;

  Scalar f0 = f(x0);
  Optimization::Constrained::Pair<Vector, Vector> c0 = c(x0);
  // Initial constraint violation
  Scalar c0_norm =
      Optimization::Constrained::compute_constraint_residuals(c0).norm();

  cout << "Initial point: x0 = (" << x0.transpose() << ")" << endl << endl;

  cout << "Initial objective value: " << f0 << endl;
  cout << "Initial constraint violation: " << c0_norm << endl << endl;

  /// SET UP NITRO
  Optimization::Constrained::TRSQPParams<Scalar> params;
  params.log_iterates = true;
  params.max_iterations = 100;
  params.mu0 = 1e-4;
  params.epsilon_mu0 = 1e-2;
  params.gradient_tolerance = 1e-6;
  params.infeasibility_tolerance = 1e-6;
  params.complementarity_tolerance = 1e-6;
  params.verbose = true;

  /// RUN OPTIMIZER!!!

  cout << "Running TRSQP optimizer ... " << endl << endl;
  Optimization::Constrained::TRSQPResult<Vector, Vector, Vector> result =
      Optimization::Constrained::TRSQP<Vector, Vector, Vector, SparseMatrix,
                                       SparseMatrix, SparseMatrix>(
          f, c, gradf, A, H, augmented_system_solver, x0, params,
          std::experimental::optional<KKTSystemFunction>(kkt_system_solver),
          std::experimental::optional<PDStrategyFunction>(pd_strategy));

  cout << endl << "TRSQP optimizer finished!" << endl << endl;

  const Vector &xf = result.x;
  cout << "Final estimate: x = (" << xf.transpose() << ")" << endl;
  cout << "Final objective value: " << result.f << endl;
  cout << "Constraint violation: " << result.infeas_norm << endl;
  cout << "Gradient of Lagrangian wrt x: " << result.grad_Lx_norm << endl;
  cout << "Final complementarity error: " << result.complementarity_norm
       << endl;

  /// WRITE A BUNCH OF STUFF TO DISK
  // Save sequence of objective values
  string obj_vals_filename = "objective_values.txt";
  ofstream obj_vals_file(obj_vals_filename);
  for (const auto f : result.objective_values)
    obj_vals_file << f << " ";
  obj_vals_file << endl;
  obj_vals_file.close();

  // Save sequence of infeasibility norms
  string infeasibility_norms_filename = "infeasibility_norms.txt";
  ofstream infeasibility_norms_file(infeasibility_norms_filename);
  for (const auto cnorm : result.infeas_norms)
    infeasibility_norms_file << cnorm << " ";
  infeasibility_norms_file << endl;
  infeasibility_norms_file.close();

  // Save sequence of gradient of Lagrangian norms
  string gradLx_norms_filename = "gradLx_norms.txt";
  ofstream gradLx_norms_file(gradLx_norms_filename);
  for (const auto gLx : result.grad_Lx_norms)
    gradLx_norms_file << gLx << " ";
  gradLx_norms_file << endl;
  gradLx_norms_file.close();

  // Save sequence of complementarity residual norms
  string comp_norms_filename = "comp_norms.txt";
  ofstream comp_norms_file(comp_norms_filename);
  for (const auto comp_norm : result.complementarity_norms)
    comp_norms_file << comp_norm << " ";
  comp_norms_file << endl;
  comp_norms_file.close();

  // Save sequence of gain ratios norms
  string gain_ratios_filename = "gain_ratios.txt";
  ofstream gain_ratios_file(gain_ratios_filename);
  for (const auto gamma : result.gain_ratios)
    gain_ratios_file << gamma << " ";
  gain_ratios_file << endl;
  gain_ratios_file.close();

  // Save sequence of barrier parameters
  string barrier_params_filename = "barrier_params.txt";
  ofstream barrier_params_file(barrier_params_filename);
  for (const auto mu : result.barrier_params)
    barrier_params_file << mu << " ";
  barrier_params_file << endl;
  barrier_params_file.close();

  // Save sequence of penalty parameters
  string penalty_params_filename = "penalty_params.txt";
  ofstream penalty_params_file(penalty_params_filename);
  for (const auto nu : result.penalty_params)
    penalty_params_file << nu << " ";
  penalty_params_file << endl;
  penalty_params_file.close();

  // Save sequence of SOC applications
  string SOC_filename = "second_order_corrections.txt";
  ofstream SOC_file(SOC_filename);
  for (const bool soc : result.SOCs)
    SOC_file << soc << " ";
  SOC_file << endl;
  SOC_file.close();

  // Save sequence of step types parameters
  string step_types_filename = "step_types.txt";
  ofstream step_types_file(step_types_filename);
  for (const Optimization::Constrained::TRSQPStepType &step_type :
       result.step_types)
    step_types_file << static_cast<std::underlying_type_t<
                           Optimization::Constrained::TRSQPStepType>>(step_type)
                    << " ";
  step_types_file << endl;
  step_types_file.close();

  if (params.log_iterates) {
    size_t num_iterates = result.iterates.size();
    Eigen::MatrixXd X(2, num_iterates);
    for (size_t k = 0; k < num_iterates; ++k)
      X.col(k) = result.iterates[k];

    string iterates_filename = "iterates.txt";
    ofstream iterates_file(iterates_filename);
    iterates_file << X << endl;
    iterates_file.close();
  }
}
