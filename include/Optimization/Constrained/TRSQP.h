/** This header file provides a lightweight template function implementing the
 * basic framework for a Byrd-Omojokun-type composite-step trust-region
 * sequential quadratic programming method for constrained nonlinear
 * optimization.  This implementation is based upon the Nonlinear Interior Point
 * Trust Region Optimizer (NITRO) algorithm described in the following papers:
 *
 * "An Interior Point Algorithm for Large-Scale Nonlinear Programming" by R.H.
 * Byrd, M.E. Hribar, and J. Nocedal
 *
 * "A Trust Region Method Based on Interior Point Techniques for Nonlinear
 * Programming" by R.H. Byrd, J.C. Gilbert, and J. Nocedal
 *
 * "An Interior Algorithm for Nonlinear Optimization that Combines Line Search
 * and Trust Region Steps" by R.A. Waltz, J.L. Morales, J. Nocedal, and D. Orban
 *
 * "On the Local Behavior of an Interior Point Method for Nonlinear Programming"
 * by R.H. Byrd, G. Liu, and J. Nocedal
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <exception>
#include <experimental/optional>
#include <functional>
#include <iostream>
#include <limits>

#include "Optimization/Constrained/Concepts.h"
#include "Optimization/Constrained/TRSQPSubproblemSolvers.h"
#include "Optimization/Constrained/TRSQPTypes.h"
#include "Optimization/Constrained/TRSQPUtilityFunctions.h"

#include "Optimization/Util/Stopwatch.h"

namespace Optimization {

namespace Constrained {

/** This template function implements a Byrd-Omojokun-type composite-step
 * trust-region sequential quadratic programming algorithm for solving
 * constrained optimization problems of the form:
 *
 * min_x f(x)
 * s.t.  ce(x) == 0
 *       ci(x) <= 0,
 *
 * where f: R^n -> R, ce: R^n -> R^{me}, and ci: R^n -> R^{mi} are all
 * twice-continuously-differentiable functions.  Here:
 *
 * - f is the objective function to be minimized
 *
 * - c is a function that accepts as input the point x, and returns a
 *   pair c(x) = (ce(x), ci(x)) containing the values of the (vector-valued)
 *   constraint functions ce and ci, evaluated at x.
 *
 * - gradf is a function that accepts as input the point x, and returns the
 *   gradient of f evaluate at x.
 *
 * - A is a function that accepts as input the point x, and returns the pair
 *   A(x) = (Ae(x), Ai(x)) of Jacobians of the constraint functions ce and ci,
 *   evaluated at x.
 *
 * - Hess is a function that accepts as input the point x and the pair
 *   (lambda_e, lambda_i) of Lagrange multipliers associated with the equality
 *   and inequality constraints, and returns the Hessian of the Lagrangian:
 *
 *      L(x, lambda) := f(x) + lambda_e'*ce(x) + lambda_i'*ci(x)
 *
 *   with respect to x, evaluated at (x,lambda)
 *
 * - x0 is the initial point at which to start the nonlinear optimization
 *
 * - args is a set of [optional] user-definable variadic template parameters
 *   that will be passed into each of the user-supplied functions whenever
 *   they are called
 *
 * - params is an [optional] set of parameters for the TRSQP algorithm.
 *
 * - augmented_linear_system_solver() and kkt_system_solver() are user-supplied
 *   functions for solving the linear systems needed to compute composite-step
 *   trust-region SQP and primal-dual Newton steps, respectively
 *
 * - primal_dual_strategy() is a user-supplied function that returns 'true'
 *   whenever a primal-dual update should be attempted.
 *
 * - user_function is an [optional] user-supplied function that is provided
 *   access to various quantities of interest as the algorithm runs, and can
 *   also be used to terminate the algorithm by returning 'true'
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Hessian,
          typename Scalar = double, typename... Args>
TRSQPResult<Vector, EqVector, IneqVector, Scalar> TRSQP(
    const Objective<Vector, Scalar, Args...> &f,
    const PairFunction<Vector, EqVector, IneqVector, Args...> &c,
    const VectorFunction<Vector, Args...> &gradf,
    const PairFunction<Vector, EqJacobian, IneqJacobian, Args...> &A,
    const HessianFunction<Vector, EqVector, IneqVector, Hessian, Args...> &Hess,
    const AugmentedSystemSolverFunction<Vector, EqVector, IneqVector,
                                        EqJacobian, IneqJacobian, Args...>
        &augmented_linear_system_solver,
    const Vector &x0, Args &... args,
    const TRSQPParams<Scalar> &params = TRSQPParams<Scalar>(),
    const std::experimental::optional<KKTSystemSolverFunction<
        Vector, EqVector, IneqVector, EqJacobian, IneqJacobian, Hessian,
        Args...>> &kkt_system_solver = std::experimental::nullopt,
    const std::experimental::optional<PrimalDualStrategyFunction<
        Vector, EqVector, IneqVector, EqJacobian, IneqJacobian, Hessian, Scalar,
        Args...>> &primal_dual_strategy = std::experimental::nullopt,
    const std::experimental::optional<TRSQPUserFunction<
        Vector, EqVector, IneqVector, EqJacobian, IneqJacobian, Hessian, Scalar,
        Args...>> &user_function = std::experimental::nullopt) {

  /// Argument checking

  /// Termination criteria

  if (params.infeasibility_tolerance < 0)
    throw std::invalid_argument(
        "Infeasibility tolerance must be a nonnegative real value");

  if (params.gradient_tolerance < 0)
    throw std::invalid_argument(
        "Gradient tolerance must be a nonnegative real value");

  if (params.complementarity_tolerance < 0)
    throw std::invalid_argument(
        "Complementarity tolerance must be a nonnegative real value");

  if (params.max_iterations < 0)
    throw std::invalid_argument(
        "Maximum number of iterations must be a nonnegative integer value");

  if (params.max_STPCG_iterations < 0)
    throw std::invalid_argument(
        "Maximum number of conjugate gradient iterations "
        "must be a nonnegative integer value");

  if (params.max_computation_time < 0)
    throw std::invalid_argument(
        "Maximum computation time must be a nonnegative real value");

  /// Trust-region control parameters

  if (params.DeltaMin <= 0)
    throw std::invalid_argument("Minimum admissible trust-region radius must "
                                "be a positive real value");

  if (params.Delta0 < params.DeltaMin)
    throw std::invalid_argument(
        "Initial trust-region radius must be a greater than or equal to "
        "minimum admissible trust-region radius");

  if ((params.zeta <= 0) || (params.zeta >= 1))
    throw std::invalid_argument(
        "Trust-region scaling factor for normal step "
        "computation (zeta) must be a value in the range (0,1)");

  if ((params.tau <= 0) || (params.tau >= 1))
    throw std::invalid_argument(
        "Admissible fractional distance to the boundary for slack variable "
        "updates (tau_min) must be in the range (0, 1)");

  if ((params.eta1 <= 0) || (params.eta1 > params.eta2))
    throw std::invalid_argument(
        "Threshold on gain ratio for accepting an update "
        "step must satisfy 0 < eta1 <= eta2)");

  if (params.eta2 > 1)
    throw std::invalid_argument(
        "Threshold on gain ratio for a very successful update step "
        "must satisfy eta1 <= eta2 <= 1");

  if ((params.alpha1 <= 0) || (params.alpha1 >= 1))
    throw std::invalid_argument(
        "Trust-region contraction factor for unsuccessful "
        "steps must satisfy 0 < alpha1 < 1");

  if (params.alpha2 <= 1)
    throw std::invalid_argument("Trust-region expansion factor for successful "
                                "updates must satisfy alpha2 > 1");

  if (params.s0_min <= 0)
    throw std::invalid_argument(
        "Elementwise lower-bound for slack initialization "
        "(s0_min) must be a positive real value");

  /// STPCG parameters
  if ((params.cg_kappa_fgr < 0) || (params.cg_kappa_fgr >= 1))
    throw std::invalid_argument(
        "Target fractional reduction in the preconditioned gradient norm for "
        "the STPGC method must satisfy 0 <= cg_kappa_fgr < 1");

  if ((params.cg_theta < 0) || (params.cg_theta > 1))
    throw std::invalid_argument(
        "Target asymptotic convergene rate parameter for "
        "the STPCG method must "
        "satisfy 0 <= cg_theta <= 1");

  if ((params.ls_alpha <= 0) || (params.ls_alpha >= 1))
    throw std::invalid_argument("Armijo linesearch sufficient decrease "
                                "parameter must be in the range (0,1");

  if (params.max_ls_iters < 0)
    throw std::invalid_argument("Maximum number of line search iterations must "
                                "be a nonnegative integer");

  if (params.alpha_min < 0 || params.alpha_min > 1)
    throw std::invalid_argument(
        "Minimum stepsize must be a real value in the range [0, 1]");

  if (bool(primal_dual_strategy) != bool(kkt_system_solver))
    throw std::invalid_argument(
        "Either *both* compute_primal_dual_step() and primal_dual_strategy() "
        "must be supplied, or neither");

  /// Barrier subproblem parameters

  if (params.mu0 <= 0)
    throw std::invalid_argument(
        "Initial barrier parameter must be a positive real value");

  if ((params.mu_theta <= 0) || (params.mu_theta >= 1))
    throw std::invalid_argument("Multiplicative update for the barrier "
                                "parameter must be a real value "
                                "in the range (0, 1).");

  if (params.epsilon_mu0 <= 0)
    throw std::invalid_argument("Initial barrier subproblem KKT residual "
                                "tolerance must be a positive real value");

  if ((params.epsilon_mu_theta <= 0) || (params.epsilon_mu_theta >= 1))
    throw std::invalid_argument(
        "Multiplicative update for the barrier subproblem KKT residual "
        "tolerance must be a real value in the range (0, 1).");

  if ((params.rho <= 0) || (params.rho >= 1))
    throw std::invalid_argument(
        "Required ratio of total reduction in merit function due to "
        "improvement in linearized feasibility (rho) must be a real value in "
        "the range (0, 1)");

  if (params.nu0 <= 0)
    throw std::invalid_argument(
        "Initial merit function penalty parameter nu0 must "
        "be a positive real value");

  /// Declare and initialize some useful variables

  // Allocate output struct
  TRSQPResult<Vector, EqVector, IneqVector, Scalar> result;
  result.status = TRSQPStatus::IterationLimit;

  // Pair z = (x, s) consising of primal decision variable x and
  // auxiliary slack variables s (if any)
  Pair<Vector, IneqVector> z;

  // Mutable references to the elements of z (for convenience)
  Vector &x = z.first;      // Primal decision variables
  IneqVector &s = z.second; // Auxiliary slack variables

  // Initialize value of primal decision variables
  x = x0;

  // Value of objective function at current iterate x
  Scalar fx = f(x, args...);

  // Gradient of objective function at current iterate x
  Vector gradfx = gradf(x, args...);

  // Pair of vectors c(x) = (ce(x), ci(x)) containing the values of the
  // equality and inequality constraints, respectively, as the current iterate
  // x
  Pair<EqVector, IneqVector> cx = c(x, args...);

  // Set of auxiliary slack variables for the inequality constraints (if any)
  if (dim(cx.second) > 0) {
    // Initialize slack variables, ensuring that they are sufficiently
    // positive
    s = max(-cx.second, params.s0_min);
  }

  // Record problem dimensions
  size_t n = dim(x);          // Number of primal variables
  size_t me = dim(cx.first);  // Number of equality constraints
  size_t mi = dim(cx.second); // Number of inequality constraints
  size_t m = me + mi;         // Total number of constraints

  // Pair of Jacobians A(x) = (Ae(x), Ai(x)) of the  equality and inequality
  // constraints, respectively, at the current iterate x
  Pair<EqJacobian, IneqJacobian> Ax = A(x, args...);

  // Barrier parameter -- this value will be dynamically adjusted according to
  // Strategy 3 of the paper "On the Local Behavior of an Interior Point
  // Method for Nonlinear Programming" in order to ensure a suplinear
  // convergence rate for the overall algorithm
  Scalar mu = (mi > 0 ? params.mu0 : 0);

  // Initialize pair of Lagrange multipliers lambda = (lambda_e, lambda_i) for
  // the equality and inequality constraints, respectively
  Pair<EqVector, IneqVector> lambda =
      update_Lagrange_multipliers<Vector, EqVector, IneqVector, EqJacobian,
                                  IneqJacobian, Scalar, Args...>(
          gradfx, s, 0 * cx, Ax, mu, augmented_linear_system_solver, args...);

  // Hessian of Lagrangian wrt x
  Hessian HxL = Hess(x, lambda, args...);

  // Primal-dual approximation of Hessian of Lagrangian for barrier
  // subproblem with respect to auxiliary slack variables s
  IneqVector Sigma = compute_Sigma(s, lambda.second, mu);

  // Initial trust-region radius
  Scalar Delta = params.Delta0;

  // Set initial value of merit functin penalty parameter
  Scalar nu = params.nu0;

  /// UPDATE STEPS

  // Normal component for the Byrd-Omojokun composite update step
  Pair<Vector, IneqVector> v;

  // Tangential component for the Byrd-Omojokun composite update step
  Pair<Vector, IneqVector> w;

  // Number of STPCG iterations used to compute the tangential update step
  size_t num_STPCG_iters;

  // Number of line searches used to compute the primal-dual tangential update
  // step
  int num_pd_linesearch_iters;

  // Final update step used in the iteration (either Byrd-Omojokun composite
  // or primal-dual), including second-order correction (if one is applied)
  Pair<Vector, IneqVector> d;

  // Update for Lagrange multipliers
  Pair<EqVector, IneqVector> delta_lambda;

  // Boolean value indicating whether a second-order correction was applied
  // to the trial step used in the current iteration
  bool SOC_applied = false;

  // Enumeration type indicating what type of step was computed at each
  // iteration (Byrd-Omojokun composite or primal-dual)
  TRSQPStepType step_type;

  // Trial update point
  Pair<Vector, IneqVector> zplus;
  Vector &xplus = zplus.first;
  IneqVector &splus = zplus.second;

  // Value of objective at trial point
  Scalar f_xplus;

  // Value of constraints at trial point
  Pair<EqVector, IneqVector> c_xplus;

  // Gain ratio for the proposed update step
  Scalar rho;

  // Boolean value indicating whether the computed update step was accepted
  bool step_accepted = true;

  // Boolean values indicating whether any cached factorizations of the linear
  // systems required by the trust-region and primal-dual methods are current
  bool TR_system_is_current = true; // From computing initial lambdas
  bool PD_system_is_current = false;

  // Updated trust-region radius
  Scalar Delta_plus;

  /// KKT residuals

  // Gradient of Lagrangian
  //
  // L := f(x) + <ce(x), lambda_e> + <ci(x), lambda_i>
  //
  // with respect to x, evaluate at current primal-dual iterates (x, lambda)
  Vector gradLx = compute_gradLx(gradfx, lambda, Ax);
  Scalar gradLx_norm = norm(gradLx);

  // Norm of feasibility violation
  Scalar infeas_norm = compute_infeasibility(cx);

  // L2 norm of complementarity residual
  Scalar KKT_complementarity_error =
      compute_barrier_subproblem_complementarity_error<IneqVector, Scalar>(
          cx.second, lambda.second, 0.0);

  /// Subproblem tolerances

  // KKT residual for the barrier subproblem -- note that this is only used to
  // determine when to adjust the barrier parameter mu, so this computation
  // can be skipped if only equality constraints are present
  Scalar barrier_KKT_residual =
      (mi > 0 ? compute_barrier_subproblem_KKT_residual(gradLx, cx, s,
                                                        lambda.second, mu)
              : 0);

  // KKT residual tolerance for barrier subproblem
  Scalar epsilon_mu = params.epsilon_mu0;

  /// Display settings

  // Field with for displaying outer iterations
  size_t iter_field_width = floor(log10(params.max_iterations)) + 1;

  // Field width for displaying inner iterations (STPCG or line-search
  // iterations)
  size_t inner_iter_field_width =
      floor(log10(std::max(params.max_STPCG_iterations, params.max_ls_iters))) +
      1;

  if (params.verbose) {
    // Set display options for real-valued quantities
    std::cout << std::scientific;
    std::cout.precision(params.precision);
    std::cout << "Trust-region SQP optimization: " << std::endl << std::endl;
  }

  /// START OPTIMIZATION
  auto start_time = Stopwatch::tick();
  for (unsigned int k = 0; k < params.max_iterations; ++k) {
    // Get the total elapsed computation time at the *start* of this iteration
    double elapsed_time = Stopwatch::tock(start_time);

    /// CHECK TERMINATION CRITERIA

    // Check whether we have exhausted the allotted computation time
    if (elapsed_time > params.max_computation_time) {
      result.status = TRSQPStatus::ElapsedTime;
      break;
    }

    /// Too-small trust-region radius
    if (Delta < params.DeltaMin) {
      result.status = TRSQPStatus::TrustRegion;
      break;
    }

    /// KKT residuals for nonlinear program

    // Check whether the current iterates (x, lambda) satisfy the
    // KKT stopping tolerances
    if ((gradLx_norm <= params.gradient_tolerance) &&
        (infeas_norm <= params.infeasibility_tolerance) &&
        (KKT_complementarity_error <= params.complementarity_tolerance)) {
      // We've found a KKT point for the original NLP!
      result.status = TRSQPStatus::KKT;
      break;
    }

    /// Infeasibility

    // Check whether the primal iterate x is an infeasible stationary point of
    // the infeasibility measure v(x) := |ce(x)|^2 + | [ci(x)]_{+} |^2
    Scalar constraint_gradient_norm = norm(
        compute_infeasibility_gradient<Vector, EqVector, IneqVector, EqJacobian,
                                       IneqJacobian, Scalar>(cx, Ax));

    // Check for infeasible stationarity
    if ((infeas_norm > params.infeasibility_tolerance) &&
        (constraint_gradient_norm <= params.gradient_tolerance)) {
      // This is an infeasible point that is a stationary point of the
      // infeasibility measure
      result.status = TRSQPStatus::InfeasibleStationary;
      break;
    }

    /// KKT residuals for barrier subproblem
    if ((mi > 0) && (barrier_KKT_residual < epsilon_mu)) {
      // Update the barrier parameter mu and subproblem KKT residual tolerance
      // epsilon_mu using the simple multiplicative update rule proposed in
      // Algorithm III of the paper "An Interior Algorithm for Large-Scale
      // Nonlinear Programming"
      mu *= params.mu_theta;
      epsilon_mu *= params.epsilon_mu_theta;

      // Reset penalty parameter for new barrier subproblem
      nu = params.nu0;

      // Reset trust-region
      Delta = std::max(params.alpha2 * Delta, params.Delta0);
    }

    /// Display initial output to user about this iteration
    if (params.verbose) {
      std::cout << "k: ";
      std::cout.width(iter_field_width);
      std::cout << k << ", t: " << elapsed_time << ", f: ";
      std::cout.width(params.precision + 7);
      std::cout << fx << ", |c|: " << infeas_norm
                << ", |gradLx|: " << gradLx_norm
                << ", comp: " << KKT_complementarity_error << ", mu: " << mu
                << ", eps: " << epsilon_mu << ", Delta: " << Delta;
    }

    /// COMPUTE UPDATE STEP

    // Boolean value indicating whether a primal-dual update step should be
    // attempted
    bool attempt_primal_dual_step =
        (primal_dual_strategy &&
         (*primal_dual_strategy)(k, elapsed_time, x, s, lambda, fx, gradfx, HxL,
                                 Sigma, cx, Ax, mu, Delta, step_type,
                                 num_STPCG_iters, d, step_accepted, args...));

    // Reset status flags
    num_STPCG_iters = 0;
    num_pd_linesearch_iters = 0;
    SOC_applied = false;
    step_accepted = false;

    /// PRIMAL-DUAL STEP COMPUTATION

    if (attempt_primal_dual_step) {

      // Dual components of normal and tangential updates
      Pair<EqVector, IneqVector> v_lambda, w_lambda;

      compute_primal_dual_update_components<Vector, EqVector, IneqVector,
                                            EqJacobian, IneqJacobian, Hessian,
                                            Scalar, Args...>(
          cx, s, gradfx, Ax, lambda, HxL, Sigma, mu, !PD_system_is_current,
          *kkt_system_solver, v, v_lambda, w, w_lambda, args...);

      PD_system_is_current = true;

      // Assertion checking: verify that the returned normal and tangential step
      // components satisfy their defining equations

      // Normal step: W*v_z + Abar'*v_lambda = 0
      assert((compute_barrier_subproblem_Hessian_of_Lagrangian_product(
                  HxL, Sigma, v) +
              compute_Abar_transpose_product<Vector, EqVector, IneqVector,
                                             EqJacobian, IneqJacobian>(
                  Ax, v_lambda))
                 .norm() <
             1e-6 * std::max(1.0, sqrt(std::pow(v.norm(), 2) +
                                       std::pow(v_lambda.norm(), 2))));

      // Normal step:   Abar*v_z + c(z) = 0
      assert((compute_Abar_product<Vector, EqVector, IneqVector, EqJacobian,
                                   IneqJacobian>(Ax, v) +
              compute_barrier_subproblem_constraint_residuals(cx, s))
                 .norm() <
             1e-6 *
                 std::max(1.0,
                          compute_barrier_subproblem_constraint_residuals(cx, s)
                              .norm()));

      // Tangential step:  W*w_z + Abar(lambda + w_lamba) + grad_varphi = 0
      assert(
          (compute_barrier_subproblem_Hessian_of_Lagrangian_product(HxL, Sigma,
                                                                    w) +
           compute_Abar_transpose_product<Vector, EqVector, IneqVector,
                                          EqJacobian, IneqJacobian>(
               Ax, lambda + w_lambda) +
           compute_barrier_subproblem_objective_gradient(gradfx, s, mu))
              .norm() <
          1e-6 *
              std::max(
                  1.0,
                  (compute_barrier_subproblem_objective_gradient(gradfx, s,
                                                                 mu) +
                   compute_Abar_transpose_product<Vector, EqVector, IneqVector,
                                                  EqJacobian, IneqJacobian>(
                       Ax, lambda))
                      .norm()));

      // Tangential step:  A*w_z = 0
      assert((compute_Abar_product<Vector, EqVector, IneqVector, EqJacobian,
                                   IneqJacobian>(Ax, w)
                  .norm()) < 1e-6 * std::max(1.0, w.norm()));

      // Check whether the primal tangential update w satisfies the
      // following two (necessary and sufficient) conditions for providing a
      // valid backtracking linesearch direction of descent:
      //
      //(1) w is a direction of descent for the barrier subproblem
      //(2) w is a direction of positive curvature for the Hessian of the
      //    Lagrangian of the barrier subproblem

      Pair<Vector, IneqVector> grad_varphi =
          compute_barrier_subproblem_objective_gradient(gradfx, s, mu);
      Scalar grad_varphi_w = grad_varphi.dot(w);

      Scalar wtWw =
          w.dot(compute_barrier_subproblem_Hessian_of_Lagrangian_product(
              HxL, Sigma, w));

      if ((grad_varphi_w <= 0) && (wtWw >= 0)) {
        // The tangential update step w is a valid descent direction for the
        // local quadratic model of barrier objective, and therefore the
        // composite primal-dual update steps d = v + w and
        // delta_lambda = v_lambda + w_lambda are valid directions for a
        // backtracking line search
        Pair<Vector, IneqVector> dz = v + w;
        Pair<EqVector, IneqVector> dlambda = v_lambda + w_lambda;

        // Compute the maximum admissible steplength satisfying the
        // fraction-to-boundary rule for the auxiliary slack variables
        Scalar alpha =
            compute_maximum_admissible_steplength(s, dz.second, params.tau);

        // Compute the inner product of the gradient grad_varphi(z) of the
        // barrier objective with the primal update step dz
        Scalar grad_varphi_dz = grad_varphi.dot(dz);

        // Compute barrier residual norm
        Scalar cz_norm =
            compute_barrier_subproblem_constraint_residuals(cx, s).norm();

        // Compute value of quadratic term dzWdz
        Pair<Vector, IneqVector> Wdz =
            compute_barrier_subproblem_Hessian_of_Lagrangian_product(HxL, Sigma,
                                                                     dz);

        /// STEP 1:  COMPUTE PENALTY UPDATE

        if (cz_norm >= std::max(params.infeasibility_tolerance,
                                sqrt(std::numeric_limits<Scalar>::epsilon()))) {
          // Compute numerator for penalty update (cf. eq. (3.5) in the paper
          // "An Interior Algorithm for Nonlinear Optimization that Combines
          // Line-Search and Trust-Region Steps")
          Scalar nu_trial_numerator = grad_varphi_dz;
          Scalar dztWdz = dz.dot(Wdz);
          if (dztWdz > 0)
            nu_trial_numerator += dztWdz / 2;

          // Denominator of trial update (cf. eq. (3.5))
          // cznorm := |c(x,s)| = |ce(x)    |
          //                      |ci(x) + s|
          //
          // the norm of the residuals of the constraints for the barrier
          // subproblem

          Scalar nu_trial_denominator = (1 - params.rho) * cz_norm;

          // Compute updated penalty (eq. 3.3)
          nu = std::max(nu, nu_trial_numerator / nu_trial_denominator + 1);
        } // Penalty update

        /// STEP 2: BACKTRACKING LINESEARCH

        // Compute directional derivative of merit function along the
        // direction of the primal update step (cf. eq. (3.7))
        Scalar Dphi = grad_varphi_dz - nu * cz_norm;

        // Verify that the directional derivative Dphi satisfies eq. (3.8)
        assert(Dphi <= -params.rho * nu * cz_norm);

        // Evaluate merit function at current iterate z
        Scalar phi_z = evaluate_merit_function(fx, cx, s, mu, nu);

        /// BACKTRACKING ARMIJO LINESEARCH
        alpha *= 2;
        num_pd_linesearch_iters = -1; // Number of linesearches performed

        do {
          ++num_pd_linesearch_iters;
          alpha /= 2;

          // Compute trial primal and dual update steps
          d = alpha * dz;
          delta_lambda = alpha * dlambda;

          // Compute trial point
          zplus = z + d;

          // Evaluate objective and constraints at new trial point
          f_xplus = f(xplus, args...);
          c_xplus = c(xplus, args...);

          // Perform slack reset (cf. eq. (3.13) of the paper "An Interior
          // Point Algorithm for Nonlinear Optimization That Combines Line
          // Search and Trust Region Steps")
          if (mi > 0) {
            splus = reset_slacks(splus, c_xplus.second);
          }

          // Evaluate merit function at trial iterate zplus
          Scalar phi_zplus =
              evaluate_merit_function(f_xplus, c_xplus, splus, mu, nu);

          //  Test Armijo step acceptance criterion (cf. eq. (3.9), although
          //  note that this equation has a typo in the paper)
          rho = (phi_zplus - phi_z) / (alpha * Dphi);
          step_accepted = (rho > params.ls_alpha);

          // Special case: if this is the first iteration of linesearch (full
          // steplength), try computing a second-order correction
          if (!step_accepted && (num_pd_linesearch_iters == 0)) {

            /// Special case: if the full-length step does not satisfy the
            /// Armijo sufficient decrease condition, try computing a
            /// second-order correction
            SOC_applied = true;

            // Vectors to hold second-order correction steps
            Pair<Vector, IneqVector> dz_soc;
            Pair<EqVector, IneqVector> dlambda_soc;

            // Compute predicted gradient of Lagrangian at trial point
            // (d + dz, lambda + delta_lambda)
            Pair<Vector, IneqVector> gradLz_plus =
                grad_varphi + alpha * Wdz +
                compute_Abar_transpose_product<Vector, EqVector, IneqVector,
                                               EqJacobian, IneqJacobian>(
                    Ax, lambda + delta_lambda);

            /// Compute second-order correction (cf. eq. (3.10) in the paper)

            (*kkt_system_solver)(
                HxL, Sigma, Ax, false, -gradLz_plus,
                -compute_barrier_subproblem_constraint_residuals(c_xplus,
                                                                 splus),
                dz_soc, dlambda_soc, args...);

            // Compute second-order-corrected update step (cf. eq. (3.11))
            d += dz_soc;
            delta_lambda += dlambda_soc;

            // Compute maximum admissible steplengths for primal and dual
            // second-order-corrected steps (cf. eq. (3.12))
            Scalar gamma_z_soc =
                compute_maximum_admissible_steplength(s, d.second, params.tau);
            Scalar gamma_lambda_soc = compute_maximum_admissible_steplength(
                lambda.second, delta_lambda.second, params.tau);

            // Compute scaled second-order primal and dual updates
            d *= gamma_z_soc;
            delta_lambda *= gamma_lambda_soc;

            // Compute second-order-corrected trial point
            zplus = z + d;

            // Evaluate objective and constraints at new trial point
            f_xplus = f(xplus, args...);
            c_xplus = c(xplus, args...);

            // Perform slack reset (cf. eq. (3.13) of the paper "An Interior
            // Point Algorithm for Nonlinear Optimization That Combines Line
            // Search and Trust Region Steps")
            if (mi > 0) {
              splus = reset_slacks(splus, c_xplus.second);
            }

            // Evaluate merit function at trial iterate zplus
            phi_zplus =
                evaluate_merit_function(f_xplus, c_xplus, splus, mu, nu);

            // Test acceptance at second-order corrected trial point
            rho = (phi_zplus - phi_z) / (alpha * Dphi);
            step_accepted = (rho > params.ls_alpha);

          } // Second-order correction

        } while ((!step_accepted) &&
                 (num_pd_linesearch_iters < params.max_ls_iters) &&
                 (params.ls_alpha * alpha >= params.alpha_min));

        if (step_accepted) {
          // Set step type
          step_type = TRSQPStepType::PrimalDual;

          // Update Lagrange multipliers
          lambda += delta_lambda;

          // Update trust-region step
          Delta_plus = std::max(Delta, 2 * compute_trust_region_norm(d, s));

        } // Step accepted
      }   // Primal-dual step is valid
    }     // Primal-dual step computation

    /// BYRD-OMOJOKUN COMPOSITE UPDATE STEP COMPUTATION
    if (!step_accepted) {

      // Reset second-order correction flag
      SOC_applied = false;

      // We are falling back to composite step type
      step_type = TRSQPStepType::TrustRegion;

      if (!TR_system_is_current) {
        // Compute least-squares multiplier updates
        delta_lambda = update_Lagrange_multipliers<Vector, EqVector, IneqVector,
                                                   EqJacobian, IneqJacobian,
                                                   Scalar, Args...>(
            gradfx, s, lambda, Ax, mu, augmented_linear_system_solver, args...);

        lambda += delta_lambda;
        TR_system_is_current = true;

        // Check implementation of user-supplied compute_multipliers()
        // function in DEBUG mode. The least-squares Lagrange multipliers
        // should be minimizers of
        //
        // l(lambda) = |Ahat'*lambda + (gradfx, -mu*e)|^2
        //
        // and should therefore satisfy:
        //
        // Ahat*Ahat'*lambda + Ahat*(gradfx, -mu*e) = 0
        assert(compute_gradient_of_multiplier_loss(gradfx, Ax, lambda, s, mu)
                   .norm() <= 1e-6);

        // Compute updated Hessians using the updated Lagrange multiplier
        // estimates
        HxL = Hess(x, lambda, args...);
        Sigma = compute_Sigma(s, lambda.second, mu);

        // Since we have just changed the Hessian models HxL and Sigma
        PD_system_is_current = false;
      }

      /// Compute normal update step
      v = compute_normal_step<Vector, EqVector, IneqVector, EqJacobian,
                              IneqJacobian, Scalar, Args...>(
          cx, s, Ax, Delta, params.zeta, params.tau,
          augmented_linear_system_solver, args...);

      // Check implementation of user-supplied compute_normal_step()
      // function in DEBUG mode.

      // Verify that the normal update step reduces the linearized model
      // of the constraints
      assert(vpred(cx, s, Ax, v) >= -1e-8);

      // Verify that the normal update step satisfies the trust-region
      // bound (3.19)
      assert(compute_trust_region_norm(v, s) <=
             (1 + 1e-8) * params.zeta * Delta);

      // Verify that the normal update step satisfies the
      // fraction-to-the-boundary rule (3.19), if inequality constraints
      // are present
      assert((mi > 0 ? (min(v.second + (params.tau / 2) * s) >= -1e-8) : true));

      /// Compute tangential update step
      Scalar tangential_step_M_norm;

      w = compute_tangential_step<Vector, EqVector, IneqVector, EqJacobian,
                                  IneqJacobian, Hessian, Scalar, Args...>(
          gradfx, s, Ax, HxL, Sigma, v, mu, Delta, params.tau,
          params.max_STPCG_iterations, params.cg_kappa_fgr, params.cg_theta,
          augmented_linear_system_solver, tangential_step_M_norm,
          num_STPCG_iters, args...);

      // Check implementation of user-supplied compute_tangential_step()
      // function in DEBUG mode.

      // Verify that the tangential update step reduces the quadratic
      // objective of the trust-region subproblem
      assert(evaluate_quadratic_model(v + w, gradfx, s, HxL, Sigma, mu) <=
             evaluate_quadratic_model(v, gradfx, s, HxL, Sigma, mu));

      // Verify that the trust-region update step lies in the kernel of
      // the constraint Jacobian (cf. eqs. (3.25)--(3.26)

      assert((compute_Abar_product<Vector, EqVector, IneqVector, EqJacobian,
                                   IneqJacobian>(Ax, w)
                  .norm()) < 1e-8 * std::max(w.norm(), 1.0));

      // Verify that the tangential update step satisfies the
      // fraction-to-the-boundary rule (3.29), if inequality constraints
      // are present
      assert((mi > 0 ? (min(v.second + w.second + params.tau * s) >= -1e-8)
                     : true));

      /// Compute composite update step
      d = v + w;

      /// PENALTY UPDATE

      // Compute reduction in linearized constraint violation on the
      // normal update step (note that since the tangential step w in
      // ker(A), this is equivalent to evaluating with the composite step
      // as well)
      Scalar vpredk = vpred(cx, s, Ax, v); // cf. eq. (3.50) in the paper

      // Evaluate quadratic model of objective on the full (composite)
      // step
      Scalar q = evaluate_quadratic_model(d, gradfx, s, HxL, Sigma, mu);

      // Update the penalty parameter according to the "Penalty Parameter
      // Procedure" described in the paper "An Interior Point Algorithm for
      // Large-Scale Nonlinear Programming"
      if (vpredk >= sqrt(std::numeric_limits<Scalar>::epsilon())) {

        // Update penalty parameter (cf. eq. (3.52))
        nu = std::max(nu, q / ((1 - params.rho) * vpredk));
      }

      /// CONSTRUCT TRIAL POINT AND EVALUATE GAIN RATIO

      // Compute updated trial points
      zplus = z + d;

      // Evaluate objective and constraints at new trial point
      f_xplus = f(xplus, args...);
      c_xplus = c(xplus, args...);

      // Perform slack reset (cf. eq. (3.13) of the paper "An Interior Point
      // Algorithm for Nonlinear Optimization That Combines Line Search and
      // Trust Region Steps")
      if (mi > 0)
        splus = reset_slacks(splus, c_xplus.second);

      // Compute predicted reduction in merit function from linearized
      // model
      Scalar predk = -q + nu * vpredk; // cf. eq. (3.49) in the paper

      // Compute actual reduction in merit function (cf. eq. (3.53) in the
      // paper)
      Scalar aredk = evaluate_merit_function(fx, cx, s, mu, nu) -
                     evaluate_merit_function(f_xplus, c_xplus, splus, mu, nu);

      // Compute gain ratio for this trial step
      rho = aredk / predk;

      /// TEST ACCEPTANCE
      step_accepted = (rho >= params.eta1);
      Delta_plus = update_trust_region(Delta, rho, params.eta1, params.eta2,
                                       params.alpha1, params.alpha2,
                                       compute_trust_region_norm(d, s));

      // If gamma < eta1, the step will be rejected.  However, if the
      // normal component of the step is small compared to the tangential
      // component, the poor performance of this step may be due to the
      // curvature of the constraints, which is not captured in the
      // linearized model of the constraints used in the SQP subproblem
      // (Maratos effect).  Therefore, we can attempt to "fix up" the step
      // by computing a second order correction
      if (!step_accepted && (compute_trust_region_norm(v, s) <
                             .1 * compute_trust_region_norm(w, s))) {

        /// SECOND-ORDER CORRECTION

        Pair<Vector, IneqVector> y =
            compute_normal_step<Vector, EqVector, IneqVector, EqJacobian,
                                IneqJacobian, Scalar, Args...>(
                c_xplus, splus, Ax, std::numeric_limits<Scalar>::max(), 1.0,
                1.0, augmented_linear_system_solver, args...);

        // Apply second-order correction to composite update step d and
        // trial point zplus
        d += y;
        zplus += y;

        SOC_applied = true;

        // If the second-order-corrected step still satisfies the
        // fraction-to-the-boundary condition for the slack variables
        // (if any) ...
        if ((mi == 0) || min(zplus.second - (1 - params.tau) * s) >= 0) {

          // Re-evaluate the actual reduction in the merit function using
          // the updated trial point
          f_xplus = f(xplus, args...);
          c_xplus = c(xplus, args...);

          // Perform slack reset (cf. eq. (3.13) of the paper "An Interior
          // Point Algorithm for Nonlinear Optimization That Combines Line
          // Search and Trust Region Steps")
          if (mi > 0) {
            splus = reset_slacks(splus, c_xplus.second);
          }

          Scalar aredk_soc =
              evaluate_merit_function(fx, cx, s, mu, nu) -
              evaluate_merit_function(f_xplus, c_xplus, splus, mu, nu);

          // Compute gain ratio for this trial step
          Scalar rho_soc = aredk_soc / predk;

          if (rho_soc >= params.eta1) {
            // Accept the second-order-corrected update, but DO NOT modify
            // trust-region radius
            step_accepted = true;
            Delta_plus = Delta;
            rho = rho_soc;
          }
        }
      } // second-order correction
    }   // Byrd-Omojokun composite step computation

    // Compute norm of primal updates step dx
    Scalar dx_norm = norm(d.first);

    /// Display output to user, if requested

    if (params.verbose) {
      std::cout << ", "
                << (step_type == TRSQPStepType::TrustRegion ? "CG " : "LS ")
                << "iters: ";
      std::cout.width(inner_iter_field_width);
      std::cout << (step_type == TRSQPStepType::TrustRegion
                        ? num_STPCG_iters
                        : num_pd_linesearch_iters);
      std::cout << ", |dx|: " << dx_norm << ", rho: ";
      std::cout.width(params.precision + 7);
      std::cout << rho << ".  ";
      std::cout << (step_accepted ? "Accepted" : "Rejected") << std::endl;
    }

    /// Record info from this iteration
    result.time.push_back(elapsed_time);
    result.objective_values.push_back(fx);
    result.infeas_norms.push_back(infeas_norm);
    result.grad_Lx_norms.push_back(gradLx_norm);
    result.complementarity_norms.push_back(KKT_complementarity_error);
    result.barrier_params.push_back(mu);
    result.trust_region_radius.push_back(Delta);
    result.STPCG_iters.push_back(num_STPCG_iters);
    result.linesearch_iters.push_back(num_pd_linesearch_iters);
    result.dx_norms.push_back(dx_norm);
    result.penalty_params.push_back(nu);
    result.step_types.push_back(step_type);
    result.SOCs.push_back(SOC_applied);
    result.gain_ratios.push_back(rho);
    if (params.log_iterates)
      result.iterates.push_back(x);

    /// Call user-supplied function, if one was passed
    if (user_function) {
      // Call the user-supplied function
      bool user_requested_termination = (*user_function)(
          k, elapsed_time, x, s, lambda, fx, gradfx, HxL, Sigma, cx, Ax, mu,
          Delta, step_type, num_STPCG_iters, v, w, d, nu, rho, SOC_applied,
          step_accepted, args...);

      if (user_requested_termination) {
        result.status = TRSQPStatus::UserFunction;
        break;
      }
    }

    /// Update cached values and prepare for next iteration
    Delta = Delta_plus;
    if (step_accepted) {
      // Update local quadratic model of objective and constraints

      // NB:  We use the *current* value of mu to compute the Lagrange
      // multipliers lambda and primal-dual approximation of the Hessian
      // Sigma with respect to s for the next iteration, *even if we
      // update the barrier parameter mu for the next iteration*, in order
      // to prevent the Lagrange multipliers and Hessian from changing too
      // abruptly when the barrier parameter is updated, in accordance
      // with the strategy described in Sec. 3.1 of the paper "An Interior
      // Point Algorithm for Large-Scale Nonlinear Programming" (cf. eqs.
      // (3.14)--(3.15))
      z = zplus;
      fx = f_xplus;
      cx = c_xplus;
      gradfx = gradf(x, args...);
      Ax = A(x, args...);

      TR_system_is_current = false;
      PD_system_is_current = false;

      // If the accepted step was a trust-region step, compute updated
      // Lagrange multipliers at the new iterate
      if (step_type == TRSQPStepType::TrustRegion) {
        lambda += update_Lagrange_multipliers<Vector, EqVector, IneqVector,
                                              EqJacobian, IneqJacobian, Scalar,
                                              Args...>(
            gradfx, s, lambda, Ax, mu, augmented_linear_system_solver, args...);

        // Check implementation of user-supplied compute_multipliers()
        // function in DEBUG mode. The least-squares Lagrange multipliers
        // should be minimizers of
        //
        // F(lambda) = |Ahat'*lambda + (gradfx, -mu*e)|^2
        //
        // and should therefore satisfy:
        //
        // Ahat*Ahat'*lambda + Ahat*(gradfx, -mu*e) = 0
        assert(compute_gradient_of_multiplier_loss(gradfx, Ax, lambda, s, mu)
                   .norm() <= 1e-6);

        TR_system_is_current = true;
      }

      // Update gradLx
      gradLx = compute_gradLx(gradfx, lambda, Ax);

      // Update Hessians
      HxL = Hess(x, lambda, args...);
      Sigma = compute_Sigma(s, lambda.second, mu);

      // Update KKT residuals at new trial point
      gradLx_norm = norm(gradLx);
      infeas_norm = compute_infeasibility(cx);
      KKT_complementarity_error =
          compute_barrier_subproblem_complementarity_error<IneqVector, Scalar>(
              cx.second, lambda.second, 0.0);

      // Update barrier subproblem KKT residuals
      barrier_KKT_residual = (mi > 0 ? compute_barrier_subproblem_KKT_residual(
                                           gradLx, cx, s, lambda.second, mu)
                                     : 0);
    }
  } // Iterations:  for (unsigned int k = 0; k < params.max_iterations; ++k) ...

  /// Record final outputs
  result.elapsed_time = Stopwatch::tock(start_time);
  result.x = x;
  result.lambda = lambda;
  result.f = fx;
  result.infeas_norm = infeas_norm;
  result.grad_Lx_norm = gradLx_norm;
  result.complementarity_norm = KKT_complementarity_error;

  /// FINAL OUTPUT

  if (params.verbose) {
    std::cout << std::endl
              << std::endl
              << "Optimization finished!" << std::endl;

    // Print the reason for termination
    switch (result.status) {
    case TRSQPStatus::KKT:
      std::cout << "Found first-order KKT point!" << std::endl;
      break;

    case TRSQPStatus::TrustRegion:
      std::cout
          << "Algorithm terminated because the trust-region radius decreased "
             "below the minimum admissible value ("
          << Delta << " < " << params.DeltaMin << ")" << std::endl;
      break;

    case TRSQPStatus::InfeasibleStationary:
      std::cout << "Optimization converged to infeasible stationary point"
                << std::endl;
      break;

    case TRSQPStatus::IterationLimit:
      std::cout << "Algorithm exceeded maximum number of outer iterations"
                << std::endl;
      break;
    case TRSQPStatus::ElapsedTime:
      std::cout << "Algorithm exceeded maximum allowed computation time: ("
                << result.elapsed_time << " > " << params.max_computation_time
                << " seconds)" << std::endl;
      break;
    case TRSQPStatus::UserFunction:
      std::cout << "Algorithm terminated due to user-supplied stopping "
                   "criterion"
                << std::endl;
      break;
    }

    std::cout << "Final objective value: " << result.f << std::endl;
    std::cout << "Infeasibility norm: " << result.infeas_norm << std::endl;
    std::cout << "Norm of gradient of Lagrangian: " << result.grad_Lx_norm
              << std::endl;
    std::cout << "Complementarity residual norm: "
              << result.complementarity_norm << std::endl;
    std::cout << "Total elapsed computation time: " << result.elapsed_time
              << " seconds" << std::endl
              << std::endl;

    // Reset std::cout output stream to default display parameters
    std::cout << std::defaultfloat;
    std::cout.precision(6);
  }

  return result;
}

} // namespace Constrained
} // namespace Optimization
