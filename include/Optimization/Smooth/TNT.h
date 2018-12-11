/** This header file provides a lightweight template function implementing
 * a truncated-Newton trust-region algorithm for large-scale superlinear
 * optimization on Riemannian manifolds.
 *
 * For more information on trust-region algorithms, please see the excellent
 * references:
 *
 * "Trust-Region Methods" by Conn, Gould, and Toint
 *
 * "Trust-Region Methods on Riemannian Manifolds" by Absil, Baker and Gallivan
 *
 * "Optimization Algorithms on Matrix Manifolds" by Absil, Mahoney, and
 *  Sepulchre
 *
 * Copyright (C) 2017 - 2018 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <experimental/optional>
#include <functional>
#include <iostream>
#include <limits>

#include "Optimization/Smooth/Concepts.h"
#include "Optimization/Util/Stopwatch.h"

namespace Optimization {

namespace Smooth {

/** An alias template for a user-definable function that can be
 * used to access various interesting bits of information about the internal
 * state of a truncated-Newton optimization algorithm as it runs.  More
 * precisely, this function is called at the end of each major (outer)
 * iteration, and is provided access to the following quantities:
 *
 * i: index of current iteration
 * t: total elapsed computation time at the *start* of the current iteration
 * x: iterate at the *start* of the current iteration
 * f: objective value at x
 * g: Riemannian gradient at x
 * HessOp:  Hessian operator at x
 * Delta: trust-region radius at the *start* of the current iteration
 * num_STPCG_iters: number of iterations of the Steihaug-Toint preconditioned
 *                  conjugate-gradient method performed when computing the
 *                  trust-update step h
 * h:  the trust-region update step computed during this iteration
 * df: decrease in objective value obtained by applying the trust-region update
 * rho: value of the gain ratio for the proposed update step h
 * accepted:  Boolean value indicating whether the proposed trust-region update
 *            step h was accepted
 */
template <typename Variable, typename Tangent, typename Scalar = double,
          typename... Args>
using TNTUserFunction =
    std::function<void(size_t i, double t, const Variable &x, Scalar f,
                       const Tangent &g,
                       const LinearOperator<Variable, Tangent, Args...> &HessOp,
                       Scalar Delta, size_t num_STPCG_iters, const Tangent &h,
                       Scalar df, Scalar rho, bool accepted, Args &... args)>;

/** This function implements the Steihaug-Toint truncated preconditioned
 * conjugate-gradient algorithm used to compute the update step in a
 * truncated-Newton trust-region optimization method.  This
 * specific implementation follows the pseudocode given as Algorithm 7.5.1 in
 * the reference "Trust-Region Methods" by Conn, Gould, and Toint.  Here:
 *
 * - X is the value of the optimization variable at which the linear model
 *   is constructed
 * - grad is the gradient vector computed for the model objective at X
 * - Hess is a linear operator that models the Hessian of the objective at X
 * - metric is the Riemannian metric at the current iterate X
 * - precon is a *positive-definite* preconditioning operator
 * - Delta is the trust-region radius around X, in the M-norm determined by
 *   the preconditioner; if the preconditioner acts as v -> M^{-1} v, then the
 *   trust-region is given by
 *
 *   || s ||_M <= Delta
 *
 * - kappa_fgr and theta are parameters that control the stopping criteria
 *   for the algorithm; termination occurs whenever the predicted gradient g_k
 *   at the iterate obtained by applying the current update step (as predicted
 *   by the local quadratic model) satisfies:
 *
 *   || g_k || <= ||g_0|| min [kappa_fgr, || g_0 ||^theta]
 *
 *   (cf. Algorithm 7.5.1 of "Trust-Region Methods").
 *
 * - update_step_M_norm is a return value that gives the norm || h ||_M of
 *   the update step in the M-norm determined by the preconditioner
 */

template <typename Variable, typename Tangent, typename Scalar = double,
          typename... Args>
Tangent
STPCG(const Variable &X, const Tangent &grad,
      const LinearOperator<Variable, Tangent, Args...> &Hess,
      const RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
      const std::experimental::optional<
          LinearOperator<Variable, Tangent, Args...>> &precon,
      Scalar &update_step_M_norm, size_t &num_iterations, Scalar Delta,
      Args &... args, size_t max_iterations = 1000, Scalar kappa_fgr = .1,
      Scalar theta = .5) {
  /// INITIALIZATION

  // The current estimate for the truncated-Newton update step s_k; initialized
  // to zero
  Tangent s_k = Tangent::Zero(grad.rows(), grad.cols());

  // The gradient at the current update step s_k
  Tangent g_k = grad;
  // The preconditioned value of g_k
  Tangent v_k = (precon ? (*precon)(X, g_k, args...) : g_k);
  // The current step direction p_k for updating s_k
  Tangent p_k = -v_k;

  // Value of inner product < s_k, M * p_k >; initially zero since s0 = 0
  Scalar sk_M_pk = 0;
  // Squared M-norm of current Newton step estimate s_k:  || s_k ||_M^2;
  // initially zero since s0 = 0
  Scalar sk_M_2 = 0;
  // Squared M-norm of current update step direction p_k: || p_k ||_M^2
  Scalar pk_M_2 = metric(X, g_k, v_k, args...);

  /// Useful cached variables
  // Squared radius of trust-region
  Scalar Delta_2 = Delta * Delta;
  // Norm of initial (unpreconditioned) gradient
  Scalar g0_norm = sqrt(metric(X, grad, grad, args...));
  // Target norm of (unpreconditioned) gradient after applying update step s_k
  Scalar target_grad_norm =
      g0_norm * std::min(kappa_fgr, std::pow(g0_norm, theta));

  Scalar alpha_k; // Scalar used to compute the full Newton step along direction
                  // p_k
  Scalar beta_k;
  Scalar kappa_k; // Value of inner product < p_k, H*p_k >

  num_iterations = 0;
  while (num_iterations < max_iterations) {
    /// CHECK TERMINATION CRITERIA

    /// "Standard" termination criteria based upon predicted gradient after
    /// applying the current update step s_k
    if (std::sqrt(metric(X, g_k, g_k, args...)) <= target_grad_norm) {
      update_step_M_norm = std::sqrt(sk_M_2);
      return s_k;
    }

    /// Next, check termination criteria based upon check for negative curvature
    /// or overly-long steplengths

    // Compute kappa
    kappa_k = metric(X, p_k, Hess(X, p_k, args...), args...);

    // Compute (full) steplength scalar alpha_k
    alpha_k = metric(X, g_k, v_k, args...) / kappa_k;

    // Compute norm of proposed (full) step
    Scalar skplus1_M_2 =
        sk_M_2 + 2 * alpha_k * sk_M_pk + alpha_k * alpha_k * pk_M_2;

    if ((kappa_k <= 0) || (skplus1_M_2 > Delta_2)) {
      /** Either p_k is a direction of negative curvature, or the full
       * (unrestricted) step along this direction leaves the trust-region.  In
       * either case, we would like to rescale the stepsize alpha_k to ensure
       * that the proposed step terminates _on_ the trust-region boundary, and
       * then return the resulting step as our final answer */

      Scalar sigma_k =
          (-sk_M_pk + sqrt(sk_M_pk * sk_M_pk + pk_M_2 * (Delta_2 - sk_M_2))) /
          pk_M_2;

      update_step_M_norm = Delta;

      return s_k + sigma_k * p_k;
    }

    /// UPDATE!  Compute values for next iteration

    // Update Newton step s_k -> s_kplus1
    s_k = s_k + alpha_k * p_k;
    // Update estimate for gradient after applying s_k:  g_k -> g_kplus1
    g_k = g_k + alpha_k * Hess(X, p_k, args...);
    // Update preconditioned gradient estimate: v_k -> v_kplus1
    v_k = (precon ? (*precon)(X, g_k, args...) : g_k);

    // Compute beta_k for *current* iterate (using *updated* values g_kplus1 and
    // v_kplus1 and *current* iterate
    // value alpha_k, kappa_k)
    beta_k = metric(X, g_k, v_k, args...) / (alpha_k * kappa_k);

    // Update inner products and norms
    sk_M_2 = skplus1_M_2;
    sk_M_pk = beta_k * (sk_M_pk + alpha_k * pk_M_2);
    pk_M_2 = metric(X, g_k, v_k, args...) + beta_k * beta_k * pk_M_2;

    // Update search direction p_k
    p_k = -v_k + beta_k * p_k;

    num_iterations++;
  } // while( ... )

  update_step_M_norm = sqrt(sk_M_2);
  return s_k;
}

/** A lightweight struct containing a few additional algorithm-specific
 * configuration parameters for a truncated-Newton trust-region method
 * (cf. Algorithm 6.1.1 of "Trust-Region Methods") */
template <typename Scalar = double>
struct TNTParams : public SmoothOptimizerParams<Scalar> {
  /// Trust-region control parameters

  /** Initial size of trust-region radius */
  Scalar Delta0 = 1;

  /** Lower-bound on the gain ratio for accepting a proposed step (should
satisfy 0 < eta1 <= eta2) */
  Scalar eta1 = .05;

  /** Lower-bound on the gain ratio for a 'very successful' iteration (should
   * satisfy eta1 <= eta2 < 1). */
  Scalar eta2 = .9;

  /** Multiplicative factor for decreasing the trust-region radius on an
   * unsuccessful iteration; this parameter should satisfy 0 < alpha1 < 1. */
  Scalar alpha1 = .25;

  /** Multiplicative factor for increasing the trust-region radius on a very
   * successful iteration.  This parameter should satisfy alpha2 > 1. */
  Scalar alpha2 = 2.5;

  /// Truncated preconditioned conjugate-gradient control parameters
  // See Section 7.5.1 of "Trust-Region Methods" for details

  /** Maximum number of conjugate-gradient iterations to apply to solve each
   * instance of the trust-region subproblem */
  size_t max_TPCG_iterations = 1000;

  /** Stopping tolerance for the norm of the predicted gradient (acccording to
   * the local quadratic model): terminate the inner iterations if ||g_pred||
   * kappa_fgr * || g_init ||.  This value should lie in the range (0, 1) */
  Scalar kappa_fgr = .1;

  /** Stopping tolerance for the norm of the predicted gradient (according to
   * the local quadratic model):  terminate the inner iterations if ||g_pred|| <
   * ||g_init||^{1+theta}.  This value should be positive, and controls the
   * asymptotic convergence rate of the TNT algorithm: specifically, for theta >
   * 0, TNT convergences q-superlinearly with order (1 + theta) (cf. Theorem 2.3
   * of "Truncated-Newton Algorithms for Large-Scale Unconstrained
   * Optimization", by R.S. Dembo and T. Steihaug.) */
  Scalar theta = .5;

  /// Additional termination criteria

  /** Stopping tolerance for the norm of the preconditioned gradient
   * || M^{-1} g ||.  Note that this stopping criterion has the advantage of
   * (approximate) scale invariance */
  Scalar preconditioned_gradient_tolerance = 1e-6;

  /** Stopping condition based on the size of the trust-region radius; terminate
   * if the radius is reduced below this quantity */
  Scalar Delta_tolerance = 1e-6;
};

/** A set of status flags indicating the stopping criterion that triggered
 * algorithm termination */
enum class TNTStatus {

  /** The algorithm obtained a solution satisfying the gradient tolerance */
  GRADIENT,

  /** The algorithm obtained a solution satisfying the preconditioned gradient
     tolerance */
  PRECONDITIONED_GRADIENT,

  /** The algorithm terminated because the relative decrease in function value
     obtained after the last accepted update was less than the specified
     tolerance */
  RELATIVE_DECREASE,

  /** The algorithm terminated because the norm of the last accepted update
     step
     was less than the specified tolerance */
  STEPSIZE,

  /** The algorithm terminated because the trust-region radius decreased below
     the specified threshold */
  TRUST_REGION,

  /** The algorithm exhausted the allotted number of major (outer) iterations
   */
  ITERATION_LIMIT,

  /** The algorithm exhausted the allotted computation time */
  ELAPSED_TIME
};

/** A useful struct used to hold the output of a truncated-Newton trust-region
optimization method */
template <typename Variable, typename Scalar = double>
struct TNTResult : public SmoothOptimizerResult<Variable, Scalar> {

  /** The norm of the preconditioned gradient at the returned estimate */
  Scalar preconditioned_grad_f_x_norm;

  /** The stopping condition that triggered algorithm termination */
  TNTStatus status;

  /** The preconditioned norm of the gradient at the START of each iteration
   */
  std::vector<Scalar> preconditioned_gradient_norms;

  /** The number of (inner) iterations performed by the Steihaug-Toint
   * preconditioned conjugate-gradient method during each major (outer)
   * iteration */
  std::vector<size_t> inner_iterations;

  /** The M-norm of the update step computed during each iteration */
  std::vector<Scalar> update_step_M_norm;

  /** The gain ratio of the update step computed during each iteration */
  std::vector<Scalar> rho;

  /** The trust-region radius at the START of each iteration */
  std::vector<Scalar> trust_region_radius;
};

/** This function implements a Riemannian truncated-Newton trust-region method,
 * using the Steihaug-Toint preconditioned truncated conjugate-gradient method
 * as the inner (approximate) trust-region subproblem solver. This
 * implementation follows the one given as Algorithm 6.1.1 of "Trust-Region
 * Methods", generalized to the setting of a generic Riemannian manifold (see
 * also the paper "Trust-Region Methods on Riemannian Manifolds" by P.-A. Absil,
 * C.G. Baker, and K.A. Gallivan, and/or "Optimization Algorithms on Matrix
 * Manifolds" by P.-A. Absil, R. Mahoney, and R. Sepulchre).
 *
 * Here:
 *
 * - f is the objective function to be minimized.
 *
 * - QM is a function that constructs a quadratic model for the objective f on
 *   the tangent space of the manifold M at X. Concretely, this is a
 *   std::function having the signature:
 *
 *   void QM(const Variable &X, Tangent &gradient, LinearOperator<Variable,
 *        Tangent, Args...> &Hessian, Args &... args)>;
 *
 *   this function sets the argument 'gradient' to the Riemannian gradient
 *   grad f(X) of f at X, and the argument 'Hessian' to the Riemanninan Hessian
 *   Hess f(X) of f at X (recall that this is a linear operator
 *   Hess f(X) : T_X(M) -> T_X(M) on the tangent space T_X(M) of M at X that
 *   maps each tangent vector V to the covariant derivative of the gradient
 *   vector field grad f(X) along V at X; cf. Secs. 3.6 and 5.5 of
 *   "Optimization Methods on Matrix Manifolds").
 *
 * - metric is the Riemannian metric for M (a smooth assignment of an inner
 *   product to each of M's tangent spaces, cf. Sec. 3.6 of "Optimization
 *   Methods on Matrix Manifolds").
 *
 * - retract is a retraction operator: this is a function taking as input a
 *   point X in M and a tangent vector V in the tangent space T_X(M) of M at X,
 *   and returns a new point Y in M that is (intuitively) gotten by 'moving
 *   along V from X' (cf. Sec. 4.1 of "Optimization Methods on Riemannian
 *   Manifolds" for a precise definition).
 *
 * - x0 (in M) is the initialization point for the Riemannian truncated-Newton
 *   trust-region algorithm.
 */
template <typename Variable, typename Tangent, typename Scalar = double,
          typename... Args>
TNTResult<Variable, Scalar>
TNT(const Objective<Variable, Scalar, Args...> &f,
    const QuadraticModel<Variable, Tangent, Args...> &QM,
    const RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
    const Retraction<Variable, Tangent, Args...> &retract, const Variable &x0,
    Args &... args,
    const std::experimental::optional<
        LinearOperator<Variable, Tangent, Args...>> &precon =
        std::experimental::nullopt,
    const TNTParams<Scalar> &params = TNTParams<Scalar>(),
    const std::experimental::optional<
        TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
        std::experimental::nullopt) {
  /// Declare and initialize some useful variables

  // Square root of machine precision for type Scalar
  Scalar sqrt_eps = sqrt(std::numeric_limits<Scalar>::epsilon());

  // Output struct
  TNTResult<Variable, Scalar> result;
  result.status =
      TNTStatus::ITERATION_LIMIT; // "Default" stopping condition (i.e. will
                                  // trigger if no other is)

  // Current iterate and proposed next iterates;
  Variable x, x_proposed;

  // Function value at the current iterate and proposed iterate
  Scalar f_x, f_x_proposed;

  // Gradient and preconditioned gradient at the current iterate
  Scalar grad_f_x_norm, preconditioned_grad_f_x_norm;

  // Trust-region radius
  Scalar Delta;

  // Update step norm in the Euclidean and M-norms
  Scalar h_norm, h_M_norm;

  // Relative decrease between subsequent accepted function iterations
  Scalar relative_decrease;

  // Gradient and Hessian operator at the current iterate
  Tangent grad;
  LinearOperator<Variable, Tangent, Args...> Hess;

  // Some useful constants for display purposes

  // Field with for displaying outer iterations
  size_t iter_field_width = floor(log10(params.max_iterations)) + 1;

  // Field width for displaying inner iterations
  size_t inner_iter_field_width = floor(log10(params.max_TPCG_iterations)) + 1;

  /// INITIALIZATION

  // Set initial iterate ...
  x = x0;
  // Function value ...
  f_x = f(x, args...);

  // And local quadratic model
  QM(x, grad, Hess, args...);

  grad_f_x_norm = sqrt(metric(x, grad, grad, args...));
  if (precon) {
    // Precondition gradient
    Tangent preconditioned_gradient = (*precon)(x, grad, args...);
    preconditioned_grad_f_x_norm = sqrt(
        metric(x, preconditioned_gradient, preconditioned_gradient, args...));
  } else {
    // We use the identity preconditioner, so the norms of the gradient and
    // preconditioned gradient are identical
    preconditioned_grad_f_x_norm = grad_f_x_norm;
  }

  // Initialize trust-region radius
  Delta = params.Delta0;

  if (params.verbose) {
    // Set display options for real-valued quantities
    std::cout << std::scientific;
    std::cout.precision(params.precision);
  }
  if (params.verbose) {
    std::cout << "Truncated-Newton trust-region optimization: " << std::endl
              << std::endl;
  }

  // Start clock
  auto start_time = Stopwatch::tick();

  /// ITERATIONS

  // Basic trust-region iteration -- see Algorithm 6.1.1 of "Trust-Region
  // Methods"
  for (size_t iteration = 0; iteration < params.max_iterations; iteration++) {
    double elapsed_time = Stopwatch::tock(start_time);

    if (elapsed_time > params.max_computation_time) {
      result.status = TNTStatus::ELAPSED_TIME;
      break;
    }

    // Record output
    result.time.push_back(elapsed_time);
    result.objective_values.push_back(f_x);
    result.gradient_norms.push_back(grad_f_x_norm);
    result.preconditioned_gradient_norms.push_back(
        preconditioned_grad_f_x_norm);
    result.trust_region_radius.push_back(Delta);

    if (params.log_iterates)
      result.iterates.push_back(x);

    if (params.verbose) {
      std::cout << "Iter: ";
      std::cout.width(iter_field_width);
      std::cout << iteration << ", time: " << elapsed_time << ", f: ";
      std::cout.width(params.precision + 7);
      std::cout << f_x << ", |g|: " << grad_f_x_norm
                << ", |M^{-1}g|: " << preconditioned_grad_f_x_norm;
    }

    // Test gradient-based stopping criterion
    if (grad_f_x_norm < params.gradient_tolerance) {
      result.status = TNTStatus::GRADIENT;
      break;
    }
    if (preconditioned_grad_f_x_norm <
        params.preconditioned_gradient_tolerance) {
      result.status = TNTStatus::PRECONDITIONED_GRADIENT;
      break;
    }

    /// STEP 2:  Solve the local trust-region subproblem at the current
    /// iterate using the computed quadratic model using the Steihaug-Toint
    /// truncated preconditioned conjugate-gradient method

    // Norm of the update in the norm determined by the preconditioner
    size_t inner_iterations;
    Tangent h = STPCG<Variable, Tangent, Scalar, Args...>(
        x, grad, Hess, metric, precon, h_M_norm, inner_iterations, Delta,
        args..., params.max_TPCG_iterations, params.kappa_fgr, params.theta);
    h_norm = sqrt(metric(x, h, h, args...));

    if (params.verbose) {
      std::cout << ", Delta: " << Delta << ", inner iters: ";
      std::cout.width(inner_iter_field_width);
      std::cout << inner_iterations << ", |h|: " << h_norm
                << ", |h|_M: " << h_M_norm;
    }

    /// STEP 3:  Update and evaluate trial point

    // Get trial point
    x_proposed = retract(x, h, args...);

    // Evalute objective at trial point
    f_x_proposed = f(x_proposed, args...);

    // Predicted model decrease
    Scalar dm = -metric(x, grad, h, args...) -
                .5 * metric(x, h, Hess(x, h, args...), args...);

    // Actual function decrease
    Scalar df = f_x - f_x_proposed;

    // Relative function decrease
    relative_decrease = df / (sqrt_eps + fabs(f_x));

    // Evaluate gain ratio
    Scalar rho = df / dm;

    if (params.verbose) {
      std::cout << ", df: ";
      std::cout.width(params.precision + 7);
      std::cout << df << ", rho: ";
      std::cout.width(params.precision + 7);
      std::cout << rho << ". ";
    }

    /// Determine acceptance of trial point
    bool step_accepted = (!std::isnan(rho) && rho > params.eta1);

    if (params.verbose) {
      if (step_accepted)
        std::cout << "Step accepted ";
      else
        std::cout << "Step REJECTED!";
    }

    // Record output
    result.inner_iterations.push_back(inner_iterations);
    result.update_step_norm.push_back(h_norm);
    result.update_step_M_norm.push_back(h_M_norm);
    result.rho.push_back(rho);

    // Call the user-supplied function to provide access to internal algorithm
    // state
    if (user_function)
      (*user_function)(iteration, elapsed_time, x, f_x, grad, Hess, Delta,
                       inner_iterations, h, df, rho, step_accepted, args...);

    /// Update cached values if the iterate is accepted
    if (step_accepted) {
      // Accept iterate and cache values ...
      x = std::move(x_proposed);
      f_x = f_x_proposed;

      // Test relative decrease-based stopping criterion
      if (relative_decrease < params.relative_decrease_tolerance) {
        result.status = TNTStatus::RELATIVE_DECREASE;
        break;
      }

      // Test stepsize-based stopping criterion ...
      if (h_norm < params.stepsize_tolerance) {
        result.status = TNTStatus::STEPSIZE;
        break;
      }

      // ... and recompute the local quadratic model at the current iterate
      QM(x, grad, Hess, args...);

      grad_f_x_norm = sqrt(metric(x, grad, grad, args...));
      if (precon) {
        // Precondition gradient
        Tangent preconditioned_gradient = (*precon)(x, grad, args...);
        preconditioned_grad_f_x_norm = sqrt(metric(
            x, preconditioned_gradient, preconditioned_gradient, args...));
      } else {
        // We use the identity preconditioner, so the norms of the gradient
        // and preconditioned gradient are identical
        preconditioned_grad_f_x_norm = grad_f_x_norm;
      }

    } // if (step_accepted)

    /// STEP 4: Update trust-region radius
    if ((!std::isnan(rho)) && (rho >= params.eta2)) {
      // This iteration was very successful, so we want to increase the
      // trust-region radius
      Delta = std::max<Scalar>(params.alpha2 * h_M_norm, Delta);
    } else if (std::isnan(rho) || (rho < params.eta1)) {
      // This was not a successful iteration, so we should shrink the
      // trust-region radius
      Delta = params.alpha1 * h_M_norm;

      if (Delta < params.Delta_tolerance) {
        result.status = TNTStatus::TRUST_REGION;
        break;
      }
    } // trust-region update

    if (params.verbose)
      std::cout << std::endl;
  } // end trust-region iterations
  result.elapsed_time = Stopwatch::tock(start_time);

  // Record output
  result.x = x;
  result.f = f_x;
  result.grad_f_x_norm = grad_f_x_norm;
  result.preconditioned_grad_f_x_norm = preconditioned_grad_f_x_norm;

  if (params.verbose) {
    std::cout << std::endl
              << std::endl
              << "Optimization finished!" << std::endl;

    // Print the reason for termination
    switch (result.status) {
    case TNTStatus::GRADIENT:
      std::cout << "Found first-order critical point! (Gradient norm: "
                << grad_f_x_norm << ")" << std::endl;
      break;
    case TNTStatus::PRECONDITIONED_GRADIENT:
      std::cout << "Found first-order critical point! (Preconditioned "
                   "gradient norm: "
                << preconditioned_grad_f_x_norm << ")" << std::endl;
      break;
    case TNTStatus::RELATIVE_DECREASE:
      std::cout
          << "Algorithm terminated due to insufficient relative decrease: "
          << relative_decrease << " < " << params.relative_decrease_tolerance
          << std::endl;
      break;
    case TNTStatus::STEPSIZE:
      std::cout
          << "Algorithm terminated due to excessively small step size: |h| = "
          << h_norm << " < " << params.stepsize_tolerance << std::endl;
      break;
    case TNTStatus::TRUST_REGION:
      std::cout << "Algorithm terminated due to excessively small trust region "
                   "radius: "
                << Delta << " < " << params.Delta_tolerance << std::endl;
      break;
    case TNTStatus::ITERATION_LIMIT:
      std::cout << "Algorithm exceeded maximum number of outer iterations"
                << std::endl;
      break;
    case TNTStatus::ELAPSED_TIME:
      std::cout << "Algorithm exceeded maximum allowed computation time: ("
                << result.elapsed_time << " > " << params.max_computation_time
                << " seconds)" << std::endl;
      break;
    }

    std::cout << "Final objective value: " << result.f << std::endl;
    std::cout << "Norm of Riemannian gradient: " << result.grad_f_x_norm
              << std::endl;
    std::cout << "Norm of preconditioned Riemannian gradient: "
              << result.preconditioned_grad_f_x_norm << std::endl;
    std::cout << "Total elapsed computation time: " << result.elapsed_time
              << " seconds" << std::endl
              << std::endl;
  }

  return result;
}

/** Syntactic sugar for the above Riemannian TNT function: this interface
 * enables the user to supply separate functions that return the Riemannian
 * gradient and Hessian operator at X, rather than a single QuadraticModel
 * function.  Here:
 *
 * - gradF is a function that accepts as input a point X in M, and returns
 *   grad f(X), the Riemannian gradient of f at X.
 *
 * - HC is a function that accepts as input a point X in M, and returns
 *   Hess f(X), the linear operator Hess f(X) : T_X(M) -> T_X(M) on the tangent
 *   space T_X(M) of M at X that assigns to each tangent vector V the covariant
 *   derivative of the gradient vector field grad f(X) along V.
 */
template <typename Variable, typename Tangent, typename Scalar = double,
          typename... Args>
TNTResult<Variable, Scalar>
TNT(const Objective<Variable, Scalar, Args...> &f,
    const VectorField<Variable, Tangent, Args...> &grad_f,
    const LinearOperatorConstructor<Variable, Tangent, Args...>
        &HessianConstructor,
    const RiemannianMetric<Variable, Tangent, Scalar, Args...> &metric,
    const Retraction<Variable, Tangent, Args...> &retract, const Variable &x0,
    Args &... args,
    const std::experimental::optional<
        LinearOperator<Variable, Tangent, Args...>> &precon =
        std::experimental::nullopt,
    const TNTParams<Scalar> &params = TNTParams<Scalar>(),
    const std::experimental::optional<
        TNTUserFunction<Variable, Tangent, Scalar, Args...>> &user_function =
        std::experimental::nullopt) {

  // Construct a QuadraticModel function from the passed VectorField and
  // HessianConstructor functions
  QuadraticModel<Variable, Tangent, Args...> QM =
      [&grad_f, &HessianConstructor](
          const Variable &X, Tangent &grad,
          LinearOperator<Variable, Tangent, Args...> &Hess, Args &... args) {
        // Set gradient
        grad = grad_f(X, args...);

        // Set Hessian operator
        Hess = HessianConstructor(X, args...);
      };

  // Now call TNT using this QuadraticModel function
  return TNT<Variable, Tangent, Scalar, Args...>(
      f, QM, metric, retract, x0, args..., precon, params, user_function);
}

/** These next functions provide a convenient specialization/simplification of
 * the Riemannian TNT interface for the (common) use case of optimization over
 * Euclidean spaces.  These functions make the following assumptions:
 *
 * - The metric is the standard Euclidean metric:  g(X, V1, V2) := <V1, V2> for
 *   all X in M
 *
 * - The retraction is the standard Euclidean retraction:  R_X(V) := X + V for
 *   all X in M.
 *
 * - We exploit the global parallelism and self-duality of Euclidean spaces to
 *   represent both *points* X in Euclidean space and *tangent vectors* V using
 *   the *same* data type (Vector).
 */

template <typename Vector, typename Scalar = double, typename... Args>
using EuclideanTNTUserFunction =
    TNTUserFunction<Vector, Vector, Scalar, Args...>;

template <typename Vector, typename Scalar = double, typename... Args>
TNTResult<Vector, Scalar> EuclideanTNT(
    const Objective<Vector, Scalar, Args...> &f,
    const EuclideanQuadraticModel<Vector, Args...> &QM, const Vector &x0,
    Args &... args,
    const std::experimental::optional<EuclideanLinearOperator<Vector, Args...>>
        &precon = std::experimental::nullopt,
    const TNTParams<Scalar> &params = TNTParams<Scalar>(),
    const std::experimental::optional<
        EuclideanTNTUserFunction<Vector, Scalar, Args...>> &user_function =
        std::experimental::nullopt) {

  /// Run TNT algorithm using these Euclidean operators
  return TNT<Vector, Vector, Scalar, Args...>(
      f, QM, EuclideanMetric<Vector, Scalar, Args...>,
      EuclideanRetraction<Vector, Args...>, x0, args..., precon, params,
      user_function);
}

/** Syntactic sugar: enables the user to supply separate functions that return
 * the Euclidean gradient and Hessian operator at X, rather than a single
 * QuadraticModel function */
template <typename Vector, typename Scalar = double, typename... Args>
TNTResult<Vector, Scalar> EuclideanTNT(
    const Objective<Vector, Scalar, Args...> &f,
    const EuclideanVectorField<Vector, Args...> &nabla_f,
    const EuclideanLinearOperatorConstructor<Vector, Args...>
        &HessianConstructor,
    const Vector &x0, Args &... args,
    const std::experimental::optional<EuclideanLinearOperator<Vector, Args...>>
        &precon = std::experimental::nullopt,
    const TNTParams<Scalar> &params = TNTParams<Scalar>(),
    const std::experimental::optional<
        EuclideanTNTUserFunction<Vector, Scalar, Args...>> &user_function =
        std::experimental::nullopt) {

  EuclideanQuadraticModel<Vector, Args...> QM =
      [&nabla_f, &HessianConstructor](
          const Vector &X, Vector &grad,
          EuclideanLinearOperator<Vector, Args...> &Hess, Args &... args) {
        // Set gradient
        grad = nabla_f(X, args...);

        // Set Hessian operator
        Hess = HessianConstructor(X, args...);
      };

  // Now call EuclideanTNT using this QuadraticModel function
  return EuclideanTNT<Vector, Scalar, Args...>(f, QM, x0, args..., precon,
                                               params, user_function);
}

} // namespace Smooth
} // namespace Optimization
