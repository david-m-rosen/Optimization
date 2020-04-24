/** This header file defines a few additional types that are required by the
 * TRSQP optimization method
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "Optimization/Constrained/Concepts.h"

namespace Optimization {

namespace Constrained {
/// Additional types required by the TRSQP method

/** A set of status flags indicating the stopping criterion that triggered
 * algorithm termination */
enum class TRSQPStatus {

  /** The algorithm terminated at a KKT point */
  KKT,

  /** The algorithm terminated at an infeasible stationary point */
  InfeasibleStationary,

  //  /** The algorithm terminated because the norm of the last accepted
  //  update
  //     step was less than the specified tolerance */
  //  Stepsize,

  /** The algorithm terminated because the trust-region radius decreased
   * below the specified threshold */
  TrustRegion,

  /** The algorithm exhausted the allotted number of major (outer) iterations
   */
  IterationLimit,

  /** The algorithm exhausted the allotted computation time */
  ElapsedTime,

  /** The algorithm terminated due to the user-supplied stopping criterion */
  UserFunction
};

/** A lightweight enumeration class used to indicate which type of step
 * (Composite or Primal-dual) was applied at each iteration of the TRSQP
 * algorithm.
 */
enum class TRSQPStepType {
  /** Byrd-Omojokun-type composite trust-region update step */
  TrustRegion,

  /** Update step computed by solving the joint primal-dual equation  */
  PrimalDual
};

template <typename Scalar = double>
struct TRSQPParams : public NLPOptimizerParams<Scalar> {
  /// Trust-region control parameters

  /** Minimum admissible size of trust-region radius */
  Scalar DeltaMin = 1e-10;

  /** Initial size of trust-region radius */
  Scalar Delta0 = 1;

  /** Trust-region scaling factor for normal step computation (should satisfy
   * 0 < zeta < 1) */
  Scalar zeta = .8;

  /** Admissible fractional distance to the boundary of the nonnegative
   * orthant permitted for updates to slack variables (should satisfy 0 < tau
   * < 1)
   */
  Scalar tau = .995;

  /** Elementwise lower-bound to enforce when initializing slacks */
  Scalar s0_min = 1e-3;

  /** Lower-bound on the gain ratio for accepting a proposed step (should
   * satisfy 0 < eta1 <= eta2) */
  Scalar eta1 = .001;

  /** Lower-bound on the gain ratio for a 'very successful' iteration (should
   * satisfy eta1 <= eta2 < 1). */
  Scalar eta2 = .75;

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
  size_t max_STPCG_iterations = 1000;

  /** Stopping tolerances for the computation of the tangential update step
   * using the Steihaug-Toint truncated preconditioned projected
   * conjugate-gradient algorithm: termination occurs whenever the residual
   * r_k, as measured in the P-norm determined by the constraint
   * preconditioner P:
   *
   *     || r ||_P := sqrt(r'*P_x(r))
   *
   * satisfies:
   *
   *     || r_k ||_P <= ||r_0||_P min [kappa, || r_0 ||_P^theta ]
   *
   * (cf. Algorithm 7.5.1 of "Trust-Region Methods").  kappa and theta should
   * both be in the range (0, 1).  Moreover, theta controls the asymptotic
   * convergence rate of the overall trust-region SQP algorithm;
   * specifically, the the overall algorithm converges with order (1 + theta)
   * (cf. Theorem 2.3 of "Truncated-Newton Algorithms for Large-Scale
   * Unconstrained Optimization", by R.S. Dembo and T. Steihaug.) */
  Scalar cg_kappa_fgr = .1;
  Scalar cg_theta = .5;

  /// Primal-dual line search parameters

  /** Sufficient decrease parameter for the Armijo backtracking line-search;
   * this parameter should be in the range (0,1) */
  Scalar ls_alpha = .5;

  /** Maximum number of backtracking line-search iterations to attempt */
  size_t max_ls_iters = 5;

  /** Minimum steplength for accepting a primal-dual update step */
  Scalar ls_alpha_min = 1e-6;

  /// Barrier subproblem parameters

  /** Initial value of barrier parameter mu */
  Scalar mu0 = 1e-4;

  /** Multiplicative update for barrier parameter; this value must be in the
   * range (0,1) */
  Scalar mu_theta = 1e-1;

  /** Initial KKT residual tolerance for the barrier subproblems; this value
   * must be a positive value */
  Scalar epsilon_mu0 = 1e-2;

  /** Multiplicative update for barrier parameter; this value must be in the
   * range (0,1) */
  Scalar epsilon_mu_theta = 1e-2;

  /// Merit function parameters

  /** Required ratio of total reduction in merit function to improvement in
   * linearized feasibility; this parameter should satisfy 0 < rho < 1 (cf.
   * eq. (3.49) in the paper "An Interior Point Algorithm for Large-Scale
   * Nonlinear Programming") */
  Scalar rho = .1;

  /** Initial value of merit function penalty parameter */
  Scalar nu0 = 1e-3;
};

/** A useful struct used to hold the output of a truncated-Newton
 * trust-region optimization method */
template <typename Vector, typename EqVector, typename IneqVector,
          typename Scalar = double>
struct TRSQPResult
    : public NLPOptimizerResult<Vector, EqVector, IneqVector, Scalar> {

  /** The stopping condition that triggered algorithm termination */
  TRSQPStatus status;

  /** Sequence of barrier parameters used to compute each update step */
  std::vector<Scalar> barrier_params;

  /** The trust-region radius at the START of each iteration */
  std::vector<Scalar> trust_region_radius;

  /** The number of iterations performed by the Steihaug-Toint projected
   * preconditioned conjugate-gradient method to compute the tangential
   * update step in each major (outer) iteration in which a composite
   * trust-region step is computed */
  std::vector<size_t> STPCG_iters;

  /** The number of backtracking linesearches performed during each major
   * (outer) iteration in which a primal-dual update step is computed. */
  std::vector<size_t> linesearch_iters;

  /** The norm of the (primal) update step dx */
  std::vector<size_t> dx_norms;

  /** Sequence of merit function penalty parameters used to determine
   * acceptance or rejection of each update step*/
  std::vector<Scalar> penalty_params;

  /** The type of update step applied at each iteration */
  std::vector<TRSQPStepType> step_types;

  /** A vector of Boolean values indicating whether a second-order correction
   * was applied to the update step used in each iteration */
  std::vector<bool> SOCs;

  /** The gain ratio of the update step computed during each iteration */
  std::vector<Scalar> gain_ratios;
};

/// Alias templates for linear system solvers that are utilized by the TRSQP
/// subproblem solvers

/** An alias template for a user-definable function that solves linear systems
 * of the form:
 *
 *   [I    Ahat'][v] = [b]
 *   [Ahat     0][y] = [c]
 *
 * where:
 *
 * - Ahat := [Aex       0]
 *           [Aix diag(s)]
 *
 * - Ax = (Ae(x), Ai(x)) is a pair of Jacobians for the (vector-valued) equality
 *   and inequality constraint functions c(x) := (ce(x), ci(x))
 * - s is a vector of auxiliary slack variables corresponding to the inequality
 *   constraints
 * - new_coefficient_matrix is a Boolean value indicating whether this is the
 *   first time that a linear system involving the augmented system matrix
 *
 *   [I    Ahat']
 *   [Ahat     0]
 *
 *   has been solved -- this enables the function to make use of e.g. efficient
 *   matrix factorization caching schemes in the case of solving linear systems
 *   with multiple right-hand sides
 * - b = (bx, bs) is the right-hand side vector corresponding to the (primal)
 *   optimality conditions
 * - c = (ce, ci) is the right-hand side vector corresponding to the feasibility
 *   conditions
 * - v = (vx, vs) is the part of the solution corresponding to the (augmented)
 *   primal variables
 * - y = (ye, yi) is the part of the solution corresponding to the Lagrange
 *   multipliers associated with the equality and inequality constraints.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename... Args>
using AugmentedSystemSolverFunction = std::function<void(
    const Pair<EqJacobian, IneqJacobian> &Ax, const IneqVector &s,
    bool new_coefficient_matrix, const Pair<Vector, IneqVector> &b,
    const Pair<EqVector, IneqVector> &c, Pair<Vector, IneqVector> &v,
    Pair<EqVector, IneqVector> &y, Args &...)>;

/** An alias template for a user-definable function that solves KKT systems of
 * the form
 *
 *   [W  A'][v] = [b]
 *   [A   0][y] = [c]
 *
 * where:
 *
 * - W := [HxL      0]
 *        [0    Sigma]
 *
 * - A := [Aex 0]
 *        [Aix I]
 *
 * - Ax = (Ae(x), Ai(x)) is a pair of Jacobians for the (vector-valued) equality
 *   and inequality constraint functions c(x) := (ce(x), ci(x))
 * - new_coefficient_matrix is a Boolean value indicating whether this is the
 *   first time that a linear system involving the augmented system matrix
 *
 *   [W    A']
 *   [A     0]
 *
 *   has been solved -- this enables the function to make use of e.g. efficient
 *   matrix factorization caching schemes in the case of solving linear systems
 *   with multiple right-hand sides
 * - b = (bx, bs) is the right-hand side vector corresponding to the (primal)
 *   optimality conditions
 * - c = (ce, ci) is the right-hand side vector corresponding to the feasibility
 *   conditions
 * - v = (vx, vs) is the part of the solution corresponding to the (augmented)
 *   primal variables
 * - y = (ye, yi) is the part of the solution corresponding to the Lagrange
 *   multipliers associated with the equality and inequality constraints.
 *
 * Note that the passed primal-dual system matrix is NOT guaranteed to be
 * nonsingular (in which case the desired primal-dual update step (v, y) may not
 * exist); this function should explicitly check the input primal-dual system
 * for possible ill-conditioning or singularity.
 *
 * This function should return 'true' if the KKT system was successfully solved,
 * or 'false' otherwise (in which case the primal and dual solutions v and y are
 * not required to be set).
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Hessian,
          typename... Args>
using KKTSystemSolverFunction = std::function<bool(
    const Hessian &HxL, const IneqVector &Sigma,
    const Pair<EqJacobian, IneqJacobian> &Ax, bool new_coefficient_matrix,
    const Pair<Vector, IneqVector> &b, const Pair<EqVector, IneqVector> &c,
    Pair<Vector, IneqVector> &v, Pair<EqVector, IneqVector> &y, Args &...)>;

/** An alias template for a user-defined function that specifies under what
 * conditions the application of a primal-dual update step should be
 * attempted. Here:
 *
 * - k: index of current iteration
 * - t: total elapsed computation time at the *start* of the current
 *   iteration
 * - x: current primal iterate
 * - s: current values of auxiliary slack variables
 * - lambda: current set of Lagrange multipliers
 * - fx: Current objective value
 * - gradfx: Gradient of the objective at x
 * - Hx: Hessian of the Lagrangian L(x) := f(x) + <c(x), lambda> wrt x
 * - Sigma:  Primal-dual approximation of the Hessian of the Lagrangian wrt s
 * - cx: Current value of constraint functions
 * - Ax: Current value of constraint Jacobians
 * - mu: Current value of barrier parameter
 * - epsilon:  KKT residual tolerance for the current barrier subproblem
 * - Delta: Current trust-region radius
 * - prev_step_type: The type of step computed during the *previous*
 *   iteration
 * - CG_iters: number of conjugate-gradient iterations used in the
 *   computation of the *previous* tangential update step, if it was a
 *   composite update
 * - d:  Update step computed during the *previous* iteration (including
 *   second-order correction, if one was applied)
 * - prev_step_accepted:  Boolean value indicating whether the
 *   previously-compute update step was accepted
 *
 * This function is called once per (major) iteration, *after* the
 * convergence criteria have been checked and the barrier parameter and
 * subproblem stopping tolerances have been updated, but *before* any step
 * computations are performed.
 *
 * This function should return 'true' to indicate that a primal-dual update
 * step should be attempted
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Hessian,
          typename Scalar = double, typename... Args>
using PrimalDualStrategyFunction = std::function<bool(
    size_t k, double t, const Vector &x, const IneqVector &s,
    const Pair<EqVector, IneqVector> &lambda, Scalar fx, const Vector &gradfx,
    const Hessian &Hx, const IneqVector &Sigma,
    const Pair<EqVector, IneqVector> &cx,
    const Pair<EqJacobian, IneqJacobian> &Ax, Scalar mu, Scalar epsilon,
    Scalar Delta, TRSQPStepType prev_step_type, size_t CG_iters,
    const Pair<Vector, IneqVector> &d, bool prev_step_accepted,
    Args &... args)>;

/** An alias template for a user-definable function that can be
 * used to access various internal quantities of interest as the TRSQP
 * algorithm runs.  Here:
 *
 * - k: index of current iteration
 * - t: total elapsed computation time at the *start* of the current iteration
 * - x: iterate at the *start* of the current iteration
 * - s: auxiliary slack variables
 * - lambda: Lagrange multipliers at the *start* of the current iteration
 * - f: objective value at x
 * - gradfx: Gradient of the objective at x
 * - Hx: Hessian of the Lagrangian L(x) := f(x) + <c(x), lambda> wrt x
 * - Sigma:  Primal-dual approximation of the Hessian of the Lagrangian wrt s
 * - cx: Value of constraint functions at the *start* of the current iteration
 * - Ax: Value of constraint Jacobians at the *start* of the current iteration
 * - mu: Value of barrier parameter used to compute the update step
 * - epsilon:  KKT residual tolerance for the current barrier subproblem
 * - Delta: trust-region radius at the *start* of the current iteration
 * - step_type: The type of step computed during the current iteration
 * - CG_iters: number of conjugate-gradient iterations used to compute the
 *   tangential update step
 * - v:  Normal component of the update step
 * - w:  Tangential component of the update step
 * - d:  Update step computed during the current iteration (including
 *   second-order correction, if one was applied)
 * - nu:  Current value of the merit function penalty parameter
 * - rho: value of the gain ratio for the proposed composite update step
 * - SOC_applied:  A Boolean value indicating whether a second-order
 *   correction was applied to the current update step
 * - accepted:  Boolean value indicating whether the proposed trust-region
 *   update step h was accepted
 *
 * This function is called at the end of each outer (major) iteration, after
 * all of the above-referenced quantities have been computed, but *before*
 * the update step d computed during this iteration is applied.
 *
 * This function may also return the Boolean value 'true' in order to
 * terminate the algorithm; this provides a convenient means of implementing
 * a custom (user-definable) stopping criterion.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Hessian,
          typename Scalar = double, typename... Args>
using TRSQPUserFunction = std::function<bool(
    size_t k, double t, const Vector &x, const IneqVector &s,
    const Pair<EqVector, IneqVector> &lambda, Scalar fx, const Vector &gradfx,
    const Hessian &Hx, const IneqVector &Sigma,
    const Pair<EqVector, IneqVector> &cx,
    const Pair<EqJacobian, IneqJacobian> &Ax, Scalar mu, Scalar epsilon,
    Scalar Delta, TRSQPStepType step_type, size_t CG_iters,
    const Pair<Vector, IneqVector> &v, const Pair<Vector, IneqVector> &w,
    const Pair<Vector, IneqVector> &d, Scalar nu, Scalar rho, bool SOC_applied,
    bool accepted, Args &... args)>;

} // namespace Constrained
} // namespace Optimization
