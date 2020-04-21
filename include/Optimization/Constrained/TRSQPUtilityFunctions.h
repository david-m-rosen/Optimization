/** This header file provides a set of lightweight utility functions that are
 * used in the implementation of the main TRSQP optimization method
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "Optimization/Constrained/TRSQPTypes.h"

namespace Optimization {
namespace Constrained {

/** Given:
 * - Jacobians Ax = (Ae(x), Ai(x)) of the equality and inequality constraints
 * - Vector v
 *
 * this function computes and returns the product Ax*v = (Ae(x)*v, Ai(x)*v)
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Pair<EqVector, IneqVector>
compute_A_product(const Pair<EqJacobian, IneqJacobian> &Ax, const Vector &v) {
  return Pair<EqVector, IneqVector>(
      Ax.first.rows() > 0 ? Ax.first * v : EqVector(),
      Ax.second.rows() > 0 ? Ax.second * v : IneqVector());
}

/** Given:
 * - Jacobians Ax = (Ae(x), Ai(x)) of the equality and inequality constraints
 * - Vector v = (vx, vs)
 *
 * this function computes and returns the product Abar*v, where:
 *
 * Abar := [Ae 0]
 *         [Ai I]
 *
 * is the Jacobian of the constraint function
 *
 * c(x,s) := [ce(x)    ]
 *           [ci(x) + s]
 *
 * of the barrier subproblem.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Pair<EqVector, IneqVector>
compute_Abar_product(const Pair<EqJacobian, IneqJacobian> &Ax,
                     const Pair<Vector, IneqVector> &v) {

  return Pair<EqVector, IneqVector>(
      Ax.first.rows() > 0 ? Ax.first * v.first : EqVector(),
      Ax.second.rows() > 0 ? Ax.second * v.first + v.second : IneqVector());
}

/** Given:
 * - Jacobians Ax = (Ae(x), Ai(x)) of the equality and inequality constraints
 * - Vector s of auxiliary slack variables
 * - Input vector v = (vx, vs)
 *
 * this function computes and returns the product product Ahat*v, where:
 *
 * Ahat = [Ae       0]
 *        [Ai diag(s)]
 *
 * is the scaled augmented Jacobian for the constraints of the barrier
 * subproblem.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Pair<EqVector, IneqVector>
compute_Ahat_product(const Pair<EqJacobian, IneqJacobian> &Ax,
                     const IneqVector &s, const Pair<Vector, IneqVector> &v) {
  return Pair<EqVector, IneqVector>(
      Ax.first.rows() > 0 ? Ax.first * v.first : EqVector(),
      Ax.second.rows() > 0 ? Ax.second * v.first + s.hadamard_product(v.second)
                           : IneqVector());
}

/** Given:
 * - Jacobians Ax = (Ae(x), Ai(x)) of the equality and inequality constraints
 * - Vector v = (ve, vi)
 *
 * this function computes and returns the product Ax'*v
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Vector compute_At_product(const Pair<EqJacobian, IneqJacobian> &Ax,
                          const Pair<EqVector, IneqVector> &v) {

  Vector Atv;

  if ((v.first.dim() > 0) && (v.second.dim() > 0)) {
    Atv = Ax.first.transpose() * v.first + Ax.second.transpose() * v.second;
  } else if ((v.first.dim() > 0) && (v.second.dim() == 0)) {
    Atv = Ax.first.transpose() * v.first;
  } else if ((v.first.dim() == 0) && (v.second.dim() > 0)) {
    Atv = Ax.second.transpose() * v.second;
  }

  return Atv;
}

/** Given:
 * - Jacobians Ax = (Ae(x), Ai(x)) of the equality and inequality constraints
 * - Vector v = (ve, vi)
 *
 * this function computes and returns the product Abar'*v, where:
 *
 * Abar := [Ae 0]
 *         [Ai I]
 *
 * is the Jacobian of the constraint function
 *
 * c(x,s) := [ce(x)    ]
 *           [ci(x) + s]
 *
 * of the barrier subproblem.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Pair<Vector, IneqVector>
compute_Abar_transpose_product(const Pair<EqJacobian, IneqJacobian> &Ax,
                               const Pair<EqVector, IneqVector> &v) {

  return Pair<Vector, IneqVector>(
      compute_At_product<Vector, EqVector, IneqVector, EqJacobian,
                         IneqJacobian>(Ax, v),
      v.second);
}

/** Given:
 * - Jacobians Ax = (Ae(x), Ai(x)) of the equality and inequality constraints
 * - Vector s of auxiliary slack variables
 * - Vector v = (ve, vi)
 *
 * this function computes and returns the product product Ahat'*v, where:
 *
 * Ahat = [Ae       0]
 *        [Ai diag(s)]
 *
 * is the scaled augmented Jacobian for the constraints of the barrier
 * subproblem.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Pair<Vector, IneqVector>
compute_Ahat_transpose_product(const Pair<EqJacobian, IneqJacobian> &Ax,
                               const IneqVector &s,
                               const Pair<EqVector, IneqVector> &v) {
  return Pair<Vector, IneqVector>(
      compute_At_product<Vector, EqVector, IneqVector, EqJacobian,
                         IneqJacobian>(Ax, v),
      (s.dim() > 0 ? s.hadamard_product(v.second) : IneqVector()));
}

/** Given:
 * - The gradient gradf(x) of the objective f, evaluated at x
 * - Jacobians A(x) = (Ae(x), Ai(x)) of the equality and inequality
 *   constraints, evaluated at x
 * - Lagrange multipliers lambda = (lambda_e, lambda_i) for the equality and
 *   inequality constraints
 *
 * this function computes and returns the gradient gradL_x(x, lambda) of the
 * Lagrangian:
 *
 * L(x, lambda) = f(x) + <ce(x), lambda_e>, <ci(x), lambda_i>
 *
 * with respect to x, evaluated at (x, lambda).
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Vector compute_gradLx(const Vector &gradfx,
                      const Pair<EqVector, IneqVector> &lambda,
                      const Pair<EqJacobian, IneqJacobian> &Ax) {
  return gradfx + compute_At_product<Vector, EqVector, IneqVector, EqJacobian,
                                     IneqJacobian>(Ax, lambda);
}

/** Given:
 * - Slack variables s
 * - Lagrange multipliers lambda_i for the inequality constraints
 * - Barrier parameter mu
 *
 * this function computes and returns the L2 norm of the complementarity
 * residual
 *
 * r := s o lambda - mu * e
 *
 * for the barrier subproblem, where here 'o' represents the Hadamard
 * (elementwise) product (see eq. (2.3) of "An Interior Point Algorithm for
 * Large-Scale Nonlinear Programming" for details).
 */
template <typename IneqVector, typename Scalar = double>
Scalar compute_barrier_subproblem_complementarity_error(
    const IneqVector &s, const IneqVector &lambda_i, Scalar mu) {
  Scalar squared_error = 0;
  Scalar r;

  for (size_t k = 0; k < s.dim(); ++k) {
    r = s(k) * lambda_i(k) - mu;
    squared_error += r * r;
  }

  return sqrt(squared_error);
}

/** Given the value of the constraint function c(x) = ce(x), ci(x)) at the
 * current iterate x, this function computes and returns the *residual*
 * vector:
 *
 *  r(x) := (ce(x), [ci(x)]_{+})
 *
 * for the constraints ce(x) = 0, ci(x) <= 0.
 */
template <typename EqVector, typename IneqVector>
Pair<EqVector, IneqVector>
compute_constraint_residuals(const Pair<EqVector, IneqVector> &cx) {
  return Pair<EqVector, IneqVector>(
      cx.first, (cx.second.dim() > 0 ? cx.second.max(0) : cx.second));
}

/** Given:
 * - the value of the constraint function c(x) = ce(x), ci(x))
 * - the slack variables s
 *
 * this function computes and returns the residual vector:
 *
 *  r(x,s) := [ce(x)    ]
 *            [ci(x) + s]
 *
 * for the barrier subproblem
 */
template <typename EqVector, typename IneqVector>
Pair<EqVector, IneqVector> compute_barrier_subproblem_constraint_residuals(
    const Pair<EqVector, IneqVector> &cx, const IneqVector &s) {
  return Pair<EqVector, IneqVector>(
      cx.first, (cx.second.dim() > 0 ? cx.second + s : cx.second));
}

/** Given the value of the constraint function c(x) = (ce(x), ci(x)) at the
 * current iterate x, this function computes and returns the norm of the
 * infeasibility measure:
 *
 * v(x) = |  ce(x)      |
 *        | [ci(x)]_{+} |
 *
 * at x, where [x]_{+} represents the elementwise positive part of x.
 */
template <typename EqVector, typename IneqVector, typename Scalar = double>
Scalar compute_infeasibility(const Pair<EqVector, IneqVector> &cx) {
  return compute_constraint_residuals(cx).norm();
}

/** Given:
 *
 * - Value cx = (ce(x), ci(x)) of the constraintsat the current iterate x
 * - Value Ax = (Ae(x), Ai(x)) of the constraint function Jacobians at the
 *   current iterate x
 *
 * this function computes and returns the gradient of the infeasibility
 * measure
 *
 * v(x) = |ce(x)|^2 + | [ci(x)]_{+} |^2
 *
 * at x, where [x]_{+} represents the elementwise positive part of x.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Scalar = double>
Vector
compute_infeasibility_gradient(const Pair<EqVector, IneqVector> &cx,
                               const Pair<EqJacobian, IneqJacobian> &Ax) {

  /// Here we use the fact that if r(x) is vector-valued and
  ///
  /// m(x) := |r(x)|_2
  ///
  /// then:
  ///
  /// grad m(x) = (1/|r(x)|)* A(r)'*r(x)

  // Compute the vector of constraint residuals
  Pair<EqVector, IneqVector> rx = compute_constraint_residuals(cx);
  Scalar rx_norm = rx.norm();

  return (1 / rx_norm) * compute_At_product<Vector, EqVector, IneqVector,
                                            EqJacobian, IneqJacobian>(Ax, rx);
}

/** Given:
 * - The gradient gradLx of the Lagrangian:
 *   L(x, lambda_e, lambda_i) := f(x) + <ce(x), lambda_e> + <ci(x),
 * lambda_i> with respect to x
 * - The value cx = (ce(x), ci(x)) of the constraints at the current iterate
 * x
 * - The value of the slack variables s
 * - The Lagrange multipliers lambda_i for the inequality constraints
 * - The barrier parameter mu:
 *
 * this function computes and return the L2 norm of the total KKT residual:
 *
 * F(x, lambda_e, lambda_i, s, mu) := |      gradLx        |
 *                                    | s o lambda_i - mu*e|
 *                                    |      ce(x)         |
 *                                    |    ci(x) + s       |
 *
 * for the barrier subproblem (cf. eq. (1.12) of the paper "On the Local
 * Behavior of an Interior Point Method for Nonlinear Programming").
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename Scalar = double>
Scalar compute_barrier_subproblem_KKT_residual(
    const Vector &gradLx, const Pair<EqVector, IneqVector> &cx,
    const IneqVector &s, const IneqVector &lambda_i, Scalar mu) {
  Scalar gradLx_norm = gradLx.norm();
  Scalar infeas_norm =
      compute_barrier_subproblem_constraint_residuals(cx, s).norm();
  Scalar comp_err =
      compute_barrier_subproblem_complementarity_error<IneqVector, Scalar>(
          s, lambda_i, mu);

  return sqrt(gradLx_norm * gradLx_norm + infeas_norm * infeas_norm +
              comp_err * comp_err);
}

/** Given:
 * - Slack variables s
 * - Lagrange multipliers lambda_i for the inequality constraints
 * - Current value of the barrier parameter mu
 *
 * this function computes and returns the vector of elements of the
 * (diagonal) primal-dual Hessian approximation Sigma defined by:
 *
 * Sigma_kk = (lambda_i)_k / s_k, if (lambda_i)_k > 0
 *            mu / (s_k)^2,       if (lambda_i)_k <= 0
 *
 * (cf. eq. (3.14) of the paper "An Interior Point Algorithm for Large-Scale
 * Nonlinear Programming).
 */
template <typename IneqVector, typename Scalar = double>
IneqVector compute_Sigma(const IneqVector &s, const IneqVector &lambda_i,
                         Scalar mu) {
  if (s.dim() > 0) {
    IneqVector Sigma = s;
    for (size_t k = 0; k < s.dim(); ++k)
      Sigma(k) = (lambda_i(k) > 0 ? lambda_i(k) / s(k) : mu / ((s(k) * s(k))));
    return Sigma;
  } else
    return IneqVector();
}

/** Given:
 * - Slack variables s
 * - Values of current inequality constraints ci(x)
 *
 * this function updates the slack variables s according to:
 *
 *    s = max(s, -ci(x)).
 *
 * which is equivalent to the condition:
 *
 *   sk = -ci(x)_k for all k such that -ci(x)_k > sk.
 *
 * Note that whenever sk is updated according to this rule, then both
 * (i) ci(x)_k + sk = 0 (i.e. the updated valued of sk better satisfies the
 * constraints) and (ii) the value of sk strictly increases.  Since s enters
 * the merit function only in (i) the equality constraints and (ii) the
 * barrier term (which is monotonically decreasing in s), this slack update
 * has the effect of monotonically decreasing the value of the merit
 * function (cf. Sec. 3.3 of the paper "An Interior Algorithm for Nonlinear
 * Optimization that Combines Line Search and Trust Region Steps").
 */
template <typename IneqVector>
IneqVector reset_slacks(const IneqVector &s, const IneqVector &cix) {
  return s.max(-cix);
}

/** Given:
 * - Slack variables s
 * - Update step ds
 * - Maximum admissible fraction-to-the-boundary parameter tau in (0,1]
 *
 * this function computes the maximum admissible steplength alpha_max:
 *
 * alpha_max := max { alpha in (0, 1] | s + alpha*ds >= (1-tau)*s }
 */
template <typename IneqVector, typename Scalar = double>
Scalar compute_maximum_admissible_steplength(const IneqVector &s,
                                             const IneqVector &ds, Scalar tau) {
  if ((s.dim() == 0) || (ds.min() >= 0)) {
    // If all elements of ds are positive, then no point along the ray s +
    // alpha ds will will intersect the boundary of the nonnegative orthant, so
    // return the full steplength 1

    return 1;
  } else {

    // *Any* positive element of -tau*s / ds > 0 gives the maximum
    // steplength before the fraction-to-the-boundary condition is violated
    // in the corresponding coordinate, and therefore we must return the
    // *minimum* of these

    // Extract and return the *minimum positive* element of -tau*s / ds
    IneqVector neg_tau_s_over_ds =
        -tau * s.hadamard_product(ds.hadamard_inverse());

    Scalar max_steplength_to_boundary = std::numeric_limits<Scalar>::max();
    for (size_t k = 0; k < s.dim(); ++k)
      if (neg_tau_s_over_ds(k) > 0)
        max_steplength_to_boundary =
            std::min<Scalar>(max_steplength_to_boundary, neg_tau_s_over_ds(k));

    return std::min<Scalar>(max_steplength_to_boundary, 1);
  }
}

/** Given:
 * - A Pair of Vectors v representing an update step
 * - Slack variables s
 *
 * this function computes and returns the norm of v in the scaled norm:
 *
 * |(vx, vs)|_M = |(vx, S^-1 vs)|_2
 *
 * used to define the trust-region (cf. eq. (3.16) in the paper "An Interior
 * Point Algorithm for Large-Scale Nonlinear Programming").
 */
template <typename Vector, typename IneqVector, typename Scalar = double>
Scalar compute_trust_region_norm(const Pair<Vector, IneqVector> &v,
                                 const IneqVector &s) {
  Scalar squared_norm = 0;
  if (v.first.dim() > 0)
    squared_norm += std::pow(v.first.norm(), 2);
  if (v.second.dim() > 0)
    squared_norm +=
        std::pow(v.second.hadamard_product(s.hadamard_inverse()).norm(), 2);

  return sqrt(squared_norm);
}

/** Given:
 * - Constraint violations c(x) = (ce(x), ci(x)) at the current iterate x
 * - Slack variables s
 * - Constraint Jacobians A(x) = (Ae(x), Ai(x)) at the current iterate x
 * - Update step v = (vx, vs)
 *
 * this function evaluates the linearized constraint model:
 *
 * m(v) := |ce(x)     + Ae*vx      |
 *         |ci(x) + s + Ai*vx + vs |
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian>
Pair<EqVector, IneqVector>
evaluate_linearized_constraint_model(const Pair<EqVector, IneqVector> &cx,
                                     const IneqVector &s,
                                     const Pair<EqJacobian, IneqJacobian> &Ax,
                                     const Pair<Vector, IneqVector> &v) {

  return compute_barrier_subproblem_constraint_residuals(cx, s) +
         compute_Abar_product<Vector, EqVector, IneqVector, EqJacobian,
                              IneqJacobian>(Ax, v);
}

/** Given:
 * - Constraint violations c(x) = (ce(x), ci(x)) at the current iterate x
 * - Slack variables s
 * - Constraint Jacobians A(x) = (Ae(x), Ai(x)) at the current iterate x
 * - Update step v = (vx, vs)
 *
 * this function computes and the reduction in the norm of the linearized
 * constraint violation after applying the update step v
 *
 * vpred = |ce(x)    |^2  - |ce(x)      + Ae(x)*vx     |^2
 *         |ci(x) + s|      |ci(x) + s  + Ai(x)*vx + vs|
 *
 * (cf. eq. (3.50) in the paper "An Interior Point Algorithm for Large-Scale
 * Nonlinear Programming").
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Scalar = double>
Scalar vpred(const Pair<EqVector, IneqVector> &cx, const IneqVector &s,
             const Pair<EqJacobian, IneqJacobian> &Ax,
             const Pair<Vector, IneqVector> &v) {

  // Current constraint violation
  Scalar rnorm = compute_barrier_subproblem_constraint_residuals(cx, s).norm();

  // Violation of linearized constraints after applying the update v
  Scalar rplus_norm = evaluate_linearized_constraint_model(cx, s, Ax, v).norm();

  return rnorm - rplus_norm;
}

/** Given:
 * - Value of object fx, evaluated at the current iterate x
 * - Value of auxiliary slack variables s
 * - Barrier parameter mu
 *
 * this function evaluates the objective function of the barrier subproblem:
 *
 * varphi(x, s; mu) := f(x) - mu* sum_k log(s(k))
 */
template <typename IneqVector, typename Scalar = double>
Scalar evaluate_barrier_subproblem_objective(Scalar fx, const IneqVector &s,
                                             Scalar mu) {
  // Objective term
  Scalar varphi = fx;

  // Logarithmic barrier term
  for (size_t k = 0; k < s.dim(); ++k)
    varphi -= mu * log(s(k));

  return varphi;
}

/** Given:
 * - Gradient gradfx of the f, evaluated at the current iterate x
 * - Value of auxiliary slack variables s
 * - Barrier parameter mu
 *
 * this function constructs and returns the gradient of the objective:
 *
 * varphi(x, s; mu) := f(x) - mu * sum_k log(s(k))
 *
 * of the barrier subproblem
 */
template <typename Vector, typename IneqVector, typename Scalar = double>
Pair<Vector, IneqVector>
compute_barrier_subproblem_objective_gradient(const Vector &gradfx,
                                              const IneqVector &s, Scalar mu) {

  return Pair<Vector, IneqVector>(gradfx, -mu * s.hadamard_inverse());
}

/** Given:
 * - Gradient gradfx of the f, evaluated at the current iterate x
 * - Value of auxiliary slack variables s
 * - Value of constraint Jacobians Ax
 * - Lagrange multipliers lambda
 * - Barrier parameter mu
 *
 * this function constructs and returns the gradient of the Lagrangian of
 * the barrier subproblem with respect to z := (x,s):
 *
 * gradLz := grad_z_varphi + A'*lambda
 *
 * where
 *
 * A := [Ae 0]
 *      [Ai I]
 *
 * is the Jacobian of the barrier subproblem constraint function:
 *
 * c(z) := |ce(x)    |
 *         |ci(x) + s|
 *
 * with respect to z.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Scalar = double>
Pair<Vector, IneqVector> compute_barrier_subproblem_gradient_of_Lagrangian(
    const Vector &gradfx, const IneqVector &s,
    const Pair<EqJacobian, IneqJacobian> &Ax,
    const Pair<EqVector, IneqVector> &lambda, Scalar mu) {

  return compute_barrier_subproblem_objective_gradient(gradfx, s, mu) +
         compute_Abar_transpose_product<Vector, EqVector, IneqVector,
                                        EqJacobian, IneqJacobian>(Ax, lambda);
}

/** Given:
 * - Tangent vector v := (vx, vs)
 * - Hessian HxL of Lagrangian wrt x at current iterate x
 * - Primal-dual approximation to the Hessian of the Lagrangian wrt
 * auxiliary slack variables Sigma
 *
 * this function computes and returns the product W*v, where
 *
 * W = [HxL     0]
 *     [0   Sigma]
 *
 * is the model Hessian for the Lagrangian of the barrier subproblem:
 *
 * min f(x) - mu * sum_k log(sk)
 *
 * st.  ce(x) = 0
 *      ci(x) + s = 0
 *
 * with respect to z := (x,s)
 */
template <typename Vector, typename IneqVector, typename Hessian>
Pair<Vector, IneqVector>
compute_barrier_subproblem_Hessian_of_Lagrangian_product(
    const Hessian &HxL, const IneqVector &Sigma,
    const Pair<Vector, IneqVector> &v) {
  Pair<Vector, IneqVector> Wv;

  return Pair<Vector, IneqVector>(HxL * v.first,
                                  Sigma.hadamard_product(v.second));
}

/** Given:
 * - Update step d = (dx, ds)
 * - Gradient gradfx of objective at current iterate x
 * - Current values of slack variables s
 * - Hessian HxL of Lagrangian wrt x at current iterate x
 * - Primal-dual approximation to the Hessian of the Lagrangian wrt
 * auxiliary slack variables Sigma
 * - Current barrier parameter mu
 *
 * this function computes and returns the value of the quadratic model of
 * the tangential subproblem:
 *
 * q(d) := <gradfx, dx> - <mu*e, S^-1 * ds> + (1/2)*[<dx, HxL*dx>
 *                                                  + <ds,Sigma*ds>]
 *
 * (cf. eq. (3.25) in the paper "An Interior Point Algorithm for Large-Scale
 * Nonlinear Programming).
 */
template <typename Vector, typename IneqVector, typename Hessian,
          typename Scalar = double>
Scalar evaluate_quadratic_model(const Pair<Vector, IneqVector> &d,
                                const Vector &gradfx, const IneqVector &s,
                                const Hessian &HxL, const IneqVector &Sigma,
                                Scalar mu) {

  return d.inner_product(
             compute_barrier_subproblem_objective_gradient(gradfx, s, mu)) +
         .5 * d.inner_product(
                  compute_barrier_subproblem_Hessian_of_Lagrangian_product(
                      HxL, Sigma, d));
}

/** Given:
 * - Value of object fx, evaluated at the current iterate x
 * - Value of constraint functions cx = (ce(x), ci(x)), evaluated at current
 *   iterate x
 * - Value of auxiliary slack variables s
 * - Barrier parameter mu
 * - Penalty parameter nu
 *
 * this function evaluates the merit function:
 *
 * phi(x, s; mu, nu) := f(x) - mu* sum_k log(s(k)) + nu * | ce(x)    |
 *                                                        | ci(x) + s|_2
 *
 * used to determine update step acceptance (cf. eq. (2.8) in the paper "An
 * Interior Point Algorithm for Large-Scale Nonlinear Programming").
 */
template <typename EqVector, typename IneqVector, typename Scalar = double>
Scalar evaluate_merit_function(Scalar fx, const Pair<EqVector, IneqVector> &cx,
                               const IneqVector &s, Scalar mu, Scalar nu) {

  return evaluate_barrier_subproblem_objective(fx, s, mu) +
         nu * compute_barrier_subproblem_constraint_residuals(cx, s).norm();
}

/** Given:
 * - Current trust-region radius Delta
 * - The gain ratio rho for the computed trust-region step
 * - The gain ratios thresholds eta1 and eta2 for successful and very
 *    successful iterations
 * - The contraction and expansion factors alpha1 and alpha2
 * - The norm of the current update step d
 *
 * this function computes and returns the updated trust-region radius
 */
template <typename Scalar = double>
Scalar update_trust_region(Scalar Delta, Scalar rho, Scalar eta1, Scalar eta2,
                           Scalar alpha1, Scalar alpha2, Scalar d_norm) {
  if ((rho >= eta2) && (d_norm >= .75 * Delta)) {
    // If the current step was very successful and the update step was more
    // than 75% of the distance to the trust-region boundary, increase the
    // trust-region radius
    return alpha2 * Delta;
  } else if (rho >= eta1) {
    return Delta;
  } else {
    // The current step was rejected, so shrink the trust-region radius to
    // half of the current steplength
    return alpha1 * d_norm;
  }
}

/** Given:
 *
 * - Gradient of the objective gradfx
 * - Constraint Jacobians Ax = (Ae(x), Ai(x))
 * - Lagrange multiplier estimates lambda = (lambda_e, lambda_i)
 * - Current slack variables s
 * - Current barrier parameter mu
 *
 * this function computes and returns the gradient of the (scaled) least-squares
 * loss function:
 *
 * l(lambda) := |Ahat'*lambda + [gradfx] |^2
 *              |              [-mu*e ] |
 *
 * that the least-squares Lagrange multipliers minimize.
 *
 * [Note that this function is only used when the code is compiled in Debug
 * mode, to ensure that the least-squares Lagrange multiplier estimates are
 * computed correctly]
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Scalar = double>
Pair<EqVector, IneqVector> compute_gradient_of_multiplier_loss(
    const Vector &gradfx, const Pair<EqJacobian, IneqJacobian> &Ax,
    const Pair<EqVector, IneqVector> &lambda, const IneqVector &s, Scalar mu) {

  // Construct constant vector [grafx, -mu*e]
  Pair<EqVector, IneqVector> b(gradfx, s);
  if (dim(s) > 0)
    for (size_t k = 0; k < dim(s); ++k)
      b.second(k) = -mu;

  // Compute residual r := Ahat'*lambda + b
  Pair<Vector, IneqVector> r =
      compute_Ahat_transpose_product<Vector, EqVector, IneqVector, EqJacobian,
                                     IneqJacobian>(Ax, s, lambda) +
      b;

  // NB:  Gradient gradL = 2Ahat*(Ahat'*lambda + g) = 2*Ahat *r
  Pair<EqVector, IneqVector> gradl_lambda =
      2 * compute_Ahat_product<Vector, EqVector, IneqVector, EqJacobian,
                               IneqJacobian>(Ax, s, r);

  return gradl_lambda;
}

/** Given:
 *
 * - Gradient grad_varphi(z) of the barrier subproblem objective at the
 * current iterate z := (x, s)
 * - Current values cx of the constraint functions
 * - Current Lagrange multipliers lambda
 * - Current constraint Jacobians Ax
 * - Current Hessian models HxL and Sigma
 * - Primal and dual update steps dz and dlambda
 *
 * this function computes and returns the residual of the following linear
 * system, whose solution yields the primal-dual Newton update step:
 *
 * [W A'][dz]      = [-gradLz]
 * [A 0 ][dlambda] = [-cx]
 *
 * (see eq. (2.6) of the paper "An Interior Algorithm for Nonlinear
 * Optimization That Combines Line Search and Trust-Region Steps"
 *
 * [Note that this function is only used when the code is compiled in Debug
 * mode, to ensure that the least-squares Lagrange multiplier estimates are
 * computed correctly]
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Hessian,
          typename Scalar = double, typename... Args>
Scalar compute_primal_dual_system_residual_norm(
    const Pair<Vector, IneqVector> &grad_varphi,
    const Pair<EqVector, IneqVector> &cx, const IneqVector &s,
    const Pair<EqVector, IneqVector> &lambda,
    const Pair<EqJacobian, IneqJacobian> &Ax, const Hessian &HxL,
    const IneqVector &Sigma, const Pair<Vector, IneqVector> &dz,
    const Pair<EqVector, IneqVector> &dlambda) {

  // Construct initial optimality residual
  Pair<Vector, IneqVector> r_opt =
      grad_varphi +
      compute_barrier_subproblem_Hessian_of_Lagrangian_product(HxL, Sigma, dz) +
      compute_Abar_transpose_product(Ax, lambda + dlambda);

  // Construct initial feasibility residual
  Pair<Vector, IneqVector> r_feas =
      evaluate_linearized_constraint_model(cx, s, Ax, dz);

  Scalar residual_norm =
      sqrt(std::pow(r_opt.norm(), 2) + std::pow(r_feas.norm(), 2));

  return residual_norm;
}

} // namespace Constrained
} // namespace Optimization
