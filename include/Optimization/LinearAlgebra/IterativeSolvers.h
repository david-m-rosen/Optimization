/** This header file provides several Krylov-subspace-based iterative
 * linear-algebra methods commonly used in the implementation of optimization
 * algorithms.
 *
 * Copyright (C) 2017 - 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "Optimization/LinearAlgebra/Concepts.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <experimental/optional>
#include <iostream>
#include <limits>

namespace Optimization {

namespace LinearAlgebra {

/** An alias template for a user-definable function that can be
 * used to access various interesting bits of information about the internal
 * state of the Steihaug-Toint truncated preconditioned projected
 * conjugate-gradient algorithm as it runs. Here:
 *
 * - k: Index of current iteration
 * - g: Gradient vector for quadratic model
 * - H: Hessian operator for quadratic model
 * - P: Optional preconditioning operator
 * - At: Optional transpose of constraint operator
 * - sk: The value of the estimated solution at the *start* of the current
 *   iteration
 * - rk: Value of the residual rk := g + H*sk at the *start of the current
 *   iteration
 * - vk: Projected preconditioned residual at the *start* of the current
 *   iteration
 * - pk: The update direction computed during this iteration
 * - alpha_k: The steplength along pk computed during this iteration
 *
 * This function is called at the end of each iteration, after all of the
 * above-referenced quantities have been computed, but *before* the update step
 * alpha_k * pk computed during this iteration is to update the solution sk.
 *
 * This function may also return the Boolean value 'true' in order to
 * terminate the STPCG algorithm; this provides a convenient means of
 * implementing a custom (user-definable) stopping criterion.
 */
template <typename Vector, typename Multiplier, typename Scalar = double,
          typename... Args>
using STPCGUserFunction =
    std::function<bool(size_t k, const Vector &g,
                       const SymmetricLinearOperator<Vector, Args...> &H,
                       const std::experimental::optional<LinearOperator<
                           Vector, std::pair<Vector, Multiplier>, Args...>> &P,
                       const std::experimental::optional<
                           LinearOperator<Multiplier, Vector, Args...>> &At,
                       const Vector &sk, const Vector &rk, const Vector &vk,
                       const Vector &pk, Scalar alpha_k, Args &... args)>;

/** An alias template for preconditioners used in conjunction with the
 * Steihaug-Toint truncated preconditioned projected conjugate-gradient
 * algorithm for solving trust-region problems of the form:
 *
 * min_s <g, s> + (1/2)<s, H*s>
 * s.t.  A*s = 0
 *       |s|_P <= Delta
 *
 * A preconditioner P of this type must be a linear map that accepts
 * as input a vector s of the same type as the decision variable of the
 * minimization problem, and returns a *pair* of values:
 *
 * P(s) = (v, lambda)
 *
 * that satisfy the linear system:
 *
 * [M A'][v] = [s]
 * [A 0 ][l] = [0]
 *
 * where M is a positive definite approximation of H (used to precondition the
 * projected CG algorithm).
 */
template <typename Vector, typename Multiplier, typename... Args>
using STPCGPreconditioner =
    LinearOperator<Vector, std::pair<Vector, Multiplier>, Args...>;

/** This function implements the Steihaug-Toint truncated preconditioned
 * projected conjugate-gradient algorithm for approximately solving a
 * trust-region problem of the form:
 *
 *   min_s <g, s> + (1/2)<s, H*s>
 *   s.t.  A*s = 0
 *         |s|_P <= Delta
 *
 * This specific implementation is based upon Algorithms 5.4.2 and 7.5.1 in
 * the reference "Trust-Region Methods" by Conn, Gould, and Toint, and
 * Algorithms 6.1 and 6.3 of the paper "On the Solution of Equality
 * Constrained Quadratic Programming Problems Arising in Optimization" by
 * N.I.M. Gould, M.E. Hribar, and J. Nocedal.
 *
 * Here:
 *
 * - g is the gradient of the model objective
 * - H is a symmetric linear operator defining the Hessian of the model
 *   objective
 * - inner_product is the inner product operator for the space containing
 *   the trust-region subproblem's decision variable s.
 * - P is an optional constraint preconditioning operator that accepts as
 *   input a vector s and returns the pair
 *
 *   P(s) = (x, lambda)
 *
 *   where x and lambda are the (unique) solution of the following linear
 *   system:
 *
 *   [M A'][x] = [s]
 *   [A 0 ][l]   [0]
 *
 *   and M is a positive definite preconditioner for H.  Note that in the
 *   case no linear constraints are present, the projected preconditioning
 *   operator P reduces to P(s) = M^{-1} s, where M is a positive-definite
 *   approximation of H.  If this argument is omitted, a default (no-op)
 *   preconditioner is used (corresponding to M = I).
 *
 *   || s ||_M <= Delta
 *
 * - At is an optional linear operator that computes the transpose AT : L ->
 *   S of the linear operator A : S -> L whose null space determines the
 *   feasible set of the trust-region subproblem.  Omission of this argument
 *   indicates that the trust-region subproblem has no linear constraints.
 *
 * - update_step_M_norm is a return value that gives the norm || h ||_M of
 *   the update step in the M-norm determined by the preconditioner
 *
 * - num_iterations is a return value that gives the number of iterations
 *   executed by the algorithm
 *
 * - Delta is the radius of the trust-region in the M-norm determined by the
 *   positive-definite matrix M defining the constraint preconditioner P:
 *
 *   |s|_M = sqrt(s'*M*s)
 *
 * - max_iterations is the maximum admissible number of iterations for the
 *   STPCG algorithm
 *
 * - kappa_fgr and theta are parameters that control the stopping criteria
 *   of the algorithm; termination occurs whenever the residual r_k, as
 *   measured in the P-norm determined by the constraint preconditioner P:
 *
 *     || r ||_P := sqrt(r'*P_x(r))
 *
 *   satisfies:
 *
 *     || r_k ||_P <= ||r_0||_P min [kappa, || r_0 ||_P^theta ]
 *
 *   (cf. Algorithm 7.5.1 of "Trust-Region Methods").  kappa and theta should
 *   both be in the range (0, 1)
 *
 * - user_function is an optional user-defined function that can be used to
 *   inspect various variables of interest as the algorithm runs, and
 *   (optionally) to define a custom stopping criterion for the algorithm.
 *
 * - epsilon is a numerical tolerance for determining whether a vector lies
 *   in the kernel of  the operator H, defined as |H*x|/|x| < epsilon.
 */
template <typename Vector, typename Multiplier, typename Scalar = double,
          typename... Args>
Vector STPCG(const Vector &g, const SymmetricLinearOperator<Vector, Args...> &H,
             const InnerProduct<Vector, Scalar, Args...> &inner_product,
             Args &... args, Scalar &update_step_M_norm, size_t &num_iterations,
             Scalar Delta, size_t max_iterations = 1000, Scalar kappa_fgr = .1,
             Scalar theta = .5,
             const std::experimental::optional<
                 STPCGPreconditioner<Vector, Multiplier, Args...>> &P =
                 std::experimental::nullopt,
             const std::experimental::optional<
                 LinearOperator<Multiplier, Vector, Args...>> &At =
                 std::experimental::nullopt,
             const std::experimental::optional<
                 STPCGUserFunction<Vector, Multiplier, Scalar, Args...>>
                 &user_function = std::experimental::nullopt,
             Scalar epsilon = 1e-8) {

  /// Argument checking

  if (Delta <= 0)
    throw std::invalid_argument(
        "Trust-region radius (Delta) must be a positive real value");

  if (max_iterations < 0)
    throw std::invalid_argument("Maximum number of iterations (max_iterations) "
                                "must be a nonnegative integer");

  if ((kappa_fgr < 0) || (kappa_fgr >= 1))
    throw std::invalid_argument(
        "Target fractional reduction of the gradient norm "
        "(kappa_fgr) must be a real value in the range [0,1)");

  if ((theta < 0) || (theta > 1))
    throw std::invalid_argument(
        "Target superlinear convergence rate (theta) must "
        "be a real value in the range [0,1]");

  if ((epsilon <= 0) || (epsilon >= 1))
    throw std::invalid_argument(
        "Relative norm tolerance for declaring a vector to "
        "lie in the kernel of H (epsilon) should be a "
        "small positive number in the range (0,1)");

  /// INITIALIZATION

  // The current estimate for the truncated-Newton update step s_k;
  // initialized to zero
  Vector s_k = 0 * g; // s_k and g must have the same dimensions

  // The residual at iteration k: r_k := H*s_k + g.  Note
  Vector r_k = g;

  // Preconditioned residual vector
  Vector v_k;

  // Lagrange multipliers for constraint preconditioning operator
  Multiplier lambda_k;

  // Update direction for s_k
  Vector p_k;

  // Working space to hold Hessian-vector products H*p_k
  Vector Hp_k;

  // Preconditioner return value
  if (!P) {
    // If no preconditioner was supplied then v_k === r_k
    v_k = r_k;
  } else {
    // Precondition the current residual vector r_k
    std::tie(v_k, lambda_k) = (*P)(r_k, args...);

    if (At) {
      // Following Section 6 of the paper "On the Solution of Equality-
      // Constrained Quadratic Programming Problems Arising in Optimization",
      // we extract the Lagrange multiplier estimates λ corresponding to the
      // solution of:
      //
      //   min |r - A'λ|_{M^{-1}}

      // and use these to subtract off the corresponding component A'λ (which,
      // by the Fundamental Theorem of Algebra, lies in the orthogonal
      // complement of ker(A)).  As described in the paper, the purpose of this
      // is to ensure that, when applying the preconditioning operator to
      // compute P(r) = (v, λ), the Lagrange multiplier estimate λ is small in
      // norm relative to v, so that the preconditioned search direction v can
      // be computed to high relative accuracy.
      r_k -= (*At)(lambda_k, args...);
    }
  } // preconditioning

  // Set initial search direction
  p_k = -v_k;

  // Value of inner product < s_k, M * p_k >; initially zero since s0 = 0
  Scalar sk_M_pk = 0;

  // Squared M-norm of current Newton step estimate s_k:  || s_k ||_M^2;
  // initially zero since s0 = 0
  Scalar sk_M_2 = 0;

  // Squared M-norm of current update step direction p_k: || p_k ||_M^2
  Scalar pk_M_2 = inner_product(r_k, v_k, args...);

  /// Useful cached variables

  // Squared radius of trust-region
  Scalar Delta_2 = Delta * Delta;

  // Magnitude of initial (preconditioned) residual in the norm specified by the
  // (optional) preconditioner P: r0_norm = sqrt(r0'*P*r0)
  Scalar r0_norm = sqrt(inner_product(r_k, v_k, args...));

  // Target norm of preconditioned residual after applying update step s_k
  Scalar target_rk_norm =
      r0_norm * std::min(kappa_fgr, std::pow(r0_norm, theta));

  Scalar alpha_k; // Steplength for full Newton step along search direction p_k
  Scalar beta_k;
  Scalar kappa_k; // Value of inner product < p_k, H*p_k >

  for (num_iterations = 0; num_iterations < max_iterations; ++num_iterations) {
    /// CHECK TERMINATION CRITERIA

    /// Termination criteria based upon predicted residual after applying the
    /// current update step s_k
    if (std::sqrt(inner_product(r_k, v_k, args...)) <= target_rk_norm)
      break;

    // Compute Hessian-vector product H*p_k for this iteration
    Hp_k = H(p_k, args...);

    /// Next, check termination criteria based upon check for negative curvature
    /// or overly-long steplengths

    // Compute kappa
    kappa_k = inner_product(p_k, Hp_k, args...);

    // Stopping condition based upon whether pk'Hpk = 0, up to numerical
    // tolerance(cf.Sec.7 of "On the Solution of Equality-Constrained
    // Quadratic Programming Problems Arising in Optimization ")
    if (sqrt(inner_product(Hp_k, Hp_k, args...)) /
            sqrt(inner_product(p_k, p_k, args...)) <
        epsilon) {

      // If control reaches this point in the function, the following two
      // conditions hold:
      //
      //  - The (preconditioned) residual v_k has M-norm greater than the
      //    target norm
      //
      //  - The current search direction lies in the kernel of the Hessian
      //    operator
      //
      //  In this case, we should follow the subspace spanned by p_k to the
      //  boundary of the trust-region radius
      if (inner_product(p_k, r_k, args...) < 0) {

        //  Multiply the current search direction by - 1 to ensure that it's a
        // direction of descent
        p_k *= -1;
        sk_M_pk *= -1;
      }

      // Compute sigma_k such that || s_k + sigma_k * p_k ||_M = Delta, i.e. so
      // that the next update terminates on the trust-region boundary
      Scalar sigma_k =
          (-sk_M_pk + sqrt(sk_M_pk * sk_M_pk + pk_M_2 * (Delta_2 - sk_M_2))) /
          pk_M_2;

      update_step_M_norm = Delta;

      s_k += sigma_k * p_k;
      return s_k;
    }

    // Compute (full) steplength alpha_k
    alpha_k = inner_product(r_k, v_k, args...) / kappa_k;

    // Compute norm of proposed (full) step
    Scalar skplus1_M_2 =
        sk_M_2 + 2 * alpha_k * sk_M_pk + alpha_k * alpha_k * pk_M_2;

    if ((kappa_k <= 0) || (skplus1_M_2 > Delta_2)) {
      /** Either p_k is a direction of negative curvature, or the full
       * (unrestricted) step along this direction leaves the trust-region.
       * In either case, we would like to rescale the stepsize alpha_k to
       * ensure that the proposed step terminates _on_ the trust-region
       * boundary, and then return the resulting step as our final answer
       */

      Scalar sigma_k =
          (-sk_M_pk + sqrt(sk_M_pk * sk_M_pk + pk_M_2 * (Delta_2 - sk_M_2))) /
          pk_M_2;

      update_step_M_norm = Delta;
      s_k += sigma_k * p_k;
      return s_k;
    }

    /// Call user-supplied function, if one was passed
    if (user_function && (*user_function)(num_iterations, g, H, P, At, s_k, r_k,
                                          v_k, p_k, alpha_k, args...)) {
      // User-defined stopping criterion just fired
      break;
    }

    /// UPDATE!  Compute values for next iteration

    // Update solution s_k -> s_kplus1
    s_k = s_k + alpha_k * p_k;

    // Update estimate for residual after applying s_k:  r_k -> r_kplus1
    r_k += alpha_k * Hp_k;

    // Compute preconditioned residual v_kplus1
    // Preconditioner return value
    if (!P) {
      // If no preconditioner was supplied then v_k === r_k
      v_k = r_k;
    } else {
      // Precondition the current residual vector r_k
      std::tie(v_k, lambda_k) = (*P)(r_k, args...);

      if (At) {
        // Following Section 6 of the paper "On the Solution of Equality-
        // Constrained Quadratic Programming Problems Arising in Optimization",
        // we extract the Lagrange multiplier estimates λ corresponding to the
        // solution of:
        //
        //   min |r - A'λ|_{M^{-1}}

        // and use these to subtract off the corresponding component A'λ (which,
        // by the Fundamental Theorem of Algebra, lies in the orthogonal
        // complement of ker(A)).  As described in the paper, the purpose of
        // this is to ensure that, when applying the preconditioning operator to
        // compute P(r) = (v, λ), the Lagrange multiplier estimate λ is small in
        // norm relative to v, so that the preconditioned search direction v can
        // be computed to high relative accuracy.
        r_k -= (*At)(lambda_k, args...);
      }
    } // preconditioning

    // Compute updated preconditioned inner product
    Scalar rk_vk = inner_product(r_k, v_k, args...);

    // Compute beta_k for *current* iterate (using *updated* values
    // r_kplus1 and v_kplus1 and *current* iterate value alpha_k, kappa_k)
    beta_k = rk_vk / (alpha_k * kappa_k);

    // Update inner products and norms
    sk_M_2 = skplus1_M_2;
    sk_M_pk = beta_k * (sk_M_pk + alpha_k * pk_M_2);
    pk_M_2 = rk_vk + beta_k * beta_k * pk_M_2;

    // Update search direction p_k
    p_k = -v_k + beta_k * p_k;

  } // for

  update_step_M_norm = sqrt(sk_M_2);
  return s_k;
}

} // namespace LinearAlgebra
} // namespace Optimization
