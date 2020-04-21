/** This header file provides implementations of the four basic subproblem
 * solvers required by the TRSQP optimization method:
 *
 * - compute_multiplier_updates(),
 * - compute_normal_step(),
 * - compute_tangential_step(),
 * - compute_primal_dual_step()
 *
 * The first three of these are based upon the subproblem solvers described in
 * the paper:
 *
 * "An Interior Point Algorithm for Large-Scale Nonlinear Programming" by R.H.
 * Byrd, M.E. Hribar, and J. Nocedal
 *
 * and require only the solution of linear systems of the form:
 *
 * [I     Ahat'][v] = [b]
 * [Ahat      0][y] = [c]
 *
 * where:
 *
 * Ahat := [Ae        0]
 *         [Ai  diag(s)]
 *
 * where Ax := (Ae, Ai) are the Jacobians of the (vector-valued) constraint
 * functions c(x) := (ce(x), ci(x)) describing the feasible set of the
 * optimization problem, and s is the auxiliary vector of slack variables for
 * the inequality constraints.
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "Optimization/Constrained/TRSQPTypes.h"
#include "TRSQPUtilityFunctions.h"

#include "Optimization/LinearAlgebra/IterativeSolvers.h"

namespace Optimization {
namespace Constrained {

/// Template functions for TRSQP subproblem solvers

/** This function computes and returns an *update* for the pair of Lagrange
 * multipliers lambda = (lambda_e, lambda_i) to minimize the norm of the
 * gradient of the Lagrangian of the barrier subproblem perturbed KKT conditions
 * (3.7) -- (3.8), i.e., that solve the linear least-squares problem:
 *
 * min_{delta_lambda} |Ahat'*delta_lambda + (gradfx, -mu*e) + Ahat'*lambda|^2
 *
 * (cf. eqs. (3.12)--(3.13) in the paper "An Interior Point Algorithm for
 * Large Scale Nonlinear Programming").  It does so by solving the augmented
 * linear system:
 *
 * M *  [  z   ]  = [ (-gradfx, mu*e) - Ahat'*lambda ]
 *      [lambda]    [        0                       ]
 *
 * where:
 *
 * M = [I    Ahat']
 *     [Ahat     0]
 *
 * and:
 *
 * Ahat :=[Ae       0   ]
 *        [Ai    diag(s)]
 *
 * (cf. eqs. (3.12)--(3.13) in the paper "An Interior Point Algorithm for
 * Large Scale Nonlinear Programming").
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Scalar = double,
          typename... Args>
Pair<EqVector, IneqVector> update_Lagrange_multipliers(
    const Vector &gradfx, const IneqVector &s,
    const Pair<EqVector, IneqVector> &lambda,
    const Pair<EqJacobian, IneqJacobian> &Ax, Scalar mu,
    const AugmentedSystemSolverFunction<Vector, EqVector, IneqVector,
                                        EqJacobian, IneqJacobian, Args...>
        &augmented_linear_system_solver,
    Args &... args) {

  // Construct vector mu*e
  IneqVector mue = s;
  for (size_t k = 0; k < s.dim(); ++k)
    mue(k) = mu;

  Pair<EqVector, IneqVector> b =
      Pair<EqVector, IneqVector>(-gradfx, mue) -
      compute_Ahat_transpose_product<Vector, EqVector, IneqVector, EqJacobian,
                                     IneqJacobian>(Ax, s, lambda);

  // Set second block to appropriate-sized vectors of zero
  Pair<EqVector, IneqVector> zero_vec(
      lambda.first.dim() > 0 ? 0 * lambda.first : EqVector(),
      lambda.second.dim() > 0 ? 0 * lambda.second : IneqVector());

  // Primal part of solution vector
  Pair<Vector, IneqVector> v;

  // Dual part of solution vector
  Pair<EqVector, IneqVector> delta_lambda;

  augmented_linear_system_solver(Ax, s, true, b, zero_vec, v, delta_lambda,
                                 args...);

  return delta_lambda;
}

/** This function computes and returns a normal update step that approximately
 * solves the following normal step subproblem for the barrier problem:
 *
 * min_{vx, vx}  | ce(x) +      Ae*vx      |
 *               | ci(x) + s +  Ai*vx + vs |
 *
 *          st.  |(vx, S^-1*vs)| <= zeta * Delta,
 *               vs >= -tau * s / 2
 *
 * (cf. eq. (3.19) of the paper "An Interior Point Algorithm for Large Scale
 * Nonlinear Programming").
 *
 * Letting ws := S^-1 * vs, we can rewrite the above problem as:
 *
 * min_{vx, ws}  | ce(x) +      Ae*vx          |
 *               | ci(x) + s +  Ai*vx + S * ws |
 *
 *            =  | [ ce(x)     ]  + Ahat [vx] |
 *               | [ ci(x) + s ]         [wx] |
 *
 *          st.  |(vx, ws)| <= zeta*Delta,
 *               ws >= -tau / 2
 *
 * (cf. eq. (3.20) in the paper).  This function approximately solves the
 * simplified version of the normal step subproblem written above by means of
 * the modified Powell's Dog-Leg procedure described in Section 3.2 of the paper
 * "An Interior Point Algorithm for Large Scale Nonlinear Programming", taking
 * advantage of the cached factorization of the augmented system matrix M to
 * efficiently compute the Newton step.
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Scalar = double,
          typename... Args>
Pair<EqVector, IneqVector> compute_normal_step(
    const Pair<EqVector, IneqVector> &cx, const IneqVector &s,
    const Pair<EqJacobian, IneqJacobian> &Ax, Scalar Delta, Scalar zeta,
    Scalar tau,
    const AugmentedSystemSolverFunction<Vector, EqVector, IneqVector,
                                        EqJacobian, IneqJacobian, Args...>
        &augmented_linear_system_solver,
    Args &... args) {

  // Candidate *SCALED* normal update vector vtilde = (vx, vtilde_s)
  Pair<Vector, IneqVector> vtilde;

  // Residual for subproblem constraints
  Pair<EqVector, IneqVector> r =
      compute_barrier_subproblem_constraint_residuals(cx, s);

  // Compute gradient direction vector for linearized constraint violation:
  //
  // L(vtilde) := ||Ahat*vtilde + r||^2
  Pair<Vector, IneqVector> g =
      compute_Ahat_transpose_product<Vector, EqVector, IneqVector, EqJacobian>(
          Ax, s, r);

  /// Compute the full (unrestricted) Newton step vN: this is the unique
  /// minimum-norm solution of A*vN + r = 0

  // A straightforward computation shows that this can be obtained from the
  // linear system:
  //
  // [I    Ahat'] [vN] = [ 0]
  // [Ahat    0 ] [y ] = [-r]

  // Set right-hand side vector
  Pair<Vector, IneqVector> zero_vec = 0 * g;

  Pair<Vector, IneqVector> vN;
  Pair<EqVector, IneqVector> y;

  // Solve for full Newton step using cached augmented system matrix
  augmented_linear_system_solver(Ax, s, false, zero_vec, -r, vN, y, args...);
  Scalar vN_norm = vN.norm();

  // Set fraction trust-region radius DeltaBar = zeta * Delta
  Scalar DeltaBar = zeta * Delta;

  /// Compute the longest possible steplength theta1 for vN to be feasible:
  ///
  /// - trust-region constraint | vN | <= DeltaBar
  /// - Elementwise slack positivity constraint vN,s >= -tau/2

  // Most-negative component of vN,s (if any)
  Scalar vNs_min = (vN.second.dim() > 0 ? vN.second.min() : 0);

  // Longest steplength that can be taken before violating
  // theta * vN,s >= -tau/2
  Scalar theta1_s =
      (vNs_min < 0 ? -tau / (2 * vNs_min) : std::numeric_limits<Scalar>::max());

  // Longest steplength that can be taken before violating the trust-region
  // constraint
  Scalar theta1_DeltaBar = DeltaBar / vN_norm;

  // Maximum feasible steplength for the Newton step
  Scalar theta1 = std::min(theta1_s, theta1_DeltaBar);

  if (theta1 >= 1)
    // The full Newton step is feasible, so just accept it directly
    vtilde = vN;

  else {

    /// Compute a dog-leg step
    Pair<Vector, IneqVector> vDL;

    // Compute full (unrestricted) Cauchy step vC: this is the minimizer
    // of f(alpha) := f(-alpha*g)

    // Compute steplength alpha for full Cauchy step along -g (cf. eq. (3.23)
    // in the paper)
    Scalar alpha =
        std::pow(g.norm(), 2) /
        std::pow(compute_Ahat_product<Vector, EqVector, IneqVector, EqJacobian,
                                      IneqJacobian>(Ax, s, g)
                     .norm(),
                 2);

    Pair<Vector, IneqVector> vC = -alpha * g;
    Scalar vC_norm = vC.norm();

    /// Find theta2 := max{ theta in (0, 1] | (1-theta)*vC + theta*vN
    /// feasible}
    ///                max{ theta in (0, 1] | theta*vC + theta*d is feasible }

    Scalar theta2 = -1;

    // Note that if vC => DeltaBar, then since theta1 < 1, both vC and vN lie
    // outside the trust-region radius DeltaBar.  Therefore, so does the
    // *ENTIRE* interpolation vDL(theta) := vC + theta*vN, as it is known that
    // the length of this path is *monotonically increasing* (cf. e.g. the
    // RISE2 paper).  Therefore, it is *necessary* that vC < DeltaBar in order
    // for theta2 as defined above to exist
    if (vC_norm < DeltaBar) {
      /// Compute the dog-leg interpolation step
      Pair<Vector, IneqVector> d = vN - vC; // Step from vC to vN

      // Compute maximum value of theta2 satisfying the trust-region
      // constraint
      // || vC + theta*d || <=
      Scalar vCd = vC.inner_product(d);
      Scalar vC2 = vC_norm * vC_norm;
      Scalar d2 = d.inner_product(d);

      Scalar theta2_max =
          (-vCd + sqrt(vCd * vCd + d2 * (DeltaBar * DeltaBar - vC2))) / d2;

      Scalar theta2_min = 0;

      // Ensure that this step maintains sufficient positivity of the
      // resulting auxiliary slack variables, if any
      for (size_t k = 0; k < s.dim(); ++k) {
        Scalar q = (-vC.second(k) - tau / 2) / d.second(k);
        if (d.second(k) > 0)
          theta2_min = std::max(theta2_min, q);
        else
          theta2_max = std::min(theta2_max, q);
      }

      if (theta2_min <= theta2_max) {
        // There is a non-empty interval of feasible values for theta, so
        // choose the maximum one
        theta2 = theta2_max;
      }
    } // Compute theta2

    if (theta2 > 0) {
      // If there is a feasible value of theta2
      vDL = (1 - theta2) * vC + theta2 * vN;
    } else {
      /// Find theta3 := max{ theta in (0, 1] | theta3*vC is feasible }
      // Most-negative component of vN,s (if any)
      Scalar vCs_min = (s.dim() > 0 ? vC.second.min() : 0);

      // Longest steplength that can be taken before violating vC,s >= -tau/2
      Scalar theta3_s = (vCs_min < 0 ? -tau / (2 * vCs_min)
                                     : std::numeric_limits<Scalar>::max());

      // Longest steplength that can be taken before violating the
      // trust-region constraint
      Scalar theta3_DeltaBar = DeltaBar / vC_norm;

      // Maximum feasible steplength for the Newton step
      Scalar theta3 = std::min(theta3_s, theta3_DeltaBar);
      vDL = theta3 * vC;
    }

    /// Compare dog-leg step with truncated normal step
    if ((r + compute_Ahat_product<Vector, EqVector, IneqVector, EqJacobian,
                                  IneqJacobian>(Ax, s, vDL))
            .norm() <
        (r + compute_Ahat_product<Vector, EqVector, IneqVector, EqJacobian,
                                  IneqJacobian>(Ax, s, theta1 * vN))
            .norm())
      vtilde = vDL;
    else
      vtilde = theta1 * vN;
  } // Compute dog-leg step

  /// Extract and return *UNSCALED* normal step v = (vx, vs)

  return Pair<Vector, IneqVector>(
      vtilde.first, (vtilde.second.dim() > 0 ? s.hadamard_product(vtilde.second)
                                             : vtilde.second));
}

/** This functions compute a tangential update step:
 *
 *   h := (wx, diag(S)*ws)
 *
 * where w = (wx, ws) is an approximate solution of the
 * following (scaled) tangential step subproblem (cf. eqs. (3.25)--(3.29) of the
 * paper "An Interior Point Algorithm for Large-Scale Nonlinear Programming")
 *
 *   min_w gradfx'*wx - mu*e'*ws + (G*vtilde)'*w + (1/2)w'*G*w
 *   st.   Ahat*w = 0
 *         ||w||^2 <= Delta^2 - ||vtilde||^2
 *         ws >= -tau*e - vtilde_s
 *
 * Here:
 *
 * - vtilde := (vtilde_x, vtilde_s) := (vx, diag(S)^-1* vs) and v = (vx, vs)
 *   is the (previously-computed) normal update step
 *
 * - G := [Hx                      0]
 *        [0   diag(S)*Sigma*diag(S)]
 *
 *   is the Hessian of the quadratic model objective q used for the
 *   tangential step subproblem
 *
 * - Hx is the Hessian of the Lagrangian:
 *
 *   L(x, lambda) := f(x) + lambda'*c(x)
 *
 *   of the original optimization problem with respect to x.
 *
 * - Sigma is the primal-dual approximation of the Hessian of the Lagrangian
 *   of the barrier subproblem with respect to s (cf. eq. (3.14) in the paper
 *   "An Interior Point Algorithm for Large-Scale Nonlinear Programming").
 *
 * This function approximately solves the tangential step subproblem written
 * above by means of a modified version of the Steihaug-Toint truncated
 * preconditioned projected conjugate-gradient algorithm.
 *
 * Arguments:
 *
 * - gradfx: gradient of objective f(x), evaluated at x
 * - s: Current value of auxiliary slack variables
 * - Ax:  Pair of Jacobians for the equality and inequality constraints,
 *   evaluated at x
 * - H:  Hessian of Lagrangian with respect to x
 * - Sigmas:  Primal-dual approximation of the Hessian of the Lagrangian wrt s
 * - v:  Normal update step
 * - mu:  Current value of barrier parameter
 * - Delta:  Trust-region radius
 * - tau:  Current admisible fraction-to-the-boundary for the slack update
 * - max_iterations:  Maximum admissible number of iterations for the
 *   Steihaug-Toint truncated preconditioned projected conjugate-gradient
 *   method used to computing the tangential update step
 * - kappa and theta:  Parameters defining the stopping tolerance for the
 *   residual in the Steihaug-Toint preconditioned conjugate-gradient method:
 *   termination occurs whenever the residual r_k, as measured in the P-norm
 *   determined by the constraint preconditioner P:
 *
 *     || r ||_P := sqrt(r'*P_x(r))
 *
 *   satisfies:
 *
 *     || r_k ||_P <= ||r_0||_P min [kappa, || r_0 ||_P^theta ],
 *
 * - tangential_step_M_norm:  Norm of the returned tangential update step in
 *   the M-norm determined by the constraint preconditioner P
 * - num_iters:  Actual number of iterations performed by the STPCG algorithm
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Hessian,
          typename Scalar = double, typename... Args>
Pair<Vector, IneqVector> compute_tangential_step(
    const Vector &gradfx, const IneqVector &s,
    const Pair<EqJacobian, IneqJacobian> &Ax, const Hessian &H,
    const IneqVector &Sigmas, const Pair<Vector, IneqVector> &v, Scalar mu,
    Scalar Delta, Scalar tau, size_t max_iterations, Scalar kappa, Scalar theta,
    const AugmentedSystemSolverFunction<Vector, EqVector, IneqVector,
                                        EqJacobian, IneqJacobian, Args...>
        &augmented_linear_system_solver,
    Scalar &tangential_step_M_norm, size_t &num_iterations, Args &... args) {

  // Some useful typedefs for this function
  typedef Pair<Vector, IneqVector> Primal;
  typedef Pair<EqVector, IneqVector> Dual;

  /// Construct SCALED normal slack updates vtilde = (vx, s^-1 * vs)
  IneqVector vtilde_s =
      (v.second.dim() > 0 ? v.second.hadamard_product(s.hadamard_inverse())
                          : v.second);

  /// Construct elements of the tangential step subproblem defined in eqs.
  /// (3.34)--(3.38) in the paper

  /// Construct 2nd (diagonal) block of Hessian matrix G:
  /// S * Sigma * S
  IneqVector SSigmaS =
      (s.dim() > 0 ? s.hadamard_product(Sigmas.hadamard_product(s)) : s);

  /// Construct gradient vector g = (gradfx + H*vx, -mu*e + SSigmaS*vtilde_s)
  Primal g(gradfx + H * v.first,
           s.dim() > 0 ? SSigmaS.hadamard_product(vtilde_s) : s);
  for (size_t k = 0; k < s.dim(); ++k)
    g.second(k) -= mu;

  /// Set up stuff required functions for STPCG!
  Optimization::LinearAlgebra::InnerProduct<Primal, Scalar>
      STPCG_inner_product = [](const Primal &v1, const Primal &v2) {
        // Just use the standard Eigen inner product for vectors
        return v1.inner_product(v2);
      };

  /// Construct Hessian operator G
  Optimization::LinearAlgebra::SymmetricLinearOperator<Primal> Gop =
      [&H, &SSigmaS](const Primal &v) -> Primal {
    return Primal(H * v.first, v.second.dim() > 0
                                   ? SSigmaS.hadamard_product(v.second)
                                   : v.second);
  };

  /// Construct transpose At of constraint operator A
  Optimization::LinearAlgebra::LinearOperator<Dual, Primal> Atop =
      [&Ax, &s](const Dual &lambda) -> Primal {
    return compute_Ahat_transpose_product<Vector, EqVector, IneqVector,
                                          EqJacobian, IneqJacobian>(Ax, s,
                                                                    lambda);
  };

  std::experimental::optional<
      Optimization::LinearAlgebra::LinearOperator<Dual, Primal>>
      At_arg(Atop);

  /// Construct constraint preconditioning operator P
  Dual zero_vec =
      0 *
      compute_A_product<Vector, EqVector, IneqVector, EqJacobian, IneqJacobian>(
          Ax, gradfx);

  Optimization::LinearAlgebra::STPCGPreconditioner<Primal, Dual> Pop =
      [&Ax, &s, &zero_vec, &augmented_linear_system_solver,
       &args...](const Primal &r) {
        std::pair<Primal, Dual> result;

        // Solve to perform preconditioning
        augmented_linear_system_solver(Ax, s, false, r, zero_vec, result.first,
                                       result.second, args...);

        return result;
      };

  std::experimental::optional<
      Optimization::LinearAlgebra::STPCGPreconditioner<Primal, Dual>>
      P_arg(Pop);

  /// Compute trust-region radius for tangential step

  // Compute norm of SCALED normal update:
  // vtilde = (vx, vtilde_s) = |(vx, S^-1*vs)|
  Scalar vtilde_norm = Primal(v.first, vtilde_s).norm();
  Scalar DeltaTan = sqrt(Delta * Delta - vtilde_norm * vtilde_norm);

  /// Construct a custom user function to cache the most recent iterate wplus
  /// that is feasible with respect to the bounds (3.38), and its
  /// corresponding search direction p, as described in the algorithm "PCG
  /// Procedure" in the paper.

  // Set initial feasible point: wtilde = 0
  Primal wtilde = 0 * g;
  // Set initial search direction: p =
  Primal p = -Pop(g).first;

  // Vector of lower-bounds on the right-hand side of  inequality  (3.38)
  IneqVector l = -vtilde_s;
  for (size_t k = 0; k < s.dim(); ++k)
    l(k) -= tau;

  Optimization::LinearAlgebra::STPCGUserFunction<Primal, Dual> user_func =
      [&wtilde, &p, &l](
          size_t k, const Primal &g,
          const Optimization::LinearAlgebra::SymmetricLinearOperator<Primal> &H,
          const std::experimental::optional<
              Optimization::LinearAlgebra::LinearOperator<
                  Primal, std::pair<Primal, Dual>>> &P,
          const std::experimental::optional<
              Optimization::LinearAlgebra::LinearOperator<Dual, Primal>> &At,
          const Primal &sk, const Primal &rk, const Primal &vk,
          const Primal &pk, Scalar alpha_k) {
        // If there are slack variables in the current problem, and the
        // current iterate sk satisfies the slack bound (3.38)
        if (sk.second.dim() > 0 && (sk.second - l).min() >= 0) {
          // Record current iterate sk and search direction wk
          wtilde = sk;
          p = pk;
        }
        return false; // Continue iterations
      };

  std::experimental::optional<
      Optimization::LinearAlgebra::STPCGUserFunction<Primal, Dual>>
      user_func_arg(user_func);

  /// RUN STPCG!
  Pair<Vector, IneqVector> wplus =
      Optimization::LinearAlgebra::STPCG<Primal, Dual, Scalar>(
          g, Gop, STPCG_inner_product, tangential_step_M_norm, num_iterations,
          DeltaTan, max_iterations, kappa, theta, P_arg, At_arg, user_func_arg);

  /// Extract and return final UNSCALED tangential update
  Primal w; // Final output

  if (s.dim() > 0) {
    // If inequality constraints are present, check whether the final iterate
    // returend by STPCG satisfies the bound (3.38)
    if ((wplus.second - l).min() >= 0) {
      // Final STPCG iterate satisfies the bound (3.38), so use this solution
      w.first = wplus.first;
      w.second =
          s.hadamard_product(wplus.second); // Return UNSCALED slack update
    } else {
      // Final STPCG iterate was *NOT* feasible.  Following "PCG Procedure",
      // extract the last feasible iterate wtilde and its corresponding search
      // direction p, and compute the largest steplength alpha such that
      // wtilde + alpha*p is feasible

      // Compute maximum admissible steplength for the trust-region constraint
      Scalar wtilde_p = wtilde.inner_product(p);
      Scalar wtilde_2 = wtilde.inner_product(wtilde);
      Scalar p2 = p.inner_product(p);
      Scalar alpha = (-wtilde_p + sqrt(wtilde_p * wtilde_p +
                                       p2 * (DeltaTan * DeltaTan - wtilde_2))) /
                     p2;

      // Compute maximum admissible steplength for the slack inequality
      // constraints
      for (size_t k = 0; k < s.dim(); ++k) {
        if (p.second(k) < 0) {
          // If p(n+k) < 0, compute the maximum steplength such that
          // wtilde(n+k) + alpha * p(n+k) >= l(k)
          alpha = std::min(alpha, (l(k) - wtilde.second(k)) / p.second(k));
        }
      }

      // Set w = wtilde + alpha * p
      w.first = wtilde.first + alpha * p.first;
      w.second = s.hadamard_product(wtilde.second + alpha * p.second);
    }
  } else {
    w.first = wplus.first;
    // mi == 0 => no auxiliary slacks to return
  }

  return w;
}

/** This function computes and returns the normal and tangential components of a
 * primal-dual Newton update step for the KKT system of the barrier subproblem.
 *
 * Arguments:
 *
 * - cx:  Current values of constraints c(x) = (ce(x), ci(x))
 * - s:  Current values of auxiliar (slack) variables
 * - gradfx:  Gradient of objective at x
 * - Ax:  Jacobians A(x) := (Ae(x), Ai(x)) of the constraint functions at x
 * - lambda:  Lagrange multipliers
 * - H:  Hessian of the Lagrangian:
 *
 *       L(x, lambda) := f(x) + lambda_e'ce(x) + lambda_i'ci(x)
 *
 *   with respect to x, evaluated at (x, lambda)
 * - Sigmas:  Model Hessian of the Lagrangian of the barrier subproblem wrt s
 * - mu:  Current barrier parameter
 * - new_system_matrix:  A Boolean value indicating whether the current KKT
 *   system matrix:
 *
 *     K := [W    Abar']
 *          [Abar     0]
 *
 *   where:
 *
 *     W := [H     0]
 *          [0 Sigma]
 *
 *     Abar := [Ae  0]
 *             [Ai  I]
 *
 *   is identical to the KKT system matrix supplied during the *previous* call
 *   to this function; setting this value enables the underlying linear system
 *   solvers to cache and reuse factorization of K in the case that multiple
 *   systems involving the same K must be solved
 *
 *  - kkt_system_solver is a user-supplied function that computes
 *    solutions d = (dz, dl) of linear systems of the form:
 *
 *      [W    Abar'][dz] = [b]
 *      [Abar     0][dl] = [c]
 *
 *  -  v_z and v_lambda are the primal and dual components of the normal part of
 *     the primal-dual update; these are solutions of the system:
 *
 *       [W    Abar'][v_z]      = [  0  ]
 *       [Abar     0][v_lambda] = [-c(z)]
 *
 *     where:
 *
 *        c(z) := [ce(x)    ]
 *                [ci(x) + s].
 *
 *     is the constraint function for the barrier subproblem
 *
 *   - w_z and w_lambda are the primal and dual components of the tangential
 *     part of the primal-dual update; these are solutions of the system:
 *
 *
 *       [W    Abar'][w_z]      = [-gradLz]
 *       [Abar     0][w_lambda] = [   0   ]
 *
 *     where:
 *
 *     L(z, lambda) := f(x) - mu sum_i log(s_i) + lambda'c(z)
 *
 *     is the Lagrangian of the barrier subproblem
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename EqJacobian, typename IneqJacobian, typename Hessian,
          typename Scalar = double, typename... Args>
void compute_primal_dual_update_components(
    const Pair<EqVector, IneqVector> &cx, const IneqVector &s,
    const Vector &gradfx, const Pair<EqJacobian, IneqJacobian> &Ax,
    const Pair<EqVector, IneqVector> &lambda, const Hessian &H,
    const IneqVector &Sigma, Scalar mu, bool new_kkt_system_matrix,
    const KKTSystemSolverFunction<Vector, EqVector, IneqVector, EqJacobian,
                                  IneqJacobian, Hessian, Args...>
        &kkt_system_solver,
    Pair<Vector, IneqVector> &v_z, Pair<EqVector, IneqVector> &v_lambda,
    Pair<Vector, IneqVector> &w_z, Pair<EqVector, IneqVector> &w_lambda,
    Args &... args) {

  // Compute right-hand side vectors gradLz and c(z) for the KKT system
  Pair<Vector, IneqVector> gradLz =
      compute_barrier_subproblem_gradient_of_Lagrangian(gradfx, s, Ax, lambda,
                                                        mu);

  Pair<EqVector, IneqVector> cz =
      compute_barrier_subproblem_constraint_residuals(cx, s);

  // Compute components of normal update step
  kkt_system_solver(H, Sigma, Ax, new_kkt_system_matrix, 0 * gradLz, -cz, v_z,
                    v_lambda, args...);

  // Compute components of tangential update step
  kkt_system_solver(H, Sigma, Ax, false, -gradLz, 0 * cz, w_z, w_lambda,
                    args...);
}

} // namespace Constrained
} // namespace Optimization
