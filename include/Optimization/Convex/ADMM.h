/** This header file provides several lightweight alias templates and template
 * classes implementing the Alternating Direction Method of Multipliers (ADMM)
 * algorithm for solving convex minimization problems of the form:
 *
 * min f(x) + g(y)
 *
 * s.t. Ax + By = c
 *
 * via operator splitting.  This implementation is based upon the one described
 * in Section 3.1 of "Distributed Optimization and Statistical Learning via the
 * Alternating Direction Method of Multipliers", by S. Boyd, N. Parikh, E. Chu,
 * B. Peleato, and J. Eckstein.
 *
 * Copyright (C) 2018 by David M. Rosen (drosen2000@gmail.com)
 */

#pragma once

#include "Optimization/Convex/Concepts.h"
#include "Optimization/Util/Stopwatch.h" // Useful timing functions

#include <algorithm>
#include <cmath>
#include <iostream>

namespace Optimization {
namespace Convex {

/** Alias templates for functions that return the minimizers of the augmented
 * Lagrangian:
 *
 * L_rho(x, y, lambda) := f(x) + g(y) + lambda' * (Ax + By - c)
 *                          + (rho / 2) * |Ax + By - c |_2^2
 *
 * considered as a function of its first and second arguments, respectively.
 */

template <typename VariableX, typename VariableY, typename VariableR,
          typename... Args>
using AugLagMinX =
    std::function<VariableX(const VariableX &x, const VariableY &y,
                            const VariableR &lambda, double rho,
                            Args &... args)>;

template <typename VariableX, typename VariableY, typename VariableR,
          typename... Args>
using AugLagMinY =
    std::function<VariableY(const VariableX &x, const VariableY &y,
                            const VariableR &lambda, double rho,
                            Args &... args)>;

struct ADMMParams : public OptimizerParams {

  /** (Initial) value of penalty parameter rho */
  double rho = 1.0;

  /** Termination criteria:
   *
   * We employ a termination criterion based upon the primal and dual residuals:
   *
   * r_k := Ax + By - c
   * s_k := rho * A^t * B * (y_k - y_{k-1})
   *
   * as suggested in Section 3.3.1 of "Distributed Optimization and Statistical
   * Learning via the Alternating Direction Method of Multipliers"; namely, we
   * define combined (absolute + relative) primal and dual residual stopping
   * tolerances eps_pri_k and eps_dual_k at iteration k according to:
   *
   * eps_pri_k := eps_abs_pri + eps_rel * max { |Ax_k|_2, |By_k|_2, |c|_2 }
   *
   * eps_dual_k := eps_abs_dual+ eps_rel * |A^t lambda_k|_2
   *
   * where eps_abs_pri, eps_abs_dual, and eps_rel are user-supplied tolerances,
   * and terminate if:
   *
   * |r_k|_2 <= eps_pri_k AND |s_k|_2 <= eps_dual_k
   */

  /** Absolute primal stopping tolerance */
  double eps_abs_pri = 1e-2;

  /** Absolute dual stopping tolerance */
  double eps_abs_dual = 1e-2;

  /** Relative stopping tolerance */
  double eps_rel = 1e-3;
};

/** Enum type that describes the termination status of the algorithm */
enum ADMMStatus { RESIDUAL_TOLERANCE, ITERATION_LIMIT, ELAPSED_TIME };

/** A useful struct used to hold the output of the ADMM algorithm */
template <typename VariableX, typename VariableY>
struct ADMMResult : OptimizerResult<std::pair<VariableX, VariableY>> {

  /** The stopping condition that triggered algorithm termination */
  ADMMStatus status;

  /** Primal residual at the end of each iteration */
  std::vector<double> primal_residuals;

  /** Dual residual at the end of each iteration */
  std::vector<double> dual_residuals;
};

template <typename VariableX, typename VariableY, typename VariableR,
          typename... Args>
ADMMResult<VariableX, VariableY>
ADMM(const AugLagMinX<VariableX, VariableY, VariableR, Args...> &minLx,
     const AugLagMinY<VariableX, VariableY, VariableR, Args...> &minLy,
     const LinearOperator<VariableX, VariableR, Args...> &A,
     const LinearOperator<VariableY, VariableR, Args...> &B,
     const LinearOperator<VariableR, VariableX, Args...> &At,
     const InnerProduct<VariableX, Args...> inner_product_x,
     const InnerProduct<VariableR, Args...> inner_product_r, const VariableR &c,
     const VariableX &x0, const VariableY &y0, Args... args,
     const ADMMParams &params = ADMMParams()) {

  /// Declare some useful variables

  // Current iterate for X
  VariableX x;

  // Current iterate for Y
  VariableY y;

  // Previous iterate for Y (used in the computation of the dual residual)
  VariableY y_prev;

  // Current iterate for dual variable (Lagrange multiplier) lambda
  VariableR lambda;

  // Current value of augmented Lagrangian penalty parameter
  double rho;

  // Primal residual vector
  VariableR r;

  // Dual residual vector
  VariableX s;

  // Primal and dual residuals
  double primal_residual, dual_residual;

  // Cache variables for intermediate products Ax, By
  VariableR Ax, By;

  double c_norm = sqrt(inner_product_r(c, c, args...));

  /// Output struct
  ADMMResult<VariableX, VariableY> result;
  result.status = ITERATION_LIMIT;

  /// INITIALIZATION
  x = x0;
  y = y0;
  y_prev = y0;
  rho = params.rho;
  // Compute initial value of dual variable lambda
  lambda = rho * (A(x) + B(y) - c);

  if (params.verbose) {
    std::cout << std::scientific;
    std::cout.precision(params.precision);
  }
  // Field width for displaying outer iterations
  unsigned int iter_field_width = floor(log10(params.max_iterations)) + 1;

  if (params.verbose)
    std::cout << "ADMM: " << std::endl << std::endl;

  /// ITERATE!
  auto start_time = Stopwatch::tick();
  for (unsigned int i = 0; i < params.max_iterations; ++i) {

    /// ADMM ITERATION
    // Update x by minimizing augmented Lagrangian with respect to x
    x = minLx(x, y, lambda, rho, args...);

    // Update y by minimizing augmented Lagrangian with respect to y
    y = minLy(x, y, lambda, rho, args...);

    // Compute primal residual vector
    Ax = A(x);
    By = B(y);
    r = Ax + By - c;

    // Update dual variable lambda
    lambda = lambda + rho * r;

    // Compute dual residual vector
    s = rho * At(B(y - y_prev, args...), args...);

    // Compute primal and dual residual norms
    primal_residual = sqrt(inner_product_r(r, r, args...));
    dual_residual = sqrt(inner_product_x(s, s, args...));

    double elapsed_time = Stopwatch::tock(start_time);

    /// Display output for this iteration

    // Display information about this iteration, if requested
    if (params.verbose) {
      std::cout << "Iter: ";
      std::cout.width(iter_field_width);
      std::cout << i << ", time: " << elapsed_time << ", primal residual: ";
      std::cout.width(params.precision + 7);
      std::cout << primal_residual << ", dual residual";
      std::cout.width(params.precision + 7);
      std::cout << dual_residual << std::endl;
    }

    /// Record output
    result.time.push_back(elapsed_time);
    result.primal_residuals.push_back(primal_residual);
    result.dual_residuals.push_back(dual_residual);

    if (params.log_iterates)
      result.iterates.emplace_back(x, y);

    /// TEST STOPPING CRITERIA

    /// Compute primal and dual stopping tolerances
    // Primal stopping tolerance
    double Ax_norm = sqrt(inner_product_r(Ax, Ax, args...));
    double By_norm = sqrt(inner_product_r(By, By, args...));
    double eps_primal =
        params.eps_abs_pri +
        params.eps_rel * std::max<double>({Ax_norm, By_norm, c_norm});

    // Dual stopping tolerance
    const VariableX At_lambda = At(lambda, args...);
    double At_lambda_norm =
        sqrt(inner_product_x(At_lambda, At_lambda, args...));
    double eps_dual = params.eps_abs_dual + params.eps_rel * At_lambda_norm;

    // Test residual-based stopping criterion
    if ((primal_residual < eps_primal) && (dual_residual < eps_dual)) {
      result.status = RESIDUAL_TOLERANCE;
      break;
    }

    // Test elapsed-time-based stopping criterion
    if (elapsed_time > params.max_computation_time) {
      result.status = ELAPSED_TIME;
      break;
    }

    /// CACHE PARAMETERS FOR NEXT ITERAITON
    y_prev = y;

  } // ADMM ITERATIONS

  /// RECORD FINAL OUTPUT
  result.x = {x, y};
  result.elapsed_time = Stopwatch::tock(start_time);

  /// Print final output, if requested
  if (params.verbose) {
    std::cout << std::endl
              << std::endl
              << "Optimization finished!" << std::endl;

    // Print the reason for termination
    switch (result.status) {
    case RESIDUAL_TOLERANCE:
      std::cout << "Found minimizer!  Primal residual: " << primal_residual
                << ", dual residual: " << dual_residual << std::endl;
      break;
    case ITERATION_LIMIT:
      std::cout << "Algorithm exceeded maximum number of outer iterations"
                << std::endl;
      break;
    case ELAPSED_TIME:
      std::cout << "Algorithm exceeded maximum allowed computation time: "
                << result.elapsed_time << " > " << params.max_computation_time
                << std::endl;
      break;
    }
  } // if(verbose)

  return result;
}

/** This next function provides a convenient specialization of the ADMM
 * interface for the (common) use case in which a single data type is used to
 * represent each variable */

template <typename Variable, typename... Args>
ADMMResult<Variable, Variable>
ADMM(const AugLagMinX<Variable, Variable, Variable, Args...> &minLx,
     const AugLagMinY<Variable, Variable, Variable, Args...> &minLy,
     const LinearOperator<Variable, Variable, Args...> &A,
     const LinearOperator<Variable, Variable, Args...> &B,
     const LinearOperator<Variable, Variable, Args...> &At,
     const InnerProduct<Variable, Args...> inner_product, const Variable &c,
     const Variable &x0, const Variable &y0, Args... args,
     const ADMMParams &params = ADMMParams()) {
  return ADMM<Variable, Variable, Variable, Args...>(
      minLx, minLy, A, B, At, inner_product, inner_product, c, x0, y0, args...,
      params);
}

} // namespace Convex
} // namespace Optimization
