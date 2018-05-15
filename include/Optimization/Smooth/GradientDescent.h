/** This header file provides a lightweight template function implementing
 * gradient descent on a Riemannian manifold.
 *
 * David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <experimental/optional>
#include <iostream>
#include <limits>
#include <math.h>

#include "Optimization/Smooth/Concepts.h"
#include "Optimization/Util/Stopwatch.h"

namespace Optimization {
namespace Smooth {

/** An alias template for a user-definable function that can be
 * used to access various interesting bits of information about the internal
 * state of a gradient descent algorithm as it runs.  More
 * precisely, this function is called at the end of each iteration, and is
 * provided access to the following quantities:
 *
 * i: index of current iteration
 * t: total elapsed computation time at the *start* of the current iteration
 * x: iterate at the *start* of the current iteration
 * f: objective value at x
 * g: Riemannian gradient at x
 * h: the update step computed during this iteration
 * df: decrease in objective value obtained by applying the update step
 */
template <typename Variable, typename Tangent, typename... Args>
using GradientDescentUserFunction =
    std::function<void(unsigned int i, double t, const Variable &x, double f,
                       const Tangent &g, const Tangent &h, double df,
                       Args &... args)>;

/** A lightweight struct containing a few additional algorithm-specific
 * configuration parameters for a gradient descent optimization method
 * (cf. Section 4.2 of "Optimization Algorithms on Matrix Manifolds") */
struct GradientDescentParams : public SmoothOptimizerParams {

  // Initial trial stepsize for the Armijo linesearch (must be > 0)
  double alpha = 1.0;

  // Shrinkage factor for the backtracking linesearch (must be in (0,1)).
  double beta = .5;

  // Sufficient fractional decrease for step acceptance (must be in (0,1)).
  double sigma = .5;

  // Maximum number of linesearch iterations to attempt
  unsigned int max_ls_iterations = 100;
};

/** A set of status flags indicating the stopping criterion that triggered
* algorithm termination */
enum class GradientDescentStatus {
  /** The algorithm obtained a solution satisfying the gradient tolerance */
  GRADIENT,

  /** The algorithm terminated because the relative decrease in function value
     obtained after the last accepted update was less than the specified
     tolerance */
  RELATIVE_DECREASE,

  /** The algorithm terminated because the norm of the last accepted update step
     was less than the specified tolerance */
  STEPSIZE,

  /** The algorithm terminated because linesearch failed to determine a stepsize
     along the gradient direction that generated a sufficient decrease in
     function value */
  LINESEARCH,

  /** The algorithm exhausted the allotted number of major (outer) iterations */
  ITERATION_LIMIT,

  /** The algorithm exhausted the allotted computation time */
  ELAPSED_TIME,
};

/** A useful struct used to hold the output of a gradient descent optimization
 * method */
template <typename Variable>
struct GradientDescentResult : public SmoothOptimizerResult<Variable> {

  GradientDescentStatus status;

  // Number of linesearch iterations performed during each iteration of the
  // algorithm
  std::vector<unsigned int> linesearch_iterations;
};

template <typename Variable, typename Tangent, typename... Args>
GradientDescentResult<Variable>
GradientDescent(const Objective<Variable, Args...> &f,
                const VectorField<Variable, Tangent, Args...> &grad_f,
                const RiemannianMetric<Variable, Tangent, Args...> &metric,
                const Retraction<Variable, Tangent, Args...> &retract,
                const Variable &x0, Args &... args,
                const GradientDescentParams &params = GradientDescentParams(),
                const std::experimental::optional<
                    GradientDescentUserFunction<Variable, Tangent, Args...>>
                    &user_function = std::experimental::nullopt) {
  /// Declare and initialize some useful variables

  // Square root of machine precision for doubles
  double sqrt_eps = sqrt(std::numeric_limits<double>::epsilon());

  // Current and proposed next iterates
  Variable x, x_proposed;

  // Gradient at the current iterate
  Tangent grad_f_x;

  // Norm of the Riemannian gradient at the current iterate
  double grad_f_x_norm;

  // Proposed update vector
  Tangent h;

  // Norm of update step
  double h_norm;

  // Number of linesearches performed during the current iteration
  unsigned int ls_iters = 0;

  // Function values at the current iterate proposed iterates, respectively
  double f_x, f_x_proposed;

  // Decrease in function value between subsequent iterations
  double df;

  // Relative decrease in function value between the current and proposed
  // iterates
  double relative_decrease;

  // Some useful constants for display purposes

  // Field with for displaying outer iterations
  unsigned int iter_field_width = floor(log10(params.max_iterations)) + 1;

  // Field width for displaying linesearch iterations
  unsigned int linesearch_iter_field_width =
      floor(log10(params.max_ls_iterations)) + 1;

  // Output struct
  GradientDescentResult<Variable> result;
  result.status = GradientDescentStatus::ITERATION_LIMIT;

  /// INITIALIZATION

  // Set initial iterate and function value
  x = x0;
  f_x = f(x, args...);

  // Compute gradient
  grad_f_x = grad_f(x, args...);
  grad_f_x_norm = sqrt(metric(x, grad_f_x, grad_f_x, args...));

  if (params.verbose) {
    // Set display options for real-valued quantities
    std::cout << std::scientific;
    std::cout.precision(params.precision);
  }
  if (params.verbose) {
    std::cout << "Gradient descent optimization: " << std::endl << std::endl;
  }

  // Start clock
  auto start_time = Stopwatch::tick();

  for (unsigned int iteration = 0; iteration < params.max_iterations;
       iteration++) {
    double elapsed_time = Stopwatch::tock(start_time);

    if (elapsed_time > params.max_computation_time) {
      result.status = GradientDescentStatus::ELAPSED_TIME;
      break;
    }

    // Record output
    result.time.push_back(elapsed_time);
    result.objective_values.push_back(f_x);
    result.gradient_norms.push_back(grad_f_x_norm);

    if (params.log_iterates)
      result.iterates.push_back(x);

    if (params.verbose) {
      std::cout << "Iter: ";
      std::cout.width(iter_field_width);
      std::cout << iteration << ", time: " << elapsed_time << ", f: ";
      std::cout.width(params.precision + 7);
      std::cout << f_x << ", |g|: " << grad_f_x_norm;
    }

    // Test gradient-based stopping criterion
    if (grad_f_x_norm < params.gradient_tolerance) {
      result.status = GradientDescentStatus::GRADIENT;
      break;
    }

    /// Armijo linesearch

    // Armijo stepsize
    // Set t_A here so that t_A = alpha when executing the first iteration of
    // the following do-while
    double t_A = params.alpha / params.beta;
    ls_iters = 0;

    bool accept = false;
    do {
      ls_iters++;
      // Shrink stepsize
      t_A *= params.beta;

      // Compute updated search step and proposed iterate
      h = -t_A * grad_f_x;
      x_proposed = retract(x, h, args...);

      // Compute function value at the proposed iterate and its improvement over
      // current iterate
      f_x_proposed = f(x_proposed, args...);
      df = f_x - f_x_proposed;

      accept = (df > params.sigma * t_A * grad_f_x_norm * grad_f_x_norm);

    } while ((!accept) && (ls_iters < params.max_ls_iterations));

    if (params.verbose) {
      std::cout << ", ls iters: ";
      std::cout.width(linesearch_iter_field_width);
      std::cout << ls_iters;
    }

    // Was linesearch successful?
    if (!accept) {
      result.status = GradientDescentStatus::LINESEARCH;
      break;
    }

    /// Iterate accepted!

    // Compute norm of update step
    h_norm = t_A * grad_f_x_norm;

    relative_decrease = df / (fabs(f_x) + sqrt_eps);

    // Record output
    result.linesearch_iterations.push_back(ls_iters);
    result.update_step_norm.push_back(h_norm);

    // Call the user-supplied function to provide access to internal algorithm
    // state
    if (user_function)
      (*user_function)(iteration, elapsed_time, x, f_x, grad_f_x, h, df,
                       args...);

    // Display output, if requested
    if (params.verbose) {
      std::cout << ", |h|: " << h_norm << ", df: " << df;
    }

    /// Update cached variables
    x = x_proposed;
    f_x = f_x_proposed;

    // Update gradient
    grad_f_x = grad_f(x, args...);
    grad_f_x_norm = sqrt(metric(x, grad_f_x, grad_f_x, args...));

    // Test additional stopping criteria
    if (relative_decrease < params.relative_decrease_tolerance) {
      result.status = GradientDescentStatus::RELATIVE_DECREASE;
      break;
    }

    if (h_norm < params.stepsize_tolerance) {
      result.status = GradientDescentStatus::STEPSIZE;
      break;
    }

    if (params.verbose)
      std::cout << std::endl;

  } // gradient descent iteration

  result.elapsed_time = Stopwatch::tock(start_time);

  // Record output
  result.x = x;
  result.f = f_x;
  result.grad_f_x_norm = grad_f_x_norm;

  if (params.verbose) {
    std::cout << std::endl
              << std::endl
              << "Optimization finished!" << std::endl;

    // Print the reason for termination
    switch (result.status) {
    case GradientDescentStatus::GRADIENT:
      std::cout << "Found first-order critical point! (Gradient norm: "
                << grad_f_x_norm << ")" << std::endl;
      break;
    case GradientDescentStatus::RELATIVE_DECREASE:
      std::cout
          << "Algorithm terminated due to insufficient relative decrease: "
          << relative_decrease << " < " << params.relative_decrease_tolerance
          << std::endl;
      break;
    case GradientDescentStatus::STEPSIZE:
      std::cout
          << "Algorithm terminated due to excessively small step size: |h| = "
          << h_norm << " < " << params.stepsize_tolerance << std::endl;
      break;
    case GradientDescentStatus::LINESEARCH:
      std::cout << "Algorithm terminated due to linesearch's inability to find "
                   "a stepsize with sufficient decrease"
                << std::endl;
      break;
    case GradientDescentStatus::ITERATION_LIMIT:
      std::cout << "Algorithm exceeded maximum number of outer iterations"
                << std::endl;
      break;
    case GradientDescentStatus::ELAPSED_TIME:
      std::cout << "Algorithm exceeded maximum allowed computation time: "
                << result.elapsed_time << " > " << params.max_computation_time
                << std::endl;
      break;
    }

    std::cout << "Final objective value: " << result.f << std::endl;
    std::cout << "Total elapsed computation time: " << result.elapsed_time
              << " seconds" << std::endl
              << std::endl;
  }

  return result;
}

/** This next function provides a convenient specialization of the gradient
 * descent interface for the (common) use case of optimization over Euclidean
 * spaces */

template <typename Vector, typename... Args>
using EuclideanGradientDescentUserFunction =
    GradientDescentUserFunction<Vector, Vector, Args...>;

template <typename Vector, typename... Args>
GradientDescentResult<Vector> EuclideanGradientDescent(
    const Objective<Vector, Args...> &f,
    const EuclideanVectorField<Vector, Args...> grad_f, const Vector &x0,
    Args &... args,
    const GradientDescentParams &params = GradientDescentParams(),
    const std::experimental::optional<
        EuclideanGradientDescentUserFunction<Vector, Args...>> &user_function =
        std::experimental::nullopt) {

  /// Run gradient descent using these Euclidean operators
  return GradientDescent<Vector, Vector, Args...>(
      f, grad_f, EuclideanMetric<Vector, Args...>,
      EuclideanRetraction<Vector, Args...>, x0, args..., params, user_function);
}
} // Smooth
} // Optimization
