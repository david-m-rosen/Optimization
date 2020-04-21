/** This header file provides a set of concepts that are useful for implementing
 * nonlinear programming methods for constrained optimization problems of the
 * following form:
 *
 * min_x  f(x)
 *   s.t. ce(x) =  0
 *        ci(x) <= 0
 *
 * where f, ce, and ci are continuously-differentiable functions
 *
 * For more information, please consult the excellent references:
 *
 * "Numerical Optimization", 2nd edition, by Nocedal and Wright
 *
 * "Trust-Region Methods" by Conn, Gould and Toint
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "Optimization/Base/Concepts.h"
#include "Optimization/Constrained/Pair.h"

#include <cmath>

/** Useful typedefs for implementing templated/parameterized nonlinear
 * programming methods.
 *
 * In the following definitions:
 *
 * - Vector is the type of the argument x of the objective function  f(x)
 *   (i.e., this is the type of variable to be optimized).
 *
 * - EqVector and IneqVector are the return types of the (vector-valued)
 *   equality and inequality constraint functions ce(x) and ci(x), respectively.
 *
 * Each of Vector, EqVector and IneqVector must model elements of an inner
 * product space; in particular, this means that they must support operations of
 * addition, subtraction, and scalar multiplication.  In addition, these types T
 * must implement member functions matching the following signatures:
 *
 *     - size_t dim() const:  Return the dimension of the vector (must return
 *       0 if vector is empty or unintialized)
 *
 *     - Scalar max() const:  Return the maximum element of the vector
 *
 *     - T max(const Scalar & s) const:  Return the vector whose ith
 *       element is given by max(*this_i, s)
 *
 *     - T max(const T& w) const:  Return the vector whose ith element
 *       is max(*this_i, w_i)
 *
 *     - Scalar min() const:  Return the minimum element of the vector
 *
 *     - T hadamard_product(const T& w) const:  Return the Hadamard
 *       (elementwise) product of *this and w
 *
 *     - T hadamard_inverse() const:  Return the Hadamard (elementwise)
 *       multiplicative inverse of the vector
 *
 *     - Scalar inner_product(const T& w):  Return the inner product of *this
 *       and w
 *
 *     - Scalar norm() const:  Return the norm of the vector
 *
 * - EqJacobian and IneqJacobian are types that are used to represent the
 *   Jacobians (derivatives) of the EqVector- and IneqVector-valued equality and
 *   inequality constraint functions, respectively.  These types must support
 *   operations of multiplication with EqVectors and IneqVectors, respectively,
 *   and a transpose() method.
 *
 * - Hessian is the type used to represent the (symmetric) Hessian of the
 *   Lagrangian L(x, lambda) := f(x) + <lambda_e, ce(x)> + <lambda_i, ci(x)>
 *   with respect to x.  It should support the operation of scalar
 *   multiplication with Vectors
 *
 * - Args is a user-definable variadic template parameter whose instances will
 *   be passed into each of the functions required to run the optimization
 *   algorithm; this enables e.g. the use of objectives accepting non-optimized
 *   parameters of arbitrary types, and/or the use of arbitrary user-defined
 *   cache variables in order to store and reuse intermediate results in
 *   different parts of the computation.
 */

namespace Optimization {

namespace Constrained {

/** An alias template for a function accepting as input an argument of type
 * Vector, and returning an argument of the same type. */
template <typename Vector, typename... Args>
using VectorFunction = std::function<Vector(const Vector &x, Args &... args)>;

/** An alias template for a function accepting as input a Vector, and returning
 * a Pair comprised of an element of type X and an element of type Y */
template <typename Vector, typename X, typename Y, typename... Args>
using PairFunction = std::function<Pair<X, Y>(const Vector &x, Args &... args)>;

/** An alias template for a function that accepts as input a Vector and a Pair
 * comprised of an element of type X and an element of type Y, and returning an
 * element of type Hessian */
template <typename Vector, typename X, typename Y, typename Hessian,
          typename... Args>
using HessianFunction = std::function<Hessian(
    const Vector &x, const Pair<X, Y> &lambda, const Args &... args)>;

/** A lightweight struct containing configuration parameters
 * for a nonlinear programming method */
template <typename Scalar = double>
struct NLPOptimizerParams : public OptimizerParams {

  /// Additional termination criteria for smooth optimization methods

  // Infeasibility tolerance
  Scalar infeasibility_tolerance = 1e-6;

  // Stopping tolerance for norm of the gradient of the Lagrangian wrt x
  Scalar gradient_tolerance = 1e-6;

  // Stopping tolerance for the norm of the complementarity residuals
  Scalar complementarity_tolerance = 1e-6;
};

/** A template struct containing the output of a nonlinear programming
 * optimization method
 */
template <typename Vector, typename EqVector, typename IneqVector,
          typename Scalar = double>
struct NLPOptimizerResult : public OptimizerResult<Vector, Scalar> {

  // Pair of Lagrange multipliers for the constraints
  Pair<EqVector, IneqVector> lambda;

  // The norm of the constraint violation at x
  Scalar infeas_norm;

  // The norm of the gradient of the Lagrangian wrt x
  Scalar grad_Lx_norm;

  // The L2 norm of the complementarity residuals
  Scalar complementarity_norm;

  // The vector of norms of the constraint violations at the *start* of each
  // iteration
  std::vector<Scalar> infeas_norms;

  // The vector of norms of the gradient of the Lagrangian wrt x at the *start*
  // of each iteration
  std::vector<Scalar> grad_Lx_norms;

  // The vector of norms of the complementarity residuals at the *start* of each
  // iteration
  std::vector<Scalar> complementarity_norms;
};

} // namespace Constrained
} // namespace Optimization
