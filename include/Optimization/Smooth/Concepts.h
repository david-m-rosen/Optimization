/** This header file provides a set of concepts that are useful for
 * implementing smooth numerical optimization methods on Riemannian manifolds.
 * For more information, please consult the excellent references:
 *
 * "Trust-Region Methods on Riemannian Manifolds" by Absil, Baker and Gallivan
 *
 * "Optimization Algorithms on Matrix Manifolds" by Absil, Mahony and Sepulchre
 *
 * Copyright (C) 2017 - 2018 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include "Optimization/Base/Concepts.h"

namespace Optimization {

namespace Smooth {

/** Useful typedefs for implementing templated/parameterized smooth numerical
 * optimization methods on Riemannian manifolds.
 *
 * In the following definitions:
 *
 * - Variable is the type of the argument of the objective function (i.e., this
 * is the type of variable to be optimized).
 *
 * - Tangent is the type used to represent elements of the tangent space at the
 * current iterate; typically, these will be Eigen vectors or matrices.
 *
 * - Args is a user-definable variadic template parameter whose instances will
 * be passed into each of the functions required to run the optimization
 * algorithm (objective, quadratic model constructor, linear operators,
 * retractions, and acceptance function); this enables e.g. the use of
 * objectives accepting non-optimized parameters of arbitrary types, and/or the
 * use of arbitrary user-defined cache variables in order to store and reuse
 * intermediate results in different parts of the computation.
 */

/** An alias template for a vector field on a smooth manifold M; this assigns to
 * each point X in M an element of the tangent space T_X(M) */
template <typename Variable, typename Tangent, typename... Args>
using VectorField = std::function<Tangent(const Variable &X, Args &... args)>;

/** An alias template for a linear operator H : T_X -> T_X on the tangent space
 * T_X of a manifold M at the point X in M */
template <typename Variable, typename Tangent, typename... Args>
using LinearOperator =
    std::function<Tangent(const Variable &X, const Tangent &V, Args &... args)>;

/** An alias template for a function that constructs a linear operator
 * A : T_X(M) -> T_X(M) on the tangent space of a manifold M at X */
template <typename Variable, typename Tangent, typename... Args>
using LinearOperatorConstructor =
    std::function<LinearOperator<Variable, Tangent, Args...>(const Variable &X,
                                                             Args &... args)>;

/** An alias template for a function that accepts as input a Variable X,
 * and returns the gradient and Hessian operator that determine the local
 * quadratic model of an objective function f(X) on the tangent space T_X. */
template <typename Variable, typename Tangent, typename... Args>
using QuadraticModel =
    std::function<void(const Variable &X, Tangent &gradient,
                       LinearOperator<Variable, Tangent, Args...> &Hessian,
                       Args &... args)>;

/** An alias template for a Riemannian metric; this assigns to each tangent
 * space T_X an inner product g : T_X x T_X -> R. */
template <typename Variable, typename Tangent, typename Scalar = double,
          typename... Args>
using RiemannianMetric = std::function<Scalar(
    const Variable &X, const Tangent &V1, const Tangent &V2, Args &...)>;

/** An alias template for a retraction operator: a std::function that accepts as
 * input a variable X and a tangent vector V from the tangent space at X, and
 * returns an 'updated' value of X obtained by moving along the underlying
 * manifold away from X in the direction of V. */
template <typename Variable, typename Tangent, typename... Args>
using Retraction = std::function<Variable(
    const Variable &X, const Tangent &update, Args &... args)>;

/** A lightweight struct containing configuration parameters
 * for a Riemannian optimization method */
template <typename Scalar = double>
struct SmoothOptimizerParams : public OptimizerParams {

  /// Additional termination criteria for smooth optimization methods

  // Stopping tolerance for norm of the Riemannian gradient
  Scalar gradient_tolerance = 1e-6;

  // Stopping tolerance for the relative decrease in function value between
  // subsequent accepted iterations
  Scalar relative_decrease_tolerance = 1e-6;

  // Stopping tolerance for the norm of the accepted stepsize; terminate if the
  // accepted step h has norm less than this value
  Scalar stepsize_tolerance = 1e-6;
};

/** A template struct containing the output of a Riemannian optimization method
 */
template <typename Variable, typename Scalar = double>
struct SmoothOptimizerResult : public OptimizerResult<Variable, Scalar> {

  // The norm of the gradient at the returned estimate
  Scalar grad_f_x_norm;

  // The norm of the gradient at the *start* of each iteration
  std::vector<Scalar> gradient_norms;

  // The norm of the update step computed during each iteration
  std::vector<Scalar> update_step_norm;
};

/** These next typedefs provide convenient specializations of the above
 * concepts for the (common) use case of optimization over Euclidean spaces,
 * making use of the (standard) identification of the tangent space T_x(R^n) at
 * a point x in R^n with the space R^n itself (i.e., we make use of a *single*
 * type Vector to model both elements x of the linear space R^n and tangent
 * vectors v).  It is assumed that objects of type Vector implement the standard
 * vector space operations, including vector addition, subtraction,
 * left-multiplication by scalars, and evaluation of the standard Euclidean
 * inner product by means of a method with the signature
 * dot(Vector& v2) const.
 */

template <typename Vector, typename... Args>
using EuclideanVectorField = VectorField<Vector, Vector, Args...>;

template <typename Vector, typename... Args>
using EuclideanLinearOperator = LinearOperator<Vector, Vector, Args...>;

template <typename Vector, typename... Args>
using EuclideanLinearOperatorConstructor =
    LinearOperatorConstructor<Vector, Vector, Args...>;

template <typename Vector, typename... Args>
using EuclideanQuadraticModel = QuadraticModel<Vector, Vector, Args...>;

template <typename Vector, typename Scalar = double, typename... Args>
Scalar EuclideanMetric(const Vector &X, const Vector &V1, const Vector &V2,
                       Args &... args) {
  return V1.dot(V2);
}

template <typename Vector, typename... Args>
Vector EuclideanRetraction(const Vector &X, const Vector &V, Args &... args) {
  return X + V;
}

} // namespace Smooth
} // namespace Optimization
