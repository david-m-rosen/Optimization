/** This header file provides a lightweight extension of the std::pair template
 * class to model elements of a product of inner-product spaces
 *
 * Copyright(C) 2020 by David M.Rosen(dmrosen @mit.edu) *
 */

#pragma once
#include <cmath>
#include <utility>

namespace Optimization {
namespace Constrained {

template <typename X, typename Y> struct Pair : std::pair<X, Y> {

  typedef std::pair<X, Y> Super;

  /// This struct has no data beyond that held by the superclass std::pair

  /// CONSTRUCTORS

  /** Basic constructor: constructs an empty Pair */
  Pair() : std::pair<X, Y>() {}

  /// Basic constructor: constructs a Pair
  Pair(const X &x, const Y &y) : Super(x, y) {}

  /// Constructor promotes a std::pair to a Pair
  Pair(const std::pair<X, Y> &p) : Super(p) {}

  /// LINEAR ALGEBRAIC MEMBER FUNCTIONS

  // Compute a sum of Pairs
  Pair<X, Y> operator+(const Pair<X, Y> &v) const {
    Pair<X, Y> sum;

    if (dim(Super::first) > 0)
      sum.first = Super::first + v.first;

    if (dim(Super::second) > 0)
      sum.second = Super::second + v.second;

    return sum;
  }

  const Pair<X, Y> &operator+=(const Pair<X, Y> &v) {

    if (dim(Super::first) > 0)
      Super::first += v.first;

    if (dim(Super::second) > 0)
      Super::second += v.second;

    return *this;
  }

  // Compute a difference of Pairs
  Pair<X, Y> operator-(const Pair<X, Y> &v) const {
    Pair<X, Y> sum;

    if (dim(Super::first) > 0)
      sum.first = Super::first - v.first;

    if (dim(Super::second) > 0)
      sum.second = Super::second - v.second;

    return sum;
  }

  const Pair<X, Y> &operator-=(const Pair<X, Y> &v) {

    if (dim(Super::first) > 0)
      Super::first -= v.first;

    if (dim(Super::second) > 0)
      Super::second -= v.second;

    return *this;
  }

  // In-place scalar multiplication
  template <typename Scalar = double> const Pair<X, Y> &operator*=(Scalar s) {

    if (dim(Super::first) > 0)
      Super::first *= s;
    if (dim(Super::second) > 0)
      Super::second *= s;

    return *this;
  }

  // Unary negation operator
  Pair<X, Y> operator-() const {
    return Pair<X, Y>(dim(Super::first) > 0 ? -Super::first : Super::first,
                      dim(Super::second) > 0 ? -Super::second : Super::second);
  }

  // Inner product operator
  template <typename Scalar = double> Scalar dot(const Pair<X, Y> &v) const {
    return (dim(Super::first) > 0 ? inner_product(Super::first, v.first)
                                  : Scalar(0)) +
           (dim(Super::second) > 0 ? inner_product(Super::second, v.second)
                                   : Scalar(0));
  }

  template <typename Scalar = double> Scalar norm() const {
    return sqrt(dot(*this));
  }
};

/// Externally-defined operations on Pairs

/// Scalar multiplication
template <typename X, typename Y, typename Scalar = double>
Pair<X, Y> operator*(Scalar s, const Pair<X, Y> &x) {
  Pair<X, Y> prod;

  if (dim(x.first) > 0)
    prod.first = s * x.first;

  if (dim(x.second) > 0)
    prod.second = s * x.second;

  return prod;
}

} // namespace Constrained
} // namespace Optimization
