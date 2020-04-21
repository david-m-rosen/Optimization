/** This small header file shows how to employ Eigen's plugin system to add a
 * set of methods to Eigen's base classes that are required to instantiate the
 * TRSQP template functions; see the Eigen file MatrixCwiseBinaryOps.h.
 *
 * Copyright (C) 2020 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

// Return the dimension of *this
inline Index dim() const { return size(); }

// Return the maximum element of *this
inline RealScalar max() const { return this->maxCoeff(); }

// Return matrix whose ith element is max(this_ij, s)
const CwiseBinaryOp<internal::scalar_max_op<Scalar, Scalar>, const Derived,
                    const ConstantReturnType>
max(const Scalar &s) const {
  return this->cwiseMax(s);
}

// Return matrix whose ith element is max(this_ij, other_ij)
template <typename OtherDerived>
const CwiseBinaryOp<internal::scalar_max_op<Scalar, Scalar>, const Derived,
                    const OtherDerived>
max(const MatrixBase<OtherDerived> &other) const {
  return CwiseBinaryOp<internal::scalar_max_op<Scalar, Scalar>, const Derived,
                       const OtherDerived>(derived(), other.derived());
}

// Return the minimum element of *this
inline RealScalar min() const { return this->minCoeff(); }

// Return matrix whose ith element is min(a_ij, s)
const CwiseBinaryOp<internal::scalar_min_op<Scalar, Scalar>, const Derived,
                    const ConstantReturnType>
min(const Scalar &s) const {
  return this->cwiseMin(s);
}

// Return Hadamard (elementwise) product with other
template <typename OtherDerived>
const EIGEN_CWISE_BINARY_RETURN_TYPE(Derived, OtherDerived, product)
    hadamard_product(const MatrixBase<OtherDerived> &other) const {
  return derived().cwiseProduct(other);
}

// Return Hadamard (elementwise) inverse of *this
const CwiseUnaryOp<internal::scalar_inverse_op<Scalar>, const Derived>
hadamard_inverse() const {
  return this->cwiseInverse();
}

//// Return inner product of *this with v
template <typename OtherDerived>
Scalar inner_product(const MatrixBase<OtherDerived> &v) const {
  return this->dot(v);
}

// Return
