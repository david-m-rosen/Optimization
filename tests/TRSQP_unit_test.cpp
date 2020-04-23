/** This unit test exercises the functionality of the TRSQP nonlinear
 * programming algorithm, applied to solve the following simple quadratic
 * program:
 *
 * min .5x'Bx + g'x
 *
 * s.t. 2*x1 + x2 = 1
 *      x1 >= 0
 *      x2 >= 0
 *
 * where:
 *
 * B = [-0.2500   -3.0311]
 *     [-3.0311    3.2500]
 *
 * g = [ 4.0389]
 *     [-1.0314]
 */

// Add definitions from TRSQPEigenExtensions to the Eigen MatrixBase class
// Note that THIS PREPROCESSOR COMMAND MUST PRECEDE ANY EIGEN HEADERS
#define EIGEN_MATRIXBASE_PLUGIN                                                \
  "Optimization/Constrained/TRSQPEigenExtensions.h"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Sparse>
#include <gtest/gtest.h>

#include "Optimization/Constrained/TRSQP.h"

/// Some useful typedefs

/// Basic data types
typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::SparseMatrix<double> SparseMatrix;

using namespace Optimization;
using namespace Constrained;
typedef Optimization::Constrained::Pair<Vector, Vector> VectorPair;

/// Function types
typedef Optimization::Objective<Vector, Scalar> ObjFunction;
typedef Optimization::Constrained::PairFunction<Vector, Vector, Vector>
    ConstraintFunction;
typedef Optimization::Constrained::VectorFunction<Vector> GradientFunction;
typedef Optimization::Constrained::PairFunction<Vector, SparseMatrix,
                                                SparseMatrix>
    JacobianFunction;
typedef Optimization::Constrained::HessianFunction<Vector, Vector, Vector,
                                                   SparseMatrix>
    HessFunction;

typedef Optimization::Constrained::AugmentedSystemSolverFunction<
    Vector, Vector, Vector, SparseMatrix, SparseMatrix>
    AugmentedSystemFunction;

typedef Optimization::Constrained::KKTSystemSolverFunction<
    Vector, Vector, Vector, SparseMatrix, SparseMatrix, SparseMatrix>
    KKTSystemFunction;

typedef Optimization::Constrained::PrimalDualStrategyFunction<
    Vector, Vector, Vector, SparseMatrix, SparseMatrix, SparseMatrix, Scalar>
    PDStrategyFunction;

class TRSQPUnitTest : public testing::Test {
protected:
  /// Test configuration

  Scalar eps_abs = 1e-6;
  Scalar eps_rel = 1e-6;
  bool verbose = false;

  /// Function handles for tests
  ObjFunction f;
  ConstraintFunction c;
  GradientFunction gradf;
  JacobianFunction A;
  HessFunction H;
  AugmentedSystemFunction augmented_system_solver;
  KKTSystemFunction kkt_system_solver;
  PDStrategyFunction pd_strategy;

  /// Objective data
  SparseMatrix B;
  Vector g;

  /// Constraint data
  // Equality constraint matrix and right-hand side
  SparseMatrix Ae;
  Vector be;

  // Inequality constraint matrix and right-hand side
  SparseMatrix Ai;
  Vector bi;

  /// Variables

  // Primal variables
  Vector x0;

  // Auxiliary slack variables s
  Vector s;

  // Dual variables
  Pair<Vector, Vector> lambda0;

  // Barrier parameter mu
  Scalar mu;

  // Trust-region radius
  Scalar Delta;

  // Trust-region contraction factor for normal step comptutation
  Scalar zeta;

  // Fraction-to-the-boundary parameter for auxiliary slack variables
  Scalar tau;

  // Tangential step parameters
  Scalar kappa_fgr;
  Scalar theta;
  size_t max_iterations;

  virtual void SetUp() {
    /// Objective
    Matrix Bdense(2, 2);
    Bdense(0, 0) = -.25;
    Bdense(0, 1) = -3.0311;
    Bdense(1, 0) = -3.0311;
    Bdense(1, 1) = 3.2500;

    B = Bdense.sparseView();

    g.resize(2);
    g << 4.0389, -1.0314;

    /// Equality constraints
    Ae.resize(1, 2);
    be.resize(1);

    // Equality constraint: 2*x1 + x2 = 1;
    Ae.insert(0, 0) = 2;
    Ae.insert(0, 1) = 1;
    be(0) = 1;

    /// Inequality constraints
    Ai.resize(2, 2);
    bi.resize(2);

    // Constraint -x1 <= 0
    Ai.insert(0, 0) = -1;
    bi(0) = 0;

    // Constraint -x2 <= 0
    Ai.insert(1, 1) = -1;
    bi(1) = 0;

    /// Variables
    x0 = Vector::Random(2);

    lambda0 = Pair<Vector, Vector>(10 * Vector::Random(Ae.rows()).cwiseAbs(),
                                   10 * Vector::Random(Ai.rows()).cwiseAbs());

    // NB: Slacks must ALWAYS be positive!
    s = 10 * Vector::Random(2).cwiseAbs();

    // Set trust-region radius
    Delta = 2.5;

    // Set barrier parameter mu
    mu = 1e-2;

    // Normal step control parameters
    zeta = .8;
    tau = .995;

    // Tangential step control parameters
    kappa_fgr = .1;
    theta = .6;
    max_iterations = 100;

    // Construct augmented system matrix M and its factorization Mfact
    //    Optimization::Constrained::construct_augmented_system_matrix(
    //        std::make_pair(Ae, Ai), s, M, Mfact);

    /// Construct function handles
    // Objective
    f = [this](const Vector &x) -> Scalar {
      return .5 * x.dot(this->B * x) + x.dot(this->g);
    };

    // Gradient of objective
    gradf = [this](const Vector &x) -> Vector { return this->B * x + this->g; };

    // Constraint function
    c = [this](const Vector &x) -> Pair<Vector, Vector> {
      return Pair<Vector, Vector>(this->Ae * x - this->be,
                                  this->Ai * x - this->bi);
    };

    // Constraint Jacobian
    A = [this](const Vector &x) -> Pair<SparseMatrix, SparseMatrix> {
      return Pair<SparseMatrix, SparseMatrix>(this->Ae, this->Ai);
    };

    // Hessian of Lagrangian wrt x
    H = [this](const Vector &x, const Pair<Vector, Vector> &lambda) {
      // This is just the matrix M defining the objective, since we have linear
      // constraints
      return this->B;
    };

    // Augmented linear system solver
    augmented_system_solver =
        [](const Pair<SparseMatrix, SparseMatrix> &Ax, const Vector &s,
           bool new_coefficient_matrix, const Pair<Vector, Vector> &b,
           const Pair<Vector, Vector> &c, Pair<Vector, Vector> &v,
           Pair<Vector, Vector> &y) {
          /// Get problem size

          size_t n = b.first.size();
          size_t me = c.first.size();
          size_t mi = c.second.size();
          size_t m = me + mi;

          size_t D = n + me + 2 * mi;

          /// Construct scaled augmented Jacobian Ahat
          Matrix Ahat = Matrix::Zero(m, n + mi);
          if (me > 0)
            Ahat.topLeftCorner(me, n) = Ax.first;
          if (mi > 0) {
            Ahat.bottomLeftCorner(mi, n) = Ax.second;
            Ahat.bottomRightCorner(mi, mi) = s.asDiagonal();
          }

          /// Construct augmented system matrix M
          Matrix M = Matrix::Zero(D, D);
          M.topLeftCorner(n + mi, n + mi).setIdentity();
          M.topRightCorner(n + mi, m) = Ahat.transpose();
          M.bottomLeftCorner(m, n + mi) = Ahat;

          /// Construct right-hand side vector for augmented linear system
          Vector rhs(D);
          rhs.head(n) = b.first;
          if (mi > 0)
            rhs.segment(n, mi) = b.second;
          if (me > 0)
            rhs.segment(n + mi, me) = c.first;
          if (mi > 0)
            rhs.segment(n + me + mi, mi) = c.second;

          /// SOLVE LINEAR SYSTEM!
          Eigen::FullPivHouseholderQR<Matrix> Mfact(M);
          Vector x = Mfact.solve(rhs);

          /// Extract and return solution components
          v.first = x.head(n);

          if (mi > 0)
            v.second = x.segment(n, mi);
          if (me > 0)
            y.first = x.segment(n + mi, me);
          if (mi > 0)
            y.second = x.segment(n + me + mi, mi);
        };

    // Augmented linear system solver
    kkt_system_solver =
        [](const SparseMatrix &H, const Vector &Sigma,
           const Pair<SparseMatrix, SparseMatrix> &Ax,
           bool new_coefficient_matrix, const Pair<Vector, Vector> &b,
           const Pair<Vector, Vector> &c, Pair<Vector, Vector> &v,
           Pair<Vector, Vector> &y) -> bool {
      /// Get problem size
      size_t n = b.first.size();
      size_t me = c.first.size();
      size_t mi = c.second.size();
      size_t m = me + mi;

      size_t D = n + me + 2 * mi;

      /// Construct augmented Hessian W
      Matrix W = Matrix::Zero(n + mi, n + mi);
      W.topLeftCorner(n, n) = H;
      if (mi > 0)
        W.bottomRightCorner(mi, mi) = Sigma.asDiagonal();

      /// Construct augmented Jacobian Abar
      Matrix Abar = Matrix::Zero(m, n + mi);
      if (me > 0)
        Abar.topLeftCorner(me, n) = Ax.first;
      if (mi > 0) {
        Abar.bottomLeftCorner(mi, n) = Ax.second;
        Abar.bottomRightCorner(mi, mi).setIdentity();
      }

      /// Construct KKT system matrix K

      Matrix K = Matrix::Zero(D, D);
      K.topLeftCorner(n + mi, n + mi) = W;
      K.topRightCorner(n + mi, m) = Abar.transpose();
      K.bottomLeftCorner(m, n + mi) = Abar;

      /// Construct right-hand side vector for augmented linear system
      Vector rhs(D);
      rhs.head(n) = b.first;
      if (mi > 0)
        rhs.segment(n, mi) = b.second;
      if (me > 0)
        rhs.segment(n + mi, me) = c.first;
      if (mi > 0)
        rhs.segment(n + me + mi, mi) = c.second;

      /// SOLVE LINEAR SYSTEM!

      // Compute symmetric eigendecomposition of K
      Eigen::SelfAdjointEigenSolver<Matrix> Kfact(K);

      // Solve linear system: K = V D V' => K^-1 = V D^-1 V'
      // x = V * D^-1 * V'
      Vector x = Kfact.eigenvectors() *
                 Kfact.eigenvalues().cwiseInverse().asDiagonal() *
                 Kfact.eigenvectors().transpose() * rhs;

      /// VERIFY STEP COMPUTATION SUCCEEDED
      if ((K * x - rhs).norm() > 1e-6 * std::max(1.0, rhs.norm()))
        return false;

      /// Extract and return solution components
      v.first = x.head(n);

      if (mi > 0)
        v.second = x.segment(n, mi);
      if (me > 0)
        y.first = x.segment(n + mi, me);
      if (mi > 0)
        y.second = x.segment(n + me + mi, mi);

      return true;
    };

    pd_strategy = [](size_t k, double t, const Vector &x, const Vector &s,
                     const Pair<Vector, Vector> &lambda, Scalar fx,
                     const Vector &gradfx, const SparseMatrix &Hx,
                     const Vector &Sigma, const Pair<Vector, Vector> &cx,
                     const Pair<SparseMatrix, SparseMatrix> &Ax, Scalar mu,
                     Scalar Delta, TRSQPStepType prev_step_type,
                     size_t CG_iters, const Pair<Vector, Vector> &d,
                     bool prev_step_accepted) -> bool {
      // Super simple strategy: always attempt primal-dual step computation
      return true;
    };
  } // SetUp()
};

/// Test computation of Lagrange multiplier updates
TEST_F(TRSQPUnitTest, update_Lagrange_multipliers) {

  /// Compute Lagrange multipliers

  size_t me = Ae.rows();
  size_t mi = Ai.rows();
  size_t m = me + mi;
  size_t n = x0.size();

  Vector gradfx = gradf(x0);

  VectorPair delta_lambda =
      update_Lagrange_multipliers<Vector, Vector, Vector, SparseMatrix,
                                  SparseMatrix, Scalar>(
          gradfx, s, lambda0, std::make_pair(Ae, Ai), mu,
          augmented_system_solver);

  Optimization::Constrained::Pair<Vector, Vector> lambda =
      lambda0 + delta_lambda;

  /// Verify that the computed Lagrange multipliers are in fact minimizers of
  /// the target linear least-squares problem:
  ///
  /// min_lambda L(lambda) := | At*lambda + r |^2
  ///
  /// where r = (gradfx, -mu*e), that minimizes the Euclidean norm of the
  /// perturbed KKT system residuals in eqs. (3.7) and (3.8) from the paper

  size_t D = n + me + 2 * mi;
  Matrix M(D, D);
  Matrix A = M.bottomLeftCorner(m, n + mi);

  Vector r(n + mi);
  r.head(n) = gradfx;
  r.tail(mi) = Vector::Constant(mi, -mu);

  // Compute solution to linear least-squares system by solving the normal
  // equations (3.12)--(3.13) from the paper using Cholesky
  Matrix AAt = A * A.transpose();
  Eigen::LDLT<Matrix> AAtFact(AAt);
  Vector lambda_vec_gt = -AAtFact.solve(A * r);

  // Verify that the gradient of the Lagrange multilier LS objective wrt
  // lambda is zero at this solution
  Vector LM_LS_grad = 2 * A * r + 2 * AAt * lambda_vec_gt;
  EXPECT_LT(LM_LS_grad.norm(), eps_abs);

  // Concatenate updated Lagrange multipliers into a vector
  Vector lambda_vec(m);
  lambda_vec.head(me) = lambda.first;
  lambda_vec.tail(mi) = lambda.second;

  // Verify that lambda_vec is what it should be
  EXPECT_LT((lambda_vec - lambda_vec_gt).norm() / lambda_vec_gt.norm(),
            eps_rel);
}

/// Test computation of normal update step using only the equality
/// constraints, and with the trust-region radius set sufficiently large that
/// the full Newton step is feasible
TEST_F(TRSQPUnitTest, EqualityOnlyFullNormalStep) {

  /// Compute the full Newton step for the equality-only normal subproblem
  /// using the pseudoinverse of Ae

  Matrix Ae_dense(Ae);
  Eigen::CompleteOrthogonalDecomposition<Matrix> cqr(Ae_dense);
  Matrix Ae_dagger = cqr.pseudoInverse();
  Vector hN_gt = -Ae_dagger * be;

  Scalar hN_gt_norm = hN_gt.norm();

  /// Compute normal step for equality-only subproblem using the function

  Pair<Vector, Vector> v = Optimization::Constrained::compute_normal_step<
      Vector, Vector, Vector, SparseMatrix, SparseMatrix, Scalar>(
      std::make_pair(be, Vector()), Vector(),
      std::make_pair(Ae, SparseMatrix()), 2 * hN_gt_norm, zeta, tau,
      augmented_system_solver);

  Vector &vx = v.first;

  // Verify that the returned normal step for the equality-only subproblem is
  // the unconstrained Newton (pseudoinverse solution) step
  EXPECT_LT((vx - hN_gt).norm() / hN_gt.norm(), eps_rel);
}

/// Test computation of normal update step using only the equality
/// constraints, and with the trust-region radius set sufficiently small that
/// only the restricted Cauchy step is feasible
TEST_F(TRSQPUnitTest, EqualityOnlyRestrictedCauchyStep) {

  /// Compute the Cauchy step for the equality-only subproblem
  // Gradient direction vector for tangential subproblem at h = 0
  Vector r = Ae.transpose() * be;
  // Optimal steplength along -g
  Scalar alpha = r.squaredNorm() / (Ae * r).squaredNorm();

  // Full (unrestricted) Cauchy step
  Vector hC = -alpha * r;
  Scalar hC_norm = hC.norm();

  /// Set restricted trust-region radius to be tiny in comparison to |hC|
  Scalar Delta = hC_norm / 100;

  /// Compute restricted Cauchy step
  Vector hDL_gt = (Delta * zeta / hC_norm) * hC;

  /// Compute normal step for equality-only subproblem using the function

  // First, compute the augmented system matrix for equality-only tangential
  // subproblem

  Pair<Vector, Vector> v = Optimization::Constrained::compute_normal_step<
      Vector, Vector, Vector, SparseMatrix, SparseMatrix, Scalar>(
      std::make_pair(be, Vector()), Vector(),
      std::make_pair(Ae, SparseMatrix()), Delta, zeta, tau,
      augmented_system_solver);

  Vector &vx = v.first;

  // Verify that the returned normal step for the equality-only subproblem is
  // the unconstrained Newton (pseudoinverse solution) step
  EXPECT_LT((vx - hDL_gt).norm() / hDL_gt.norm(), eps_rel);
}

/// Test computation of normal update step using only the equality
/// constraints, and with the trust-region radius set such that the
/// interpolated dog-leg step is feasible
TEST_F(TRSQPUnitTest, EqualityOnlyDogLegStep) {

  /// Compute the full Newton step for the equality-only normal subproblem
  /// using the pseudoinverse of Ae

  Matrix Ae_dense(Ae);
  Eigen::CompleteOrthogonalDecomposition<Matrix> cqr(Ae_dense);
  Matrix Ae_dagger = cqr.pseudoInverse();
  Vector hN = -Ae_dagger * be;

  Scalar hN_norm = hN.norm();

  /// Compute the Cauchy step for the equality-only subproblem
  // Gradient direction vector for tangential subproblem at h = 0
  Vector r = Ae.transpose() * be;
  // Optimal steplength along -g
  Scalar alpha = r.squaredNorm() / (Ae * r).squaredNorm();
  // Full (unrestricted) Cauchy step
  Vector hC = -alpha * r;
  Scalar hC_norm = hC.norm();

  /// Set the trust-region radius so that the dog-leg step has a maximal norm
  /// between that of hN and hC

  Scalar Delta = (hC_norm + hN_norm) / (2 * zeta);

  /// Compute normal step for equality-only subproblem using the function

  Pair<Vector, Vector> v = Optimization::Constrained::compute_normal_step<
      Vector, Vector, Vector, SparseMatrix, SparseMatrix, Scalar>(
      std::make_pair(be, Vector()), Vector(),
      std::make_pair(Ae, SparseMatrix()), Delta, zeta, tau,
      augmented_system_solver);

  Vector &vx = v.first;

  /// Verify that the returned normal step for the equality-only subproblem
  /// has the correct norm
  EXPECT_LT(fabs(vx.norm() - zeta * Delta), eps_abs);

  /// Verify that the returned normal step for the equality-only subproblem
  /// achieves at least as much reduction as the scaled Newton step
  Vector hN_scaled = (zeta * Delta / hN_norm) * hN;
  EXPECT_LT((Ae * vx + be).norm(), (Ae * hN_scaled + be).norm() + eps_abs);

  /// Verify that the returned normal step for the equality-only subproblem
  /// achieves at least as much reduction as the interpolated dog-leg step AND
  /// the scaled Newton step
  Scalar DeltaBar = zeta * Delta; // Scaled radius for trust-region subproblem

  Vector d = hN - hC; // Displacement from hN to hC
  Scalar hCd = hC.dot(d);
  Scalar hC2 = hC_norm * hC_norm;
  Scalar d2 = d.squaredNorm();

  Scalar theta =
      (-hCd + sqrt(hCd * hCd + d2 * (DeltaBar * DeltaBar - hC2))) / d2;
  Vector hDL_gt = hC + theta * d;

  EXPECT_LT((Ae * vx + be).norm(), (Ae * hDL_gt + be).norm() + eps_abs);
}

/// Test computation of the normal step
TEST_F(TRSQPUnitTest, NormalStep) {

  /// Compute normal step

  Pair<Vector, Vector> v = Optimization::Constrained::compute_normal_step<
      Vector, Vector, Vector, SparseMatrix, SparseMatrix, Scalar>(
      std::make_pair(be, bi), s, std::make_pair(Ae, Ai), Delta, zeta, tau,
      augmented_system_solver);

  Vector &vx = v.first;
  Vector &vs = v.second;

  /// Verify that the returned normal step successfully decreases the norm of
  /// the constraint violation for the barrier subproblem
  size_t me = be.size();
  size_t mi = bi.size();

  // Original constraint residual
  Vector b0 = Vector(me + mi);
  b0.head(me) = be;
  b0.tail(mi) = bi + s;

  // Constraint residual after update
  Vector bplus = Vector(me + mi);
  bplus.head(me) = be + Ae * vx;
  bplus.tail(mi) = bi + s + Ai * vx + vs;

  EXPECT_LT(bplus.norm(), b0.norm() + eps_abs);

  /// Verify that the returned normal step satisfies the SCALED trust-region
  /// constraint | (vx, S^-1 * vs) | <= zeta*Delta
  Vector s_inv = s.cwiseInverse();
  Vector Sinv_vs = vs.cwiseProduct(s_inv);
  EXPECT_LT((Pair<Vector, Vector>(vx, Sinv_vs).norm()), zeta * Delta);

  /// Verify that the returned update for the auxiliary slack variables
  /// satisfies the fraction-to-the-boundary rule vs >= -tau*s/2
  Vector p = vs + (tau / 2) * s;

  // Note that p should be elementwise nonnegative!
  EXPECT_GT(p.minCoeff(), -eps_abs);
}

/// Test computation of the tangential step
TEST_F(TRSQPUnitTest, ComputeTangentialStep) {

  Vector gradfx = gradf(x0);

  /// Compute Lagrange multipliers

  Pair<Vector, Vector> delta_lambda =
      Optimization::Constrained::update_Lagrange_multipliers<
          Vector, Vector, Vector, SparseMatrix, SparseMatrix, Scalar>(
          gradfx, s, lambda0, std::make_pair(Ae, Ai), mu,
          augmented_system_solver);

  Optimization::Constrained::Pair<Vector, Vector> lambda =
      lambda0 + delta_lambda;

  /// Compute normal step
  Pair<Vector, Vector> v = Optimization::Constrained::compute_normal_step<
      Vector, Vector, Vector, SparseMatrix, SparseMatrix, Scalar>(
      std::make_pair(be, bi), s, std::make_pair(Ae, Ai), Delta, zeta, tau,
      augmented_system_solver);

  /// Construct Hessian of Lagrangian for barrier subproblem
  // Hessian of Lagrangian wrt (x,s)
  SparseMatrix Hx = H(x0, lambda);

  // Diagonal block of Hessian of Lagrangian wrt s
  Vector Sigmas =
      Optimization::Constrained::compute_Sigma(s, lambda.second, mu);

  /// Compute tangential update step
  Scalar update_step_M_norm;
  size_t num_iterations;

  std::pair<Vector, Vector> w =
      Optimization::Constrained::compute_tangential_step<
          Vector, Vector, Vector, SparseMatrix, SparseMatrix, SparseMatrix,
          Scalar>(gradfx, s, std::make_pair(Ae, Ai), Hx, Sigmas, v, mu, Delta,
                  tau, max_iterations, kappa_fgr, theta,
                  augmented_system_solver, update_step_M_norm, num_iterations);

  /// Reconstruct scaled normal and tangential updates
  Pair<Vector, Vector> vtilde(v.first, v.second.cwiseProduct(s.cwiseInverse()));

  Pair<Vector, Vector> wtilde(w.first, w.second.cwiseProduct(s.cwiseInverse()));

  /// Verify that the returned tangential update wtilde satisfies (3.37)
  Scalar vtilde_norm = vtilde.norm();
  Scalar wtilde_norm = wtilde.norm();

  EXPECT_LT(wtilde_norm * wtilde_norm,
            Delta * Delta - vtilde_norm * vtilde_norm + eps_abs);

  /// Verify that the returned tangential update wtilde satisfies
  /// (3.35)--(3.36)
  EXPECT_LT(
      (Pair<Vector, Vector>(Ae * wtilde.first,
                            Ai * wtilde.first + s.cwiseProduct(wtilde.second)))
          .norm(),
      eps_abs);

  /// Verify that the returned tangential update wtilde satisfies the slack
  /// inequality (3.38)

  EXPECT_GT((wtilde.second + Vector::Constant(bi.size(), tau) + vtilde.second)
                .minCoeff(),
            -eps_abs);

  /// Verify that the tangential step decreased the model objective

  Vector SSigmaS = s.array() * Sigmas.array() * s.array();
  Scalar dq = gradfx.dot(wtilde.first) - mu * wtilde.second.sum() +
              wtilde.first.dot(Hx * vtilde.first) +
              wtilde.second.dot(SSigmaS.cwiseProduct(vtilde.second)) +
              .5 * wtilde.first.dot(Hx * wtilde.first) +
              .5 * wtilde.second.dot(SSigmaS.cwiseProduct(wtilde.second));

  EXPECT_LT(dq, eps_abs);
}

TEST_F(TRSQPUnitTest, ComputePrimalDualStep) {

  /// Compute a bunch of data

  Vector gradfx = gradf(x0);
  Pair<Vector, Vector> cx = c(x0);
  Pair<SparseMatrix, SparseMatrix> Ax = A(x0);
  Pair<Vector, Vector> gradLz =
      compute_barrier_subproblem_gradient_of_Lagrangian(gradfx, s, Ax, lambda0,
                                                        mu);

  // Problem dimensions
  size_t n = x0.size();
  size_t me = cx.first.size();
  size_t mi = cx.second.size();
  size_t m = me + mi;

  // Sample a positive-definite Hessian of appropriate dimension
  SparseMatrix HxL = Matrix(Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic>(
                                100 * Vector::Random(n).cwiseAbs()))
                         .sparseView();
  Vector Sigma = compute_Sigma(s, lambda0.second, mu);

  /// Compute normal and tangential components of primal-dual update step
  Pair<Vector, Vector> vz, vlambda, wz, wlambda;
  compute_primal_dual_update_components(cx, s, gradfx, Ax, lambda0, HxL, Sigma,
                                        mu, true, kkt_system_solver, vz,
                                        vlambda, wz, wlambda);

  /// Verify that the primal and dual components of the normal update satisfy
  /// the required equations

  // Check that W*vz + Abar'vlambda = 0
  EXPECT_LT(
      (compute_barrier_subproblem_Hessian_of_Lagrangian_product(HxL, Sigma,
                                                                vz) +
       compute_Abar_transpose_product<Vector, Vector, Vector, SparseMatrix,
                                      SparseMatrix>(Ax, vlambda))
          .norm(),
      eps_abs);

  // Check that Abar*vz + cz = 0
  EXPECT_LT(
      (compute_Abar_product<Vector, Vector, Vector, SparseMatrix, SparseMatrix>(
           Ax, vz) +
       compute_barrier_subproblem_constraint_residuals(cx, s))
          .norm(),
      eps_abs);

  /// Verify that the primal and dual components of the tangential update
  /// satisfy the required equations

  // Check that W*wz + Abar'wlambda + gradLz =  0
  EXPECT_LT(
      (compute_barrier_subproblem_Hessian_of_Lagrangian_product(HxL, Sigma,
                                                                wz) +
       compute_Abar_transpose_product<Vector, Vector, Vector, SparseMatrix,
                                      SparseMatrix>(Ax, wlambda) +
       gradLz)
          .norm(),
      eps_abs);

  // Check that Abar*wz = 0
  EXPECT_LT(
      (compute_Abar_product<Vector, Vector, Vector, SparseMatrix, SparseMatrix>(
           Ax, wz))
          .norm(),
      eps_abs);
}

/// Test complete TRSQP optimization algorithm using ONLY the composite-step
/// trust-region updates
TEST_F(TRSQPUnitTest, TrustRegionOnlyOptimization) {

  // Set up NITRO optimizer
  Optimization::Constrained::TRSQPParams<Scalar> params;
  params.max_iterations = 100;
  params.gradient_tolerance = 1e-6;
  params.infeasibility_tolerance = 1e-6;
  params.complementarity_tolerance = 1e-6;
  params.verbose = verbose;

  /// RUN OPTIMIZER!!!
  Optimization::Constrained::TRSQPResult<Vector, Vector, Vector> result =
      Optimization::Constrained::TRSQP<Vector, Vector, Vector, SparseMatrix,
                                       SparseMatrix, SparseMatrix>(
          f, c, gradf, A, H, augmented_system_solver, x0, params);

  // Extract solution
  const Vector &xstar = result.x;
  const Pair<Vector, Vector> &lambda = result.lambda;

  // Value of constraint functions at solution
  Pair<Vector, Vector> cxstar = c(xstar);

  // Gradient at solution
  Vector gradfxstar = gradf(xstar);

  // Constraint Jacobian at solution
  Pair<SparseMatrix, SparseMatrix> Axstar = A(xstar);

  /// Verify that TRSQP optimizer reported success
  EXPECT_EQ(result.status, Optimization::Constrained::TRSQPStatus::KKT);

  /// Verify that the gradient of the Lagrangian at the primal-dual solution
  /// is less than the required tolerance
  EXPECT_LT(
      (gradfxstar +
       compute_At_product<Vector, Vector, Vector, SparseMatrix, SparseMatrix>(
           Axstar, lambda))
          .norm(),
      params.gradient_tolerance);

  /// Verify that the infeasibility norm satisfies the required tolerance
  EXPECT_LT(compute_constraint_residuals(cxstar).norm(),
            params.infeasibility_tolerance);

  /// Verify that the KKT complementarity residual norm satisfies the required
  /// tolerance
  EXPECT_LT(lambda.second.cwiseProduct(cxstar.second).norm(),
            params.complementarity_tolerance);
}

/// Test complete TRSQP optimization algorithm using ONLY the composite-step
/// trust-region updates
TEST_F(TRSQPUnitTest, HybridPrimalDualAndTrustRegionOptimization) {

  // Set up NITRO optimizer
  Optimization::Constrained::TRSQPParams<Scalar> params;
  params.max_iterations = 100;
  params.gradient_tolerance = 1e-6;
  params.infeasibility_tolerance = 1e-6;
  params.complementarity_tolerance = 1e-6;
  params.verbose = verbose;

  /// RUN OPTIMIZER!!!
  Optimization::Constrained::TRSQPResult<Vector, Vector, Vector> result =
      Optimization::Constrained::TRSQP<Vector, Vector, Vector, SparseMatrix,
                                       SparseMatrix, SparseMatrix>(
          f, c, gradf, A, H, augmented_system_solver, x0, params,
          std::experimental::optional<KKTSystemFunction>(kkt_system_solver),
          std::experimental::optional<PDStrategyFunction>(pd_strategy));

  // Extract solution
  const Vector &xstar = result.x;
  const Pair<Vector, Vector> &lambda = result.lambda;

  // Value of constraint functions at solution
  Pair<Vector, Vector> cxstar = c(xstar);

  // Gradient at solution
  Vector gradfxstar = gradf(xstar);

  // Constraint Jacobian at solution
  Pair<SparseMatrix, SparseMatrix> Axstar = A(xstar);

  /// Verify that TRSQP optimizer reported success
  EXPECT_EQ(result.status, Optimization::Constrained::TRSQPStatus::KKT);

  /// Verify that the gradient of the Lagrangian at the primal-dual solution
  /// is less than the required tolerance
  EXPECT_LT(
      (gradfxstar +
       compute_At_product<Vector, Vector, Vector, SparseMatrix, SparseMatrix>(
           Axstar, lambda))
          .norm(),
      params.gradient_tolerance);

  /// Verify that the infeasibility norm satisfies the required tolerance
  EXPECT_LT(
      Optimization::Constrained::compute_constraint_residuals(cxstar).norm(),
      params.infeasibility_tolerance);

  /// Verify that the KKT complementarity residual norm satisfies the required
  /// tolerance
  EXPECT_LT(lambda.second.cwiseProduct(cxstar.second).norm(),
            params.complementarity_tolerance);
}
