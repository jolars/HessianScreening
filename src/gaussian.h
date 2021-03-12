#pragma once

#include <RcppArmadillo.h>

#include "prox.h"
#include "utils.h"

using namespace arma;

class Gaussian : public Model
{
public:
  Gaussian(vec& y,
           vec& beta,
           vec& residual,
           vec& Xbeta,
           vec& c,
           const vec& X_mean_scaled,
           const vec& X_norms_squared,
           const uword n,
           const uword p,
           const bool standardize)
    : Model{ y, beta, residual,   Xbeta, c, X_mean_scaled, X_norms_squared,
             n, p,    standardize }
  {}

  double primal(const double lambda, const uvec& screened_set)
  {
    return 0.5 * std::pow(norm(residual), 2) +
           lambda * norm(beta(screened_set), 1);
  }

  double dual() { return dot(residual, y) - 0.5 * std::pow(norm(residual), 2); }

  double scaledDual(const double lambda)
  {
    if (dual_scale == 0) {
      return 0;
    } else {
      double alpha = lambda / dual_scale;

      return alpha * dot(residual, y) - 0.5 * (alpha * squaredNorm(residual));
    }
  }

  double deviance() { return std::pow(norm(residual), 2); }

  double hessianTerm(const mat& X, const uword j) { return X_norms_squared(j); }

  double hessianTerm(const sp_mat& X, const uword j)
  {
    return X_norms_squared(j);
  }

  void updateResidual() { residual = y - Xbeta; }

  void adjustResidual(const mat& X, const uword j, const double beta_diff)
  {
    residual -= X.col(j) * beta_diff;
  }

  void adjustResidual(const sp_mat& X, const uword j, const double beta_diff)
  {
    residual -= X.col(j) * beta_diff;

    if (standardize)
      residual += X_mean_scaled(j) * beta_diff;
  }

  mat hessian(const mat& X, const uvec& ind)
  {
    return X.cols(ind).t() * X.cols(ind);
  }

  mat hessian(const sp_mat& X, const uvec& ind)
  {
    mat H = conv_to<mat>::from(X.cols(ind).t() * X.cols(ind));

    if (standardize)
      H -= X.n_rows * X_mean_scaled(ind) * X_mean_scaled(ind).t();

    return H;
  }

  mat hessianUpperRight(const mat& X, const uvec& ind_a, const uvec& ind_b)
  {
    return X.cols(ind_a).t() * X.cols(ind_b);
  }

  mat hessianUpperRight(const sp_mat& X, const uvec& ind_a, const uvec& ind_b)
  {
    mat H(ind_a.n_elem, ind_b.n_elem);

    if (ind_b.n_elem == 1) {
      uword i = 0;
      for (auto&& j : ind_a) {
        H(i, 0) = dot(X.col(j), X.col(as_scalar(ind_b)));
        i++;
      }
    } else {
      H = conv_to<mat>::from(X.cols(ind_a).t() * X.cols(ind_b));
    }

    if (standardize)
      H -= X.n_rows * X_mean_scaled(ind_a) * X_mean_scaled(ind_b).t();

    return H;
  }

  void updateGradientOfCorrelation(vec& c_grad,
                                   const mat& X,
                                   const vec& Hinv_s,
                                   const vec& s,
                                   const uvec& active_set,
                                   const uvec& restricted_set)
  {
    uvec inactive_restricted = setDiff(restricted_set, active_set);

    const vec tmp = X.cols(active_set) * Hinv_s;

    c_grad.zeros();

    c_grad(inactive_restricted) = tmp.t() * X.cols(inactive_restricted);
    c_grad(active_set) = s(active_set);
  }

  void updateGradientOfCorrelation(vec& c_grad,
                                   const sp_mat& X,
                                   const vec& Hinv_s,
                                   const vec& s,
                                   const uvec& active_set,
                                   const uvec& restricted_set)
  {
    uvec inactive_restricted = setDiff(restricted_set, active_set);

    c_grad.zeros();

    if (standardize) {
      vec tmp =
        X.cols(active_set) * Hinv_s - dot(X_mean_scaled(active_set), Hinv_s);

      double tmp_sum = sum(tmp);

      for (auto&& j : inactive_restricted) {
        c_grad(j) = dot(X.col(j), tmp) - X_mean_scaled(j) * tmp_sum;
      }
    } else {
      vec tmp = X.cols(active_set) * Hinv_s;

      for (auto&& j : inactive_restricted) {
        c_grad(j) = dot(X.col(j), tmp);
      }
    }

    c_grad(active_set) = s(active_set);
  }

  void standardizeY() { y -= mean(y); }

  double safeScreeningRadius(const double duality_gap, const double lambda)
  {
    return std::sqrt(duality_gap) / lambda;
  }
};
