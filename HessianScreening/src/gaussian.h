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

  double scaledDual(const double lambda, const double dual_scale)
  {
    if (dual_scale == 0) {
      return 0;
    } else {
      double alpha = lambda / dual_scale;

      return alpha * dot(residual, y) - 0.5 * (alpha * squaredNorm(residual));
    }
  }

  double deviance() { return std::pow(norm(residual), 2); }

  virtual double hessianTerm(const mat& X, const uword j)
  {
    return X_norms_squared(j);
  }

  virtual double hessianTerm(const sp_mat& X, const uword j)
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
    mat H = conv_to<mat>::from(X.cols(ind_a) * X.cols(ind_b).t()).t();

    if (standardize)
      H -= X.n_rows * X_mean_scaled(ind_a) * X_mean_scaled(ind_b).t();

    return H;
  }

  void updateGradientOfCorrelation(vec& c_grad,
                                   const mat& X,
                                   const vec& Hinv_s,
                                   const vec& s,
                                   const uvec& active_set,
                                   const uvec& inactive_set,
                                   const uvec& restricted_set)
  {
    uvec inactive_restricted = intersect(inactive_set, restricted_set);
    uvec inactive_notrestricted = setDiff(inactive_set, restricted_set);
    vec tmp = (X.cols(active_set) * Hinv_s);
    c_grad(inactive_restricted) =
      tmp.t() * X.cols(inactive_restricted);
    c_grad(inactive_notrestricted).zeros();
    c_grad(active_set) = s(active_set);
  }

  void updateGradientOfCorrelation(vec& c_grad,
                                   const sp_mat& X,
                                   const vec& Hinv_s,
                                   const vec& s,
                                   const uvec& active_set,
                                   const uvec& inactive_set,
                                   const uvec& restricted_set)
  {
    uvec inactive_restricted = intersect(inactive_set, restricted_set);
    uvec inactive_notrestricted = setDiff(inactive_set, restricted_set);

    if (standardize) {
      vec tmp =
        X.cols(active_set) * Hinv_s - dot(X_mean_scaled(active_set), Hinv_s);
      c_grad(inactive_restricted) =
        X.cols(inactive_restricted) * tmp.t();
        c_grad(inactive_restricted) -=
        X_mean_scaled(inactive_restricted) * sum(tmp);

    } else {
      c_grad(inactive_restricted) =
        X.cols(inactive_restricted) * (X.cols(active_set) * Hinv_s).t();
    }

    c_grad(inactive_notrestricted).zeros();
    c_grad(active_set) = s(active_set);
  }

  void standardizeY() { y -= mean(y); }
};
