#pragma once

#include <RcppArmadillo.h>

#include "prox.h"
#include "utils.h"

using namespace arma;

class Binomial : public Model
{
public:
  vec expXbeta;
  vec pr;
  vec w;

  const bool approx_hessian;

  const double p_min = 1e-9;
  const double p_max = 1 - p_min;

  Binomial(vec& y,
           vec& beta,
           vec& residual,
           vec& Xbeta,
           vec& c,
           const vec& X_mean_scaled,
           const vec& X_norms_squared,
           const uword n,
           const uword p,
           const bool standardize,
           const bool approx_hessian)
    : Model{ y, beta, residual,   Xbeta, c, X_mean_scaled, X_norms_squared,
             n, p,    standardize }
    , expXbeta(y.n_elem, fill::zeros)
    , pr(y.n_elem, fill::zeros)
    , w(y.n_elem, fill::zeros)
    , approx_hessian{ approx_hessian }
  {}

  double primal(const double lambda, const uvec& screened_set)
  {
    return -sum(y % Xbeta - log1p(expXbeta)) +
           lambda * norm(beta(screened_set), 1);
  }

  double dual() { return -sum(pr % log(pr) + (1 - pr) % log(1 - pr)); }

  double scaledDual(const double lambda)
  {
    if (dual_scale == 0) {
      return 0;
    } else {
      double alpha = lambda / dual_scale;

      vec prx = clamp(y - alpha * residual, p_min, p_max);

      return -sum(prx % log(prx) + (1 - prx) % log(1 - prx));
    }
  }

  double deviance() { return -2 * sum(y % Xbeta - log1p(expXbeta)); }

  double hessianTerm(const mat& X, const uword j)
  {
    return std::max(dot(square(X.col(j)), w), std::sqrt(datum::eps));
  }

  double hessianTerm(const sp_mat& X, const uword j)
  {
    double out = dot(square(X.col(j)), w);

    if (standardize) {
      out += std::pow(X_mean_scaled(j), 2) * sum(w) -
             2 * dot(X.col(j), w) * X_mean_scaled(j);
    }

    return std::max(out, std::sqrt(datum::eps));
  }

  void updateResidual()
  {
    expXbeta = exp(Xbeta);
    pr = clamp(expXbeta / (1 + expXbeta), p_min, p_max);
    w = pr % (1 - pr);
    residual = y - pr;
  }

  void adjustResidual(const mat& X, const uword j, const double beta_diff)
  {
    Xbeta += X.col(j) * beta_diff;
    expXbeta = exp(Xbeta);
    pr = clamp(expXbeta / (1 + expXbeta), p_min, p_max);
    w = pr % (1 - pr);
    residual = y - pr;
  }

  void adjustResidual(const sp_mat& X, const uword j, const double beta_diff)
  {
    Xbeta += X.col(j) * beta_diff;

    if (standardize)
      Xbeta -= X_mean_scaled(j) * beta_diff;

    expXbeta = exp(Xbeta);
    pr = clamp(expXbeta / (1 + expXbeta), p_min, p_max);
    w = pr % (1 - pr);
    residual = y - pr;
  }

  mat hessian(const mat& X, const uvec& ind)
  {
    if (approx_hessian) {
      return 0.25 * X.cols(ind).t() * X.cols(ind);
    } else {
      return X.cols(ind).t() * diagmat(w) * X.cols(ind);
    }
  }

  mat hessian(const sp_mat& X, const uvec& ind)
  {
    if (approx_hessian) {
      mat H = conv_to<mat>::from(X.cols(ind).t() * X.cols(ind));

      if (standardize)
        H -= X.n_rows * X_mean_scaled(ind) * X_mean_scaled(ind).t();

      H *= 0.25;

      return H;
    } else {
      mat H = conv_to<mat>::from(X.cols(ind).t() * diagmat(w) * X.cols(ind));

      if (standardize) {
        mat XmDX = X_mean_scaled(ind) * sum(diagmat(w) * X.cols(ind), 0);
        H += sum(w) * X_mean_scaled(ind) * X_mean_scaled(ind).t() - XmDX -
             XmDX.t();
      }

      return H;
    }
  }

  mat hessianUpperRight(const mat& X, const uvec& ind_a, const uvec& ind_b)
  {
    if (approx_hessian) {
      return 0.25 * X.cols(ind_a).t() * X.cols(ind_b);
    } else {
      return X.cols(ind_a).t() * diagmat(w) * X.cols(ind_b);
    }
  }

  mat hessianUpperRight(const sp_mat& X, const uvec& ind_a, const uvec& ind_b)
  {
    if (approx_hessian) {
      mat H = conv_to<mat>::from(X.cols(ind_a).t() * X.cols(ind_b));

      if (standardize)
        H -= X.n_rows * X_mean_scaled(ind_a) * X_mean_scaled(ind_b).t();

      H *= 0.25;

      return H;

    } else {
      mat H =
        conv_to<mat>::from(X.cols(ind_a).t() * diagmat(w) * X.cols(ind_b));

      if (standardize) {
        mat XamDXb = X_mean_scaled(ind_a) * sum(diagmat(w) * X.cols(ind_b));
        mat XbmDXa = X_mean_scaled(ind_b) * sum(diagmat(w) * X.cols(ind_a));
        H += sum(w) * X_mean_scaled(ind_a) * X_mean_scaled(ind_b).t() -
             XbmDXa.t() - XamDXb;
      }

      return H;
    }
  }

  void updateGradientOfCorrelation(vec& c_grad,
                                   const mat& X,
                                   const vec& Hinv_s,
                                   const vec& s,
                                   const uvec& active,
                                   const uvec& active_perm,
                                   const uvec& restricted)
  {
    const uvec inactive_restricted = find((active == false) && restricted);

    const vec tmp = w % (X.cols(active_perm) * Hinv_s);

    c_grad.zeros();

#pragma omp parallel for
    for (auto&& j : inactive_restricted) {
      c_grad(j) = dot(X.unsafe_col(j), tmp);
    }

    uvec active_set = find(active);

    c_grad(active_set) = s(active_set);
  }

  void updateGradientOfCorrelation(vec& c_grad,
                                   const sp_mat& X,
                                   const vec& Hinv_s,
                                   const vec& s,
                                   const uvec& active,
                                   const uvec& active_perm,
                                   const uvec& restricted)
  {
    const uvec inactive_restricted = find(active == false && restricted);

    const vec dsq = sqrt(w);
    const mat D = diagmat(w);
    const mat Dsq = diagmat(dsq);

    c_grad.zeros();

    if (standardize) {
      const mat Dsq_X = Dsq * X.cols(inactive_restricted);
      const mat Dsq_Xa = Dsq * X.cols(active_perm);

      const mat dsq_mu = dsq * X_mean_scaled(inactive_restricted).t();
      const mat dsq_mua_Hinv_s =
        dsq * (X_mean_scaled(active_perm).t() * Hinv_s);
      const mat Dsq_Xa_Hinv_s = Dsq_Xa * Hinv_s;

      c_grad(inactive_restricted) =
        Dsq_X.t() * (Dsq_Xa_Hinv_s - dsq_mua_Hinv_s) +
        dsq_mu.t() * (dsq_mua_Hinv_s - Dsq_Xa_Hinv_s);

    } else {
      const vec tmp = w % (X.cols(active_perm) * Hinv_s);

#pragma omp parallel for
      for (auto&& j : inactive_restricted) {
        c_grad(j) = dot(X.col(j), tmp);
      }
    }

    const uvec active_set = find(active);

    c_grad(active_set) = s(active_set);
  }

  void standardizeY() {}

  double safeScreeningRadius(const double duality_gap, const double lambda)
  {
    return std::sqrt(2 * duality_gap) / (2 * lambda);
  }
};
;
