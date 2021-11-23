#pragma once

#include <RcppArmadillo.h>

template<typename T>
arma::uvec
screenPredictors(const std::string screening_type,
                 const arma::uvec& strong,
                 const arma::uvec& ever_active,
                 const arma::vec& residual,
                 const arma::vec& c,
                 const arma::vec& c_grad,
                 const T& X,
                 const arma::vec& X_norms_squared,
                 const arma::vec& X_mean_scaled,
                 const arma::vec& y,
                 const double lambda,
                 const double lambda_next,
                 const double gamma,
                 const bool standardize)
{
  using namespace arma;

  uvec screened(X.n_cols);

  if (screening_type == "working" || screening_type == "celer") {
    screened = ever_active;
  } else if (screening_type == "strong") {
    screened = strong;
  } else if (screening_type == "hessian" ||
             screening_type == "hessian_adaptive") {
    vec c_pred = c + c_grad * (lambda_next - lambda);
    screened = (abs(c_pred) + gamma * (lambda - lambda_next) > lambda_next) ||
               ever_active;
  } else if (screening_type == "gap_safe") {
    // we use the active set strategy for the gap safe rules, so we use the
    // ever-active predictors to get a good warm start
    screened = ever_active;
  } else if (screening_type == "edpp") {
    double dual_scale = std::max(lambda, max(abs(c)));
    vec v1 = y / lambda - residual / dual_scale;
    vec v2 = y / lambda_next - residual / dual_scale;

    double norm_v1 = std::pow(norm(v1), 2);
    vec v_orth = norm_v1 != 0 ? v2 - v1 * dot(v1, v2) / norm_v1 : v2;
    vec center = residual / dual_scale + 0.5 * v_orth;
    double r_screen = 0.5 * norm(v_orth);

    vec XTcenter = matTransposeMultiply(X, center, X_mean_scaled, standardize);

    screened = r_screen * sqrt(X_norms_squared) + abs(XTcenter) +
                 std::sqrt(datum::eps) >=
               1;
  }

  return screened;
}
