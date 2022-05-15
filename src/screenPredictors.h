#pragma once

#include <RcppArmadillo.h>

using namespace arma;

template<typename T>
uvec
screenPredictors(const std::string screening_type,
                 const uvec& strong,
                 const uvec& ever_active,
                 const vec& residual,
                 const vec& c,
                 const vec& c_grad,
                 const T& X,
                 const vec& X_norms,
                 const vec& X_offset,
                 const vec& y,
                 const double lambda,
                 const double lambda_next,
                 const double gamma,
                 const bool standardize)
{
  uvec screened(X.n_cols);

  if (screening_type == "working") {
    screened = ever_active;
  } else if (screening_type == "strong") {
    screened = strong;
  } else if (screening_type == "hessian") {
    vec c_pred = c + c_grad * (lambda_next - lambda);
    screened = (abs(c_pred) + gamma * (lambda - lambda_next) > lambda_next) ||
               ever_active;
  } else if (screening_type == "gap_safe" || screening_type == "sasvi") {
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

    vec XTcenter = matTransposeMultiply(X, center, X_offset, standardize);

    screened = r_screen * X_norms + abs(XTcenter) + datum::eps >= 1;
  }

  return screened;
}
