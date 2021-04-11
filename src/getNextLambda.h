#pragma once

#include <RcppArmadillo.h>

using namespace arma;

struct GetNextLambda
{
  const vec& beta;
  const vec& c;
  const vec& c_grad;
  const vec& lambda_grid;
  const double lambda_min;
  const double lambda_min_step;
  const std::string screening_type;
  uword n_target_nonzero;
  const uword verbosity;

  double lambda_next_mod{ 1 };

  GetNextLambda(const vec& beta,
                const vec& c,
                const vec& c_grad,
                const vec& lambda_grid,
                const double lambda_min,
                const double lambda_min_step,
                const std::string screening_type,
                const uword n_target_nonzero,
                const uword verbosity)
    : beta(beta)
    , c(c)
    , c_grad(c_grad)
    , lambda_grid(lambda_grid)
    , lambda_min{ lambda_min }
    , lambda_min_step{ lambda_min_step }
    , screening_type{ screening_type }
    , n_target_nonzero{ n_target_nonzero }
    , verbosity{ verbosity } {};

  double operator()(const vec& Hinv_s,
                    const double lambda,
                    const uvec& active,
                    const uword n_new_active,
                    const uword step_number)
  {
    if (screening_type != "hessian_adaptive") {
      return lambda_grid(step_number);
    }

    uvec active_set = find(active);
    uvec inactive_set = find(active == false);

    if (inactive_set.is_empty()) {
      return lambda_min;
    }

    if (step_number > 1) {
      if (n_new_active > n_target_nonzero) {
        lambda_next_mod *= 1.01;
      } else if (n_new_active < n_target_nonzero) {
        lambda_next_mod /= 1.01;
      }

      n_target_nonzero = std::min(n_target_nonzero, inactive_set.n_elem);

      if (verbosity >= 1) {
        Rprintf("  lambda_next_mod: %f\n", lambda_next_mod);
      }
    }

    // Estimate λ for when predictors are estimated to enter the model
    const vec c_inac = c(inactive_set);
    const vec c_grad_inac = c_grad(inactive_set);

    vec a = (c_inac - lambda * c_grad_inac) / (1 - c_grad_inac);
    vec b = (lambda * c_grad_inac - c_inac) / (1 + c_grad_inac);

    a(find(a >= lambda)).zeros();
    b(find(b >= lambda)).zeros();

    // vec lambda_pred = sort(max(a, b), "descend");
    vec lambda_pred = max(a, b);

    // Estimate λ for when predictors are estimated to leave the model
    vec d = (beta(active_set) + lambda) / Hinv_s;
    d(find(d >= lambda)).zeros();

    lambda_pred = sort(join_vert(lambda_pred, d), "descend");

    double lambda_next = lambda_pred(n_target_nonzero - 1);

    lambda_next += std::sqrt(datum::eps);

    lambda_next *= lambda_next_mod;

    if (lambda - lambda_next < lambda_min_step || lambda - lambda_next < 0) {
      lambda_next = std::max(lambda - lambda_min_step, lambda_min);
    }

    return lambda_next;
  }
};
