#include "getNextLambda.h"

GetNextLambda::GetNextLambda(const arma::vec& beta,
                             const arma::vec& c,
                             const arma::vec& c_grad,
                             const arma::vec& lambda_grid,
                             const double lambda_min,
                             const double lambda_min_step,
                             const std::string screening_type,
                             const arma::uword n_target_nonzero,
                             const arma::uword verbosity)
  : beta(beta)
  , c(c)
  , c_grad(c_grad)
  , lambda_grid(lambda_grid)
  , lambda_min{ lambda_min }
  , lambda_min_step{ lambda_min_step }
  , screening_type{ screening_type }
  , n_target_nonzero{ n_target_nonzero }
  , n_target_nonzero_mod{ 1 }
  , verbosity{ verbosity } {};

double
GetNextLambda::operator()(const arma::vec& Hinv_s,
                          const double lambda,
                          const arma::uvec& active,
                          const arma::uword n_new_active,
                          const arma::uword step_number)
{
  using namespace arma;

  if (screening_type != "hessian_adaptive") {
    return lambda_grid(step_number);
  }

  uvec active_set = find(active);
  uvec inactive_set = find(active == false);

  if (inactive_set.is_empty()) {
    return lambda_min;
  }

  if (step_number > 1) {
    t =
      (1.0 - eta) * t + eta * std::max(1.0, static_cast<double>(n_new_active)) /
                          std::max(1.0, static_cast<double>(n_target_nonzero));

    n_target_nonzero = std::min(n_target_nonzero, inactive_set.n_elem);

    if (verbosity >= 1) {
      Rprintf("  t: %f\n", t);
      Rprintf("  n_target_nonzero_mod: %f\n", n_target_nonzero_mod);
    }
  }

  // Estimate λ for when predictors are estimated to enter the model
  const vec c_inac = c(inactive_set);
  const vec c_grad_inac = c_grad(inactive_set);

  vec a = (c_inac - lambda * c_grad_inac) / (1 - c_grad_inac);
  vec b = (lambda * c_grad_inac - c_inac) / (1 + c_grad_inac);

  a(find(a >= lambda)).zeros();
  b(find(b >= lambda)).zeros();

  vec lambda_pred = max(a, b);

  // Estimate λ for when predictors are estimated to leave the model
  vec d = (beta(active_set) + lambda) / Hinv_s;
  d(find(d >= lambda)).zeros();

  lambda_pred = sort(join_vert(lambda_pred, d), "descend");

  double lambda_next = lambda_pred(static_cast<int>(n_target_nonzero) - 1);

  lambda_next *= t;

  return lambda_next;
}
