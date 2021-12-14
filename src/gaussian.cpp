#include "gaussian.h"
#include "prox.h"
#include "utils.h"

Gaussian::Gaussian(const std::string family,
                   arma::vec& y,
                   arma::vec& beta,
                   arma::vec& residual,
                   arma::vec& Xbeta,
                   arma::vec& c,
                   const arma::vec& X_mean_scaled,
                   const arma::vec& X_norms_squared,
                   const arma::uword n,
                   const arma::uword p,
                   const bool standardize)
  : Model{ family,          y, beta, residual,   Xbeta, c, X_mean_scaled,
           X_norms_squared, n, p,    standardize }
{}

double
Gaussian::primal(const double lambda)
{
  return 0.5 * std::pow(norm(residual), 2) + lambda * norm(beta, 1);
}

double
Gaussian::primal(const double lambda, const arma::uvec& screened_set)
{
  return 0.5 * std::pow(arma::norm(residual), 2) +
         lambda * arma::norm(beta(screened_set), 1);
}

double
Gaussian::dual()
{
  return arma::dot(residual, y) - 0.5 * std::pow(arma::norm(residual), 2);
}

double
Gaussian::scaledDual(const double lambda)
{
  if (dual_scale == 0) {
    return 0;
  } else {
    double alpha = lambda / dual_scale;

    return alpha * arma::dot(residual, y) -
           0.5 * std::pow(alpha * arma::norm(residual), 2);
  }
}

double
Gaussian::deviance()
{
  return std::pow(arma::norm(residual), 2);
}

double
Gaussian::hessianTerm(const arma::mat& X, const arma::uword j)
{
  return X_norms_squared(j);
}

double
Gaussian::hessianTerm(const arma::sp_mat& X, const arma::uword j)
{
  return X_norms_squared(j);
}

void
Gaussian::updateResidual()
{
  residual = y - Xbeta;
}

void
Gaussian::adjustResidual(const arma::mat& X,
                         const arma::uword j,
                         const double beta_diff)
{
  residual -= X.col(j) * beta_diff;
}

void
Gaussian::adjustResidual(const arma::sp_mat& X,
                         const arma::uword j,
                         const double beta_diff)
{
  residual -= X.col(j) * beta_diff;

  if (standardize)
    residual += X_mean_scaled(j) * beta_diff;
}

arma::mat
Gaussian::hessian(const arma::mat& X, const arma::uvec& ind)
{
  return X.cols(ind).t() * X.cols(ind);
}

arma::mat
Gaussian::hessian(const arma::sp_mat& X, const arma::uvec& ind)
{
  using namespace arma;

  mat H = conv_to<mat>::from(X.cols(ind).t() * X.cols(ind));

  if (standardize)
    H -= X.n_rows * X_mean_scaled(ind) * X_mean_scaled(ind).t();

  return H;
}

arma::mat
Gaussian::hessianUpperRight(const arma::mat& X,
                            const arma::uvec& ind_a,
                            const arma::uvec& ind_b)
{
  return X.cols(ind_a).t() * X.cols(ind_b);
}

arma::mat
Gaussian::hessianUpperRight(const arma::sp_mat& X,
                            const arma::uvec& ind_a,
                            const arma::uvec& ind_b)
{
  using namespace arma;

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

void
Gaussian::updateGradientOfCorrelation(arma::vec& c_grad,
                                      const arma::mat& X,
                                      const arma::vec& Hinv_s,
                                      const arma::vec& s,
                                      const arma::uvec& active_set,
                                      const arma::uvec& restricted_set)
{
  using namespace arma;

  uvec inactive_restricted = setDiff(restricted_set, active_set);

  const vec tmp = X.cols(active_set) * Hinv_s;

  c_grad.zeros();

  c_grad(inactive_restricted) = tmp.t() * X.cols(inactive_restricted);
  c_grad(active_set) = s(active_set);
}

void
Gaussian::updateGradientOfCorrelation(arma::vec& c_grad,
                                      const arma::sp_mat& X,
                                      const arma::vec& Hinv_s,
                                      const arma::vec& s,
                                      const arma::uvec& active_set,
                                      const arma::uvec& restricted_set)
{
  using namespace arma;

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

void
Gaussian::standardizeY()
{
  y -= arma::mean(y);
}

double
Gaussian::safeScreeningRadius(const double duality_gap, const double lambda)
{
  return std::sqrt(duality_gap) / lambda;
}
