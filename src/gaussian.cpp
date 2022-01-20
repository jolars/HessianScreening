#include "gaussian.h"
#include "prox.h"
#include "utils.h"

Gaussian::Gaussian(const std::string family, const arma::vec& X_norms_squared)
  : Model{ family }
  , X_norms_squared(X_norms_squared)
{}

double
Gaussian::primal(const arma::vec& residual,
                 const arma::vec& Xbeta,
                 const arma::vec& beta,
                 const arma::vec& y,
                 const double lambda)
{
  return 0.5 * std::pow(norm(residual), 2) + lambda * norm(beta, 1);
}

double
Gaussian::primal(const arma::vec& residual,
                 const arma::vec& Xbeta,
                 const arma::vec& beta,
                 const arma::vec& y,
                 const double lambda,
                 const arma::uvec& screened_set)
{
  return 0.5 * std::pow(arma::norm(residual), 2) +
         lambda * arma::norm(beta(screened_set), 1);
}

double
Gaussian::dual(const arma::vec& theta, const arma::vec& y, const double lambda)
{
  // return 0.5 * std::pow(arma::norm(y), 2) -
  //        0.5 * lambda * lambda * std::pow(arma::norm(theta - y / lambda), 2);
  return lambda * arma::dot(theta, y) -
         0.5 * std::pow(lambda * arma::norm(theta), 2);
}

double
Gaussian::deviance(const arma::vec& residual,
                   const arma::vec& Xbeta,
                   const arma::vec& y)
{
  return std::pow(arma::norm(residual), 2);
}

void
Gaussian::updateResidual(arma::vec& residual,
                         const arma::vec& Xbeta,
                         const arma::vec& y)
{
  residual = y - Xbeta;
}

void
Gaussian::adjustResidual(arma::vec& residual,
                         arma::vec& Xbeta,
                         const arma::mat& X,
                         const arma::vec& y,
                         const arma::uword j,
                         const double beta_diff,
                         const arma::vec& X_offset,
                         const bool standardize)
{
  residual -= X.col(j) * beta_diff;
}

void
Gaussian::adjustResidual(arma::vec& residual,
                         arma::vec& Xbeta,
                         const arma::sp_mat& X,
                         const arma::vec& y,
                         const arma::uword j,
                         const double beta_diff,
                         const arma::vec& X_offset,
                         const bool standardize)
{
  residual -= X.col(j) * beta_diff;

  if (standardize)
    residual += X_offset(j) * beta_diff;
}

arma::vec
Gaussian::weights(const arma::vec& residual, const arma::vec& y)
{
  return arma::ones<arma::vec>(y.n_elem);
}

arma::mat
Gaussian::hessian(const arma::mat& X,
                  const arma::uvec& ind,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  return X.cols(ind).t() * X.cols(ind);
}

arma::mat
Gaussian::hessian(const arma::sp_mat& X,
                  const arma::uvec& ind,
                  const arma::vec& X_offset,
                  const bool standardize)
{
  using namespace arma;

  mat H = conv_to<mat>::from(X.cols(ind).t() * X.cols(ind));

  if (standardize)
    H -= X.n_rows * X_offset(ind) * X_offset(ind).t();

  return H;
}

arma::mat
Gaussian::hessianUpperRight(const arma::mat& X,
                            const arma::uvec& ind_a,
                            const arma::uvec& ind_b,
                            const arma::vec& X_offset,
                            const bool standardize)
{
  return X.cols(ind_a).t() * X.cols(ind_b);
}

arma::mat
Gaussian::hessianUpperRight(const arma::sp_mat& X,
                            const arma::uvec& ind_a,
                            const arma::uvec& ind_b,
                            const arma::vec& X_offset,
                            const bool standardize)
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
    H -= X.n_rows * X_offset(ind_a) * X_offset(ind_b).t();

  return H;
}

double
Gaussian::hessianTerm(const arma::mat& X,
                      const arma::uword j,
                      const arma::vec& X_offset,
                      const bool standardize)
{
  return X_norms_squared(j);
}

double
Gaussian::hessianTerm(const arma::sp_mat& X,
                      const arma::uword j,
                      const arma::vec& X_offset,
                      const bool standardize)
{
  return X_norms_squared(j);
}

void
Gaussian::updateGradientOfCorrelation(arma::vec& c_grad,
                                      const arma::mat& X,
                                      const arma::vec& Hinv_s,
                                      const arma::vec& s,
                                      const arma::uvec& active_set,
                                      const arma::uvec& restricted_set,
                                      const arma::vec& X_offset,
                                      const bool standardize)
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
                                      const arma::uvec& restricted_set,
                                      const arma::vec& X_offset,
                                      const bool standardize)
{
  using namespace arma;

  uvec inactive_restricted = setDiff(restricted_set, active_set);

  c_grad.zeros();

  if (standardize) {
    vec tmp = X.cols(active_set) * Hinv_s - dot(X_offset(active_set), Hinv_s);

    double tmp_sum = sum(tmp);

    for (auto&& j : inactive_restricted) {
      c_grad(j) = dot(X.col(j), tmp) - X_offset(j) * tmp_sum;
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
Gaussian::standardizeY(arma::vec& y)
{
  y -= arma::mean(y);
}

double
Gaussian::safeScreeningRadius(const double duality_gap, const double lambda)
{
  return std::sqrt(2 * std::max(duality_gap, 0.0)) / lambda;
}

double
Gaussian::toleranceModifier(const arma::vec& y)
{
  return std::pow(arma::norm(y), 2);
}
